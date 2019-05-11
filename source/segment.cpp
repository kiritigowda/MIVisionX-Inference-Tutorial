#include "segmentation.h"

Classifier::Classifier()
{
    initialized = false;
}


void Classifier::initialize()
{
    int colorPointer = 0;
	int color_slider_max = 100;
	int color_slider = 15;
	double colorValue = 0.0;
    int alpha_slider_max = 100;
	int alpha_slider = 70;
	double alphaValue = 0.7;
    threshold_slider_max = 100;
    threshold_slider = 50;
    thresholdValue = 0.5;
    cv::namedWindow(MIVisionX_LEGEND);
    cvui::init(MIVisionX_LEGEND);
    cv::namedWindow(MIVisionX_DISPLAY, cv::WINDOW_GUI_EXPANDED);
    initialized = true;
}


void Segment::threshold_on_trackbar( int, void* ){
    thresholdValue = (double) threshold_slider/threshold_slider_max ;
    return;
}

void Segment::alpha_on_trackbar( int, void* ){
    alphaValue = (double) alpha_slider/alpha_slider_max ;
    return;
}


void Segment::olor_on_trackbar( int, void* ){
    color_slider = color_slider%10;
    colorValue = (double) color_slider/color_slider_max ;
    if(colorValue <= 0.025) colorPointer = 0;
    else if(colorValue > 0.025 && colorValue <= 0.05) colorPointer = 1;
    else if(colorValue > 0.05 && colorValue <= 0.075) colorPointer = 2;
    else colorPointer = 3;

    // create display legend image
    int fontFace = CV_FONT_HERSHEY_DUPLEX;
    double fontScale = 1;
    int thickness = 1.2;
    cv::Size legendGeometry = cv::Size(325, (20 * 40) + 40);
    Mat legend = Mat::zeros(legendGeometry,CV_8UC3);
    Rect roi = Rect(0,0,325,(20 * 40) + 40);
    legend(roi).setTo(cv::Scalar(255,255,255));
    int l;
    for (l = 0; l < 20; l ++){
        int red, green, blue;
        red = (overlayColors[colorPointer][l][2]) ;
        green = (overlayColors[colorPointer][l][1]) ;
        blue = (overlayColors[colorPointer][l][0]) ;
        std::string className = segmentationClasses[l];
        putText(legend, className, Point(20, (l * 40) + 30), fontFace, fontScale, Scalar::all(0), thickness,8);
        rectangle(legend, Point(225, (l * 40)) , Point(300, (l * 40) + 40), Scalar(red,green,blue),-1);
    }
    cv::imshow("MIVision Image Segmentation", legend);

    return;
}


void Segment::findClassProb(size_t start , size_t end, int width, int height, int numClasses, float* output_layer, float threshold, float* prob, unsigned char* classImg)
{
    for(int c = 0; c < numClasses; c++)
    {
        for(int i = start; i < end; i++)
        {
            if((output_layer[i] >= threshold) && (output_layer[i] > prob[i]))
            {
                prob[i] = output_layer[i];
                classImg[i] = c + 1;
            }
        }
     
        output_layer += (width * height);
    }
}

void Segment::createMask(size_t start , size_t end, int imageWidth, unsigned char* classImg, Mat& maskImage)
{
    Vec3b pix;
    int classId = 0;
    for(int i = start; i < end; i++)
    {
        for(int j = 0; j < imageWidth; j++)
        {
            classId = classImg[(i * imageWidth) + j];
            pix.val[0] = (overlayColors[colorPointer][classId][2]) ;
            pix.val[1] = (overlayColors[colorPointer][classId][1]) ;
            pix.val[2] = (overlayColors[colorPointer][classId][0]) ;
            maskImage.at<Vec3b>(i, j) = pix;
        }
    }
    return;
}

void Segment::getMaskImage(int input_dims[4], float* prob, unsigned char* classImg, float* output_layer, float threshold, cv::Size input_geometry, Mat& maskImage)
{
    int numClasses = input_dims[1];
    int height = input_dims[2];
    int width = input_dims[3];

    int64_t freq = clockFrequency(), t0, t1;
    //int numthreads = std::thread::hardware_concurrency();
    int numthreads = 8;
    size_t start = 0, end = 0, chunk = 0;

    // Initialize buffers
    t0 = clockCounter();
    memset(prob, 0, (width * height * sizeof(float)));
    memset(classImg, 0, (width * height));
    t1 = clockCounter();
    //printf("getMaskImage: memset time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

    // Class ID generation
    t0 = clockCounter();
    // parallel processing
    start = 0;
    end = height*width;
    chunk = (end - start + (numthreads - 1))/numthreads;
    std::thread t[numthreads] ;
    for(int i = 0 ; i < numthreads ; i++ )
    {
        size_t s = start + i * chunk ;
        size_t e = s + chunk ;
        t[i] = std::thread(findClassProb, s, e, width, height, numClasses, output_layer,threshold, prob, classImg) ;
    }
    for(int i = 0 ; i < numthreads ; i++){ t[i].join() ; }
    t1 = clockCounter();
    //printf("getMaskImage: Part 1 time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

    // Mask generation
    t0 = clockCounter();
    // parallel create mask
    start = 0;
    end = input_geometry.height;
    chunk = (end - start + (numthreads - 1))/numthreads;
    std::thread M[numthreads] ;
    for(int i = 0 ; i < numthreads ; i++ )
    {
        size_t s = start + i * chunk ;
        size_t e = s + chunk ;
        M[i] = std::thread(createMask, s, e, input_geometry.width, classImg, std::ref(maskImage)) ;
    }
    for(int i = 0 ; i < numthreads ; i++){ M[i].join() ; }
    t1 = clockCounter();
    //printf("getMaskImage: Part 2 time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

    return;
}

void Segment::processOutput(vx_tensor outputTensor, float* output_layer, int input_dims[4], float* prob, unsigned char* classImg, cv::Size input_geometry, cv::Size output_geometry, 
        cv::Mat& inputImage, cv::Mat& maskImage)
{

	if(!initialized)
    {
        initialize();
    }

    if(!initialized)
    {
        printf("Fail to initialize internal buffer!\n");
        return ;
    }

	int64_t freq = clockFrequency(), t0, t1;
    t0 = clockCounter();
    vx_enum data_type = VX_TYPE_FLOAT32;
    vx_size num_of_dims = 4, dims[4] = { 1, 1, 1, 1 }, stride[4];
    vx_map_id map_id;
    float * ptr;
    vx_size count;
    vx_enum usage = VX_READ_ONLY;
    vxQueryTensor(outputTensor, VX_TENSOR_DATA_TYPE, &data_type, sizeof(data_type));
    vxQueryTensor(outputTensor, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims));
    vxQueryTensor(outputTensor, VX_TENSOR_DIMS, &dims, sizeof(dims[0])*num_of_dims);
    if(data_type != VX_TYPE_FLOAT32) {
        std::cerr << "ERROR: copyTensor() supports only VX_TYPE_FLOAT32: invalid for "  << std::endl;
        return ;
    }
    count = dims[0] * dims[1] * dims[2] * dims[3];
    vx_status status = vxMapTensorPatch(outputTensor, num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST, 0);
    if(status) {
        std::cerr << "ERROR: vxMapTensorPatch() failed for "  << std::endl;
        return ;
    }
    memcpy(output_layer, ptr, (count*sizeof(float)));
    status = vxUnmapTensorPatch(outputTensor, map_id);
    if(status) {
        std::cerr << "ERROR: vxUnmapTensorPatch() failed for "  << std::endl;
        return ;
    }
    t1 = clockCounter();
    //printf("LIVE: Copy Segmentation Output Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

    // process mask img
    t0 = clockCounter();
    float threshold = (float)thresholdValue;
    getMaskImage(input_dims, prob, classImg, output_layer, threshold, input_geometry, maskImage);
    t1 = clockCounter();
    //printf("LIVE: Create Mask Image Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

    // Resize and merge outputs img
    t0 = clockCounter();
    //Mat inputDisplay, maskDisplay;
    //cv::resize(inputImage, inputDisplay, cv::Size(output_geometry.width, output_geometry.height));
    //cv::resize(maskImage, maskDisplay, cv::Size(output_geometry.width, output_geometry.height));
    Mat outputDisplay;
    float alpha = alphaValue, beta = ( 1.0 - alpha );
    cv::addWeighted( inputImage, alpha, maskImage, beta, 0.0, outputDisplay);
    t1 = clockCounter();
    //printf("LIVE: Resize and merge Output Image Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

    // display img time
    t0 = clockCounter();
    cv::imshow("MIVision Image Segmentation - Input Image", inputImage);
    cv::imshow("MIVision Image Segmentation - Mask Image", maskImage);
    cv::imshow("MIVision Image Segmentation - Merged Image", outputDisplay );
    t1 = clockCounter();
    //printf("LIVE: Output Image Display Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

    return;
}