#include "segmentation.h"
// source: adapted from cityscapes-dataset.org
unsigned char overlayColors[20][3] = {
    {200,200,200},      // unclassified
    {128, 64,128},      // road
    {244, 35,232},      // sidewalk
    { 250, 150, 70},    // building
    {102,102,156},      // wall
    {190,153,153},      // fence
    { 0,  0,   0},      // pole
    {250,170, 30},      // traffic light
    {220,220,  0},      // traffic sign
    {0, 255, 0},        // vegetation
    {152,251,152},      // terrain
    { 135,206,250},     // sky
    {220, 20, 60},      // person
    {255,  0,  0},      // rider
    {  0,  0,255},      // car
    {  0,  0, 70},      // truck
    {  0, 60,100},      // bus
    {  0, 80,100},      // train
    {  0,  0,230},      // motorcycle
    {119, 11, 32}       // bicycle
};

// source: adapted from cityscapes-dataset.org
std::string segmentationClasses[20] = {
    "Unclassified",
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle"
};
Segment::Segment()
{
    initialized = false;
}

void Segment::threshold_on_trackbar( int, void* object){
    Segment *mSegment = (Segment *) object;
    mSegment->thresholdValue = (double) mSegment->threshold_slider/mSegment->threshold_slider_max ;
    return;
}

void Segment::alpha_on_trackbar( int, void* object){
    Segment *mSegment = (Segment *) object;
    mSegment->alphaValue = (double) mSegment->alpha_slider/mSegment->alpha_slider_max ;
    return;
}

void Segment::initialize(std::string labelText[])
{
    Segment* mSegment = new Segment;
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
    cv::namedWindow("MIVision Image Segmentation");
    cv::namedWindow("MIVision Image Segmentation - Input Image", cv::WINDOW_GUI_EXPANDED);
    cv::namedWindow("MIVision Image Segmentation - Mask Image",cv::WINDOW_GUI_EXPANDED);
    cv::namedWindow("MIVision Image Segmentation - Merged Image",cv::WINDOW_GUI_EXPANDED);

    //cv::createTrackbar("Color Scheme", "MIVision Image Segmentation", &color_slider, color_slider_max, &Segment::color_on_trackbar);
    cv::createTrackbar("Probability Threshold", "MIVision Image Segmentation", &threshold_slider, threshold_slider_max, &Segment::threshold_on_trackbar);
    cv::createTrackbar("Blend alpha", "MIVision Image Segmentation", &alpha_slider, alpha_slider_max, &Segment::alpha_on_trackbar);

    initialized = true;
    delete mSegment;
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

    return;
}

void Segment::createMask(size_t start , size_t end, int imageWidth, unsigned char* classImg, cv::Mat& maskImage)
{
    cv::Vec3b pix;
    int classId = 0;
    for(int i = start; i < end; i++)
    {
        for(int j = 0; j < imageWidth; j++)
        {
            classId = classImg[(i * imageWidth) + j];
            pix.val[0] = (overlayColors[classId][2]);
            pix.val[1] = (overlayColors[classId][1]);
            pix.val[2] = (overlayColors[classId][0]);
            maskImage.at<cv::Vec3b>(i, j) = pix;
        }
    }
    return;
}

void Segment::getMaskImage(int input_dims[4], float* prob, unsigned char* classImg, float* output_layer, cv::Size input_geometry, cv::Mat& maskImage)
{
    int numClasses = input_dims[1];
    int height = input_dims[2];
    int width = input_dims[3];
    float threshold = (float) this->thresholdValue;
    int numthreads = std::thread::hardware_concurrency();
    size_t start = 0, end = 0, chunk = 0;

    // Initialize buffers
    memset(prob, 0, (width * height * sizeof(float)));
    memset(classImg, 0, (width * height));

    // Class ID generation
    // parallel processing
    start = 0;
    end = height*width;
    chunk = (end - start + (numthreads - 1))/numthreads;
    std::thread t[numthreads] ;
    for(int i = 0 ; i < numthreads ; i++ )
    {
        size_t s = start + i * chunk ;
        size_t e = s + chunk ;
        t[i] = std::thread(&Segment::findClassProb, this, s, e, width, height, numClasses, output_layer,threshold, prob, classImg) ;
    }
    for(int i = 0 ; i < numthreads ; i++){ t[i].join() ; }
    std::cout << "done 1\n";
    // Mask 
    // parallel create mask
    start = 0;
    end = input_geometry.height;
    chunk = (end - start + (numthreads - 1))/numthreads;
    std::thread M[numthreads] ;
    for(int i = 0 ; i < numthreads ; i++ )
    {
        size_t s = start + i * chunk ;
        size_t e = s + chunk ;
        M[i] = std::thread(&Segment::createMask, this, s, e, input_geometry.width, classImg, std::ref(maskImage)) ;
    }

    std::cout << "done 2\n";
    for(int i = 0 ; i < numthreads ; i++){ M[i].join() ; }
    
    return;
}


/*void Segment::processOutput(float* output_layer, int input_dims[4], float* prob, unsigned char* classImg, cv::Size input_geometry, cv::Size output_geometry, 
        cv::Mat& inputImage, cv::Mat& maskImage, std::string labelText[])
{

	if(!initialized)
    {
        initialize(labelText);
    }

    if(!initialized)
    {
        printf("Fail to initialize internal buffer!\n");
        return ;
    }
    std::cout << "initialize done!\n";
    
    std::cout << "step 2 done\n";
    // process mask img
    float threshold = (float)thresholdValue;
    getMaskImage(input_dims, prob, classImg, output_layer, input_geometry, maskImage);
    std::cout << "step 3 done\n";
    // Resize and merge outputs img
    cv::Mat inputDisplay, maskDisplay;
    cv::resize(inputImage, inputDisplay, cv::Size(output_geometry.width, output_geometry.height));
    cv::resize(maskImage, maskDisplay, cv::Size(output_geometry.width, output_geometry.height));
    cv::Mat outputDisplay;
    float alpha = alphaValue, beta = ( 1.0 - alpha );
    cv::addWeighted( inputImage, alpha, maskImage, beta, 0.0, outputDisplay);
    std::cout << "step 4 done\n";
    // display img time
    cv::imshow("MIVision Image Segmentation - Input Image", inputImage);
    cv::imshow("MIVision Image Segmentation - Mask Image", maskImage);
    cv::imshow("MIVision Image Segmentation - Merged Image", outputDisplay );

    return;
}*/