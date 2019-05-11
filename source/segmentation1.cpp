#include "segmentation.h"
// source: adapted from cityscapes-dataset.org
unsigned char overlayColors[4][20][3] = {
    {
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
    },
    {
        {225,225,225},      // unclassified
        {160,82,45},        // road
        {0,128,0},          // sidewalk
        {47,79,79},         // building
        {255,240,245},      // wall
        {190,153,153},      // fence
        {240,255,255},      // pole
        {250,170, 30},      // traffic light
        {255,105,180},      // traffic sign
        {124,252,0},        // vegetation
        {75,0,130},         // terrain
        {0,191,255},        // sky
        {230,230,250},      // person
        {230,230,255},      // rider
        {153,50,204},       // car
        {154,60,210},       // truck
        {155,65,215},       // bus
        {107,142,35},       // train
        {128,0,0},          // motorcycle
        {255,255,0}         // bicycle
    },
    {
        {190,190,190},      // unclassified
        {160,82,45},        // road
        {128,0,0},          // sidewalk
        {47,79,79},         // building
        {255,240,245},      // wall
        {190,153,153},      // fence
        {240,255,255},      // pole
        {250,170, 30},      // traffic light
        {255,105,180},      // traffic sign
        {255,255,0},        // vegetation
        {75,0,130},         // terrain
        {0,191,255},        // sky
        {230,230,250},      // person
        {0,0,255},          // rider
        {153,50,204},       // car
        {0,128,128},        // truck
        {0,255,255},        // bus
        {107,142,35},       // train
        {0,128,0},          // motorcycle
        {124,252,0}         // bicycle
    },
    {
        {0,0,0},            // unclassified
        {128, 64,128},      // road
        {244, 35,232},      // sidewalk
        { 250, 150, 70},    // building
        {102,102,156},      // wall
        {190,153,153},      // fence
        {120,120,120},      // pole
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
    }
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


void Segment::color_on_trackbar( int, void* object){
    Segment* mSegment = (Segment *) object;
    mSegment->color_slider = mSegment->color_slider%10;
    mSegment->colorValue = (double) mSegment->color_slider/mSegment->color_slider_max ;
    if(mSegment->colorValue <= 0.025) mSegment->colorPointer = 0;
    else if(mSegment->colorValue > 0.025 && mSegment->colorValue <= 0.05) mSegment->colorPointer = 1;
    else if(mSegment->colorValue > 0.05 && mSegment->colorValue <= 0.075) mSegment->colorPointer = 2;
    else mSegment->colorPointer = 3;

    // create display legend image
    int fontFace = CV_FONT_HERSHEY_DUPLEX;
    double fontScale = 1;
    int thickness = 1.2;
    cv::Size legendGeometry = cv::Size(325, (20 * 40) + 40);
    cv::Mat legend = cv::Mat::zeros(legendGeometry,CV_8UC3);
    cv::Rect roi = cv::Rect(0,0,325,(20 * 40) + 40);
    legend(roi).setTo(cv::Scalar(255,255,255));
    int l;
    for (l = 0; l < 20; l ++){
        int red, green, blue;
        red = (overlayColors[mSegment->colorPointer][l][2]) ;
        green = (overlayColors[mSegment->colorPointer][l][1]) ;
        blue = (overlayColors[mSegment->colorPointer][l][0]) ;
        std::string className = segmentationClasses[l];
        putText(legend, className, cv::Point(20, (l * 40) + 30), fontFace, fontScale, cv::Scalar::all(0), thickness,8);
        rectangle(legend, cv::Point(225, (l * 40)) , cv::Point(300, (l * 40) + 40), cv::Scalar(red,green,blue),-1);
    }
    cv::imshow("MIVision Image Segmentation", legend);

    return;
}

void Segment::initialize(std::string labelText[])
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
    cv::namedWindow("MIVision Image Segmentation");
    cv::namedWindow("MIVision Image Segmentation - Input Image", cv::WINDOW_GUI_EXPANDED);
    cv::namedWindow("MIVision Image Segmentation - Mask Image",cv::WINDOW_GUI_EXPANDED);
    cv::namedWindow("MIVision Image Segmentation - Merged Image",cv::WINDOW_GUI_EXPANDED);

    cv::createTrackbar("Color Scheme", "MIVision Image Segmentation", &color_slider, color_slider_max, &Segment::color_on_trackbar);
    cv::createTrackbar("Probability Threshold", "MIVision Image Segmentation", &threshold_slider, threshold_slider_max, &Segment::threshold_on_trackbar);
    cv::createTrackbar("Blend alpha", "MIVision Image Segmentation", &alpha_slider, alpha_slider_max, &Segment::alpha_on_trackbar);

    initialized = true;
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

void Segment::createMask(size_t start , size_t end, int imageWidth, unsigned char* classImg, cv::Mat& maskImage)
{
    cv::Vec3b pix;
    int classId = 0;
    for(int i = start; i < end; i++)
    {
        for(int j = 0; j < imageWidth; j++)
        {
            classId = classImg[(i * imageWidth) + j];
            pix.val[0] = (overlayColors[colorPointer][classId][2]) ;
            pix.val[1] = (overlayColors[colorPointer][classId][1]) ;
            pix.val[2] = (overlayColors[colorPointer][classId][0]) ;
            maskImage.at<cv::Vec3b>(i, j) = pix;
        }
    }
    return;
}

void Segment::getMaskImage(int input_dims[4], float* prob, unsigned char* classImg, float* output_layer, float threshold, cv::Size input_geometry, cv::Mat& maskImage)
{
    
    int numClasses = input_dims[1];
    int height = input_dims[2];
    int width = input_dims[3];

    int numthreads = 8;
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
        t[i] = std::thread(&Segment::findClassProb, this, s, e, width, height, numClasses, output_layer,threshold, prob, classImg);
    }
    for(int i = 0 ; i < numthreads ; i++){ t[i].join() ; }
    //printf("getMaskImage: Part 1 time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

    // Mask generation
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
    for(int i = 0 ; i < numthreads ; i++){ M[i].join() ; }
        
    //printf("getMaskImage: Part 2 time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);
    return;
}

void Segment::processOutput(float* output_layer, int input_dims[4], float* prob, unsigned char* classImg, cv::Size input_geometry, cv::Size output_geometry, 
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
    
    // process mask img
    float threshold = (float)thresholdValue;
    getMaskImage(input_dims, prob, classImg, output_layer, threshold, input_geometry, maskImage);
    // Resize and merge outputs img
    cv::Mat inputDisplay, maskDisplay;
    cv::resize(inputImage, inputDisplay, cv::Size(output_geometry.width, output_geometry.height));
    cv::resize(maskImage, maskDisplay, cv::Size(output_geometry.width, output_geometry.height));
    cv::Mat outputDisplay;
    float alpha = alphaValue, beta = ( 1.0 - alpha );
    cv::addWeighted( inputImage, alpha, maskImage, beta, 0.0, outputDisplay);
    // display img time
    cv::imshow("MIVision Image Segmentation - Input Image", inputImage);
    cv::imshow("MIVision Image Segmentation - Mask Image", maskImage);
    cv::imshow("MIVision Image Segmentation - Merged Image", outputDisplay );

    return;
}