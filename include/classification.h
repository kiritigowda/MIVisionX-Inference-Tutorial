#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "vx_ext_opencv.h"

#if USE_OPENCV_4
using namespace cv;
#define CV_FONT_HERSHEY_SIMPLEX FONT_HERSHEY_SIMPLEX
#define CV_FONT_HERSHEY_DUPLEX FONT_HERSHEY_DUPLEX
#define CV_BGR2RGB COLOR_BGR2RGB
#define CV_DIST_L1 DIST_L1
#define CV_BGR2GRAY COLOR_BGR2GRAY
#define CV_GRAY2RGB COLOR_GRAY2RGB
#define CV_FILLED FILLED
#endif

// Name Output Display Windows
#define MIVisionX_LEGEND    "MIVisionX Image Classification"
#define MIVisionX_DISPLAY   "MIVisionX Image Classification - Display"

struct Classifier
{
	bool initialized;
	int threshold_slider_max;
	int threshold_slider;
	double thresholdValue;

	Classifier();

	void initialize();

	void threshold_on_trackbar( int threshold_slider_max, void* threshold_slider);

	void createLegendImage(std::string modelName, float modelTime_g);

	void visualize(cv::Mat &image, int channels, float *outputBuffer, std::string modelName, std::string labelText[], float modelTime_g);
};