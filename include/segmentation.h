#pragma once

#include <string>
#include <thread>
#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <inttypes.h>
#include <chrono>

#include "vx_ext_opencv.h"



class Segment
{
public:
    bool initialized;
    int threshold_slider_max;
    int threshold_slider;
    double thresholdValue;
    int color_slider_max;
    int color_slider;
    int colorPointer;
    double colorValue;
    int alpha_slider_max;
    int alpha_slider;
    double alphaValue;

    Segment();

    void initialize(std::string labelText[]);

    static void threshold_on_trackbar( int threshold_slider_max, void* threshold_slider);

    static void alpha_on_trackbar( int threshold_slider_max, void* threshold_slider);

    static void color_on_trackbar( int color_slider_max, void* color_slider);

    void findClassProb(size_t start , size_t end, int width, int height, int numClasses, float* output_layer, float threshold, float* prob, unsigned char* classImg);
    //void findClassProb(size_t start , size_t end, int width, int height, int numClasses, float* output_layer, float threshold, float* prob, unsigned char* classImg);

    void createMask(size_t start , size_t end, int imageWidth, unsigned char* classImg, cv::Mat& maskImage);
    //void createMask(size_t start , size_t end, int imageWidth, unsigned char* classImg, cv::Mat& maskImage);
    void getMaskImage(int input_dims[4], float* prob, unsigned char* classImg, float* output_layer, float threshold, cv::Size input_geometry, cv::Mat& maskImage);
    //void getMaskImage(int input_dims[4], float* prob, unsigned char* classImg, float* output_layer, cv::Size input_geometry, cv::Mat& maskImage);

    void processOutput(float* output_layer, int input_dims[4], float* prob, unsigned char* classImg, cv::Size input_geometry, cv::Size output_geometry, 
        cv::Mat& inputImage, cv::Mat& maskImage, std::string labelText[]);

};

