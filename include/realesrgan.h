#ifndef REALESRGAN_H
#define REALESRGAN_H

#include <vector>
#include <ostream>
#include <random>
#include <chrono>
#include <stdio.h>
#include <fstream>
#include <opencv2/opencv.hpp>
//ncnn
#include "net.h"
#include "cpu.h"
#include "layer.h"

class RealESRGAN
{
public:
    RealESRGAN();
    ~RealESRGAN();

    int load(const std::string& parampath, const std::string& modelpath);
    int tile_process(const cv::Mat& inimage, cv::Mat& outimage);

public:
    int inference(const cv::Mat& in, ncnn::Mat& out, int w, int h);
    int preprocess(const cv::Mat& img, cv::Mat& pad_img, int& img_pad_h, int& img_pad_w);
    cv::Mat to_ocv(const cv::Mat& source, const ncnn::Mat& result);
    // realesrgan parameters
    int scale;
    int tile_size;
    int tile_pad;
    const float norm_vals[3] = { 1 / 255.0f, 1 / 255.0f, 1 / 255.0f };
private:
    ncnn::Net net;
    
};

#endif // REALESRGAN_H
