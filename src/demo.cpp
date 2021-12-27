#include <vector>
#include <ostream>
#include <random>
#include <chrono>
#include <stdio.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "net.h"
#include "cpu.h"
#include "gfpgan.h"
static void draw_result(const cv::Mat& source, const ncnn::Mat& result)
{
    cv::Mat cv_result_32F = cv::Mat::zeros(cv::Size(512, 512), CV_32FC3);
    for (int i = 0; i < result.h; i++)
    {
        for (int j = 0; j < result.w; j++)
        {
            cv_result_32F.at<cv::Vec3f>(i, j)[2] = (result.channel(0)[i * result.w + j] + 1) / 2;
            cv_result_32F.at<cv::Vec3f>(i, j)[1] = (result.channel(1)[i * result.w + j] + 1) / 2;
            cv_result_32F.at<cv::Vec3f>(i, j)[0] = (result.channel(2)[i * result.w + j] + 1) / 2;
        }
    }

    cv::Mat cv_result_8U;
    cv_result_32F.convertTo(cv_result_8U, CV_8UC3, 255.0, 0);

    cv::imshow("source", source);
    cv::imshow("ncnn_result", cv_result_8U);
    cv::waitKey(0);

}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat img = cv::imread(imagepath, 1);
    if (img.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
    
    GFPGAN gfpgan;
    gfpgan.load("./models/encoder.param", "./models/encoder.bin", "./models/style.bin");

    ncnn::Mat ncnn_in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, 512, 512);
    const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
    const float norm_vals[3] = { 1 / 127.5f, 1 / 127.5f, 1 / 127.5f };
    ncnn_in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Mat result;
    gfpgan.process(ncnn_in, result);

    draw_result(img, result);

    return 0;
}