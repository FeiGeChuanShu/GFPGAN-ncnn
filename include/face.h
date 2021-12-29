#ifndef FACE_H
#define FACE_H
#include <float.h>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "net.h"

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float score;
    std::vector<cv::Point2f> pts;
    cv::Mat trans_inv;

};

class Face
{
public:
    Face();
    ~Face();
    int load(const std::string& param_path, const std::string& model_path);
    int detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold = 0.7f, float nms_threshold = 0.3f);
    int align_warp_face(cv::Mat& img, const std::vector<Object>& objects, std::vector<cv::Mat>& trans_matrix_inv, std::vector<cv::Mat>& trans_img);
    void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects);
private:
    ncnn::Net net;

};

#endif // FACE_H
