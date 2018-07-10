#pragma once

#include "consts.hpp"
#include "enums.hpp"

#include <opencv2/core.hpp>

class ZImage
{
    public:

    cv::Mat_<cv::Vec<uint16_t, 4>> rgba_mat;
    cv::Mat_<cv::Vec<uint16_t, 1>> z_mat;
    BlendMode mode = BlendMode::NORMAL;

    size_t width;
    size_t height;

    ZImage(){};

    ZImage(std::string rgba_file_path, std::string z_file_path, BlendMode mode);

    uint16_t& get_r(int, int);
    uint16_t& get_g(int, int);
    uint16_t& get_b(int, int);
    uint16_t& get_a(int, int);
    uint16_t& get_z(int, int);
    BlendMode get_m(int, int);
};

class ZImageSet
{
    public:

    std::vector<ZImage> z_images;
    
    ZImageSet(unsigned short images_count);
    
    bool
    resolution_check();
    
    cv::Mat_<cv::Vec<uint16_t, 4>>
    merge_images(bool invert_z, cv::Vec<float, 4> background);

    void
    expand_z(bool inverted_z);
};
