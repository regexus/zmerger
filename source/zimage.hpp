#pragma once

#include "enums.hpp"
#include "consts.hpp"

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

    uint16_t& get_r(size_t, size_t);
    uint16_t& get_g(size_t, size_t);
    uint16_t& get_b(size_t, size_t);
    uint16_t& get_a(size_t, size_t);
    uint16_t& get_z(size_t, size_t);
    BlendMode get_m(size_t, size_t);
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
