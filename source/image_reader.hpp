#pragma once

#include "enums_and_consts.hpp"

// #define DBL_EPSILON 2.2204460492503131e-16
#include <opencv2/core.hpp>

using rgbazm_ = cv::Vec<float, 6>;

class ZMergerImage : public cv::Mat
{
    public:
    
    cv::Mat_<rgbazm_> data;

    ZMergerImage(size_t rows, size_t cols);
    ZMergerImage(std::string rgba_file_path, std::string z_file_path, BlendMode mode);
    ZMergerImage(std::string zmerger_binary_file_path);

    void
    save_as_file(std::string);

    void
    save_as_binary_file(std::string);
};