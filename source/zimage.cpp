#include "zimage.hpp"
#include "enums_and_consts.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>

// Helper functions

float
blend_apha(float a_alpha, float b_alpha)
{
    return b_alpha + a_alpha*(1 - b_alpha);
}

float
blend_value(float a_value, float b_value, BlendMode mode)
{
    // Normal mode
    if (mode == BlendMode::NORMAL)
        return b_value;
    // Multiply mode
    else if (mode == BlendMode::MULTIPLY)
        return a_value * b_value;
    // Screen mode
    else if (mode == BlendMode::SCREEN)
        return a_value + b_value - a_value * b_value;
    else
        throw std::runtime_error("Unknown blending mode!");
}

void
blend_pixel(uint16_t a_r, uint16_t a_g, uint16_t a_b, uint16_t a_a, 
            uint16_t b_r, uint16_t b_g, uint16_t b_b, uint16_t b_a, 
            BlendMode mode, cv::Vec<uint16_t, 4> & result)
{   
    // Early termination in case of black alpha
    if (b_a == 0)
    {
        result = {a_r, a_g, a_b, a_a};
        return; 
    }

    float a_r_ = static_cast<float>(a_r)/MAX_16_BIT_VALUE;
    float a_g_ = static_cast<float>(a_g)/MAX_16_BIT_VALUE;
    float a_b_ = static_cast<float>(a_b)/MAX_16_BIT_VALUE;
    float a_a_ = static_cast<float>(a_a)/MAX_16_BIT_VALUE;
    float b_r_ = static_cast<float>(b_r)/MAX_16_BIT_VALUE;
    float b_g_ = static_cast<float>(b_g)/MAX_16_BIT_VALUE;
    float b_b_ = static_cast<float>(b_b)/MAX_16_BIT_VALUE;
    float b_a_ = static_cast<float>(b_a)/MAX_16_BIT_VALUE;

    float out_alpha = blend_apha(a_a_, b_a_);

    result[0] = ((1-b_a_/out_alpha)*a_r_ + (b_a_/out_alpha)*((1-a_a_)*b_r_ + a_a_*blend_value(a_r_, b_r_, mode)))*MAX_16_BIT_VALUE;
    result[1] = ((1-b_a_/out_alpha)*a_g_ + (b_a_/out_alpha)*((1-a_a_)*b_g_ + a_a_*blend_value(a_g_, b_g_, mode)))*MAX_16_BIT_VALUE;
    result[2] = ((1-b_a_/out_alpha)*a_b_ + (b_a_/out_alpha)*((1-a_a_)*b_b_ + a_a_*blend_value(a_b_, b_b_, mode)))*MAX_16_BIT_VALUE;
    result[3] = out_alpha*MAX_16_BIT_VALUE;
}

// ZImage

ZImage::ZImage(std::string rgba_file_path, std::string z_file_path, BlendMode mode)
: mode(mode)
{
    auto rgba_mat_ = cv::imread(rgba_file_path, cv::IMREAD_UNCHANGED);
    if (rgba_mat_.depth()!=CV_16U)
    {
        throw std::runtime_error("Unsupported image format! Please use 16-bit image.");
    }
    if (rgba_mat_.channels()!=3 && rgba_mat_.channels()!=4)
    {
        throw std::runtime_error("Unsupported image format! Please use rgb or rgba image.");
        if (rgba_mat_.channels()==3)
            cv::cvtColor(rgba_mat_, rgba_mat_, cv::COLOR_BGR2BGRA);
    }
    rgba_mat = cv::Mat_<cv::Vec<uint16_t, 4>>(rgba_mat_);

    auto z_mat_ = cv::imread(z_file_path, cv::IMREAD_UNCHANGED);

    if (z_mat_.channels()!=1)
    {
        throw std::runtime_error("Unsupported depth-image format! Please use grayscale images.");
    }
    if (z_mat_.depth()!=CV_16U)
    {
        throw std::runtime_error("Unsupported depth-image format! Please use 16-bit images.");
    }
    z_mat = cv::Mat_<cv::Vec<uint16_t, 1>>(z_mat_);


    if ((rgba_mat.rows!=z_mat.rows) || (rgba_mat.cols!=z_mat.cols))
    {
        std::cout << rgba_file_path <<"|"<< z_file_path <<"|"<< z_mat.cols << std::endl;
        std::cout << rgba_mat.rows <<"|"<< z_mat.rows <<"|"<< rgba_mat.cols <<"|"<< z_mat.cols << std::endl;
        std::string message = "Error, resolution missmatch found! Aborting merge process...";
        throw std::runtime_error(message);
    }

    height = rgba_mat.rows;
    width = rgba_mat.cols;
}

uint16_t& 
ZImage::get_r(size_t i, size_t j)
{
    return rgba_mat(i, j)[0];
}

uint16_t& 
ZImage::get_g(size_t i, size_t j)
{
    return rgba_mat(i, j)[1];
}

uint16_t& 
ZImage::get_b(size_t i, size_t j)
{
    return rgba_mat(i, j)[2];
}

uint16_t& 
ZImage::get_a(size_t i, size_t j)
{
    return rgba_mat(i, j)[3];
}

uint16_t& 
ZImage::get_z(size_t i, size_t j)
{
    return z_mat(i, j)[0];
}

BlendMode
ZImage::get_m(size_t i, size_t j)
{
    return mode;
}

// ZImageSet

ZImageSet::ZImageSet()
{
}

cv::Mat_<cv::Vec<uint16_t, 4>>
ZImageSet::merge_images(bool invert_z, cv::Vec<uint16_t, 4> background={0, 0, 0, MAX_16_BIT_VALUE})
{
    int height = z_images[0].height;
    int width = z_images[0].width;
    cv::Mat_<cv::Vec<uint16_t, 4>> result(height, width, background);

    #pragma omp parallel for
    for (int i = 0; i<height; ++i)
    {
        std::vector<unsigned char> sorting_vector(z_images.size());
        std::iota(sorting_vector.begin(), sorting_vector.end(), 0);
        std::vector<uint16_t> zvalues(z_images.size());

        for (int j = 0; j<width; ++j)
        {
            // Compute sorting indexes
            for (unsigned char m=0;m<z_images.size();++m) {zvalues[m] = z_images[m].get_z(i, j);}

            if (invert_z)
                std::stable_sort(sorting_vector.begin(), sorting_vector.end(), 
                    [&zvalues](unsigned char a, unsigned char b){return zvalues[a]>zvalues[b];}
                );
            else
                std::stable_sort(sorting_vector.begin(), sorting_vector.end(), 
                    [&zvalues](unsigned char a, unsigned char b){return zvalues[a]<zvalues[b];}
                );

            // // Blend the images
            for (auto k : sorting_vector)
            {
                blend_pixel(
                    result(i, j)[0], result(i, j)[1], result(i, j)[2], result(i, j)[3],
                    z_images[k].get_r(i,j), z_images[k].get_g(i,j), z_images[k].get_b(i,j), z_images[k].get_a(i,j),
                    z_images[k].get_m(i,j), result(i, j)
                );
            }
        }
    }

    return result;
}

