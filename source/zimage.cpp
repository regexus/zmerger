#include "zimage.hpp"
#include "utilities.hpp"
#include "enums.hpp"
#include "consts.hpp"

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
blend_alpha(float a_alpha, float b_alpha)
{
    return b_alpha + a_alpha*(1 - b_alpha);
}

float
blend_value(float a_value, float b_value, BlendMode mode)
{

    if (mode == BlendMode::NORMAL)
        return b_value;
    
    else if (mode == BlendMode::MULTIPLY)
        return a_value * b_value;
    
    else if (mode == BlendMode::SCREEN)
        return a_value + b_value - a_value * b_value;
    
    else
        throw std::runtime_error("Unknown blending mode!");
}

void
blend_pixel(float a_r, float a_g, float a_b, float a_a, 
            uint16_t b_r, uint16_t b_g, uint16_t b_b, uint16_t b_a, 
            BlendMode mode, cv::Vec<float, 4> & result)
{   
    // Early termination in case of black alpha
    if (b_a == 0)
    {
        result = {a_r, a_g, a_b, a_a};
        return;
    }

    // All computations are done in [0.0, 1.0] range
    float b_r_ = b_r/MAX_16_BIT_VALUE_F;
    float b_g_ = b_g/MAX_16_BIT_VALUE_F;
    float b_b_ = b_b/MAX_16_BIT_VALUE_F;
    float b_a_ = b_a/MAX_16_BIT_VALUE_F;

    float out_alpha = blend_alpha(a_a, b_a_);

    result[0] = (1-b_a_/out_alpha)*a_r + (b_a_/out_alpha)*((1-a_a)*b_r_ + a_a*blend_value(a_r, b_r_, mode));
    result[1] = (1-b_a_/out_alpha)*a_g + (b_a_/out_alpha)*((1-a_a)*b_g_ + a_a*blend_value(a_g, b_g_, mode));
    result[2] = (1-b_a_/out_alpha)*a_b + (b_a_/out_alpha)*((1-a_a)*b_b_ + a_a*blend_value(a_b, b_b_, mode));
    result[3] = out_alpha;
}

// ZImage

ZImage::ZImage(std::string rgba_file_path, std::string z_file_path, BlendMode mode)
: mode(mode)
{
    // Reading and checking the rgba-image
    auto rgba_mat_ = cv::imread(rgba_file_path, cv::IMREAD_UNCHANGED);
    if (rgba_mat_.depth()!=CV_16U && rgba_mat_.depth()!=CV_8U)
    {
        throw std::runtime_error("Unsupported rgba-image format! Please use 8-bit or 16-bit image.");
    }

    if (rgba_mat_.channels()!=3 && rgba_mat_.channels()!=4)
    {
        throw std::runtime_error("Unsupported rgba-image format! The image must have 3 (rgb) or 4 (rgba) channels.");
    }

    // Reading and checking the z-image
    auto z_mat_ = cv::imread(z_file_path, cv::IMREAD_UNCHANGED);

    if (z_mat_.channels()!=1)
    {
        throw std::runtime_error("Unsupported depth-image format! Please use grayscale images.");
    }
    if (z_mat_.depth()!=CV_16U)
    {
        throw std::runtime_error("Unsupported depth-image format! Please use 16-bit images.");
    }

    // Checking the resolution of rgba and z images
    if (rgba_mat_.size()!=z_mat_.size())
    {
        std::cout << rgba_file_path <<"|"<< z_file_path << std::endl;
        std::cout << rgba_mat_.rows <<"|"<< z_mat_.rows <<"|"<< rgba_mat_.cols <<"|"<< z_mat_.cols << std::endl;
        std::string message = ("Error, resolution missmatch found! "
        "RGBA-image must have the same resolution as Z-image. "
        "Aborting merge process...");
        throw std::runtime_error(message);
    }

    // Converting the data to 16 bit rgba if necessary.
    if (rgba_mat_.depth()==CV_8U)
        rgba_mat_.convertTo(rgba_mat_, CV_16U, MAX_16_BIT_VALUE_F/MAX_8_BIT_VALUE_F);
    if (rgba_mat_.channels()==3)
        cv::cvtColor(rgba_mat_, rgba_mat_, cv::COLOR_BGR2BGRA);

    // Saving the member variables
    rgba_mat = cv::Mat_<cv::Vec<uint16_t, 4>>(rgba_mat_);
    z_mat = cv::Mat_<cv::Vec<uint16_t, 1>>(z_mat_);
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

bool
ZImageSet::resolution_check()
{
    for (unsigned short i;i<z_images.size()-1;++i)
    {
        if ((z_images[i].height != z_images[i+1].height) || 
            (z_images[i].width != z_images[i+1].width))
        {
            return false;
        }
    }

    return true;
}

ZImageSet::ZImageSet(unsigned short images_count)
{
    z_images.resize(images_count);
}

cv::Mat_<cv::Vec<uint16_t, 4>>
ZImageSet::merge_images(bool invert_z, cv::Vec<float, 4> background={0, 0, 0, 0})
{
    auto height = z_images[0].height;
    auto width = z_images[0].width;
    cv::Mat_<cv::Vec<float, 4>> result(height, width, background);

    #pragma omp parallel for
    for (int i = 0; i<height; ++i)
    {
        std::vector<unsigned char> sorting_vector(z_images.size());
        std::vector<uint16_t> zvalues(z_images.size());

        for (int j = 0; j<width; ++j)
        {
            // Compute sorting indexes
            for (unsigned char m = 0; m < z_images.size(); ++m)
            {
                zvalues[m] = z_images[m].get_z(i, j);
            }

            // Reset the sorting vector to preserve the order of images
            std::iota(sorting_vector.begin(), sorting_vector.end(), 0);

            if (invert_z)
                std::stable_sort(sorting_vector.begin(), sorting_vector.end(),
                                 [&zvalues](unsigned char a, unsigned char b) { return zvalues[a] > zvalues[b]; });
            else
                std::stable_sort(sorting_vector.begin(), sorting_vector.end(),
                                 [&zvalues](unsigned char a, unsigned char b) { return zvalues[a] < zvalues[b]; });

            // // Blend the images
            for (auto k : sorting_vector)
            {
                blend_pixel(result(i, j)[0], result(i, j)[1], result(i, j)[2], result(i, j)[3],
                            z_images[k].get_r(i, j), z_images[k].get_g(i, j), z_images[k].get_b(i, j), z_images[k].get_a(i, j),
                            z_images[k].get_m(i, j), result(i, j));
            }
        }
    }

    return cv::Mat_<cv::Vec<uint16_t, 4>>(result*MAX_16_BIT_VALUE);
}

void
ZImageSet::expand_z(bool inverted_z)
{
    auto ellipse_kernel = cv::getStructuringElement(
        cv::MorphShapes::MORPH_ELLIPSE, cv::Size(2, 2));
    for (auto &z_image : z_images)
    {
        if (inverted_z)
            cv::erode(z_image.z_mat, z_image.z_mat, ellipse_kernel);
        else
            cv::dilate(z_image.z_mat, z_image.z_mat, ellipse_kernel);
    }
}

