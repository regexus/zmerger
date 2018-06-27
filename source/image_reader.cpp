#include "image_reader.hpp"
#include "utilities.hpp"

#include <opencv2/core/traits.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <stdexcept>

// Reads the image file and converts
// it's values to double in [0, 1] range.
cv::Mat read_and_normalize(std::string image_path)
{
    // auto open_start_time = get_time();
    auto cv_mat = cv::imread(image_path, cv::IMREAD_UNCHANGED);
    // print("Opened the image", image_path, "\nElapsed time:", time_from(open_start_time).count(), "ms.");

    float max_value;
    if (cv_mat.depth()==CV_8U)
        max_value = static_cast<float>(MAX_8_BIT_VALUE);
    else if (cv_mat.depth()==CV_16U)
        max_value = static_cast<float>(MAX_16_BIT_VALUE);
    else
        throw std::runtime_error("Unsupported image format! Please use 8-bit or 16-bit image.");

    cv_mat.convertTo(cv_mat, CV_32FC(cv_mat.channels()), 1/max_value);
    return cv_mat;
}

template <typename RGBA_TYPE>
void fill_data(cv::Mat& rgba_mat, cv::Mat& z_mat, cv::Mat_<rgbazm_>& data, BlendMode mode)
{
    for (size_t i = 0; i<rgba_mat.rows; ++i)
    {
        for (size_t j = 0; j<rgba_mat.cols; ++j)
        {   
            for (size_t k = 0; k<rgba_mat.channels(); ++k)
            {
                data(i, j)[k] = rgba_mat.at<RGBA_TYPE>(i, j)[k];
            }
            if (rgba_mat.channels()==3)
            {
                data(i, j)[3] =  1.0f;
            }
            
            data(i, j)[4] = z_mat.at<float>(i, j);
            data(i, j)[5] = static_cast<float>(mode);
        }
    }
}

ZMergerImage::ZMergerImage(size_t rows, size_t cols)
{
    data = cv::Mat_<rgbazm_>(rows, cols);
}

ZMergerImage::ZMergerImage(std::string rgba_file_path, std::string z_file_path, BlendMode mode)
{

    auto rgba_mat = read_and_normalize(rgba_file_path);
    auto z_mat = read_and_normalize(z_file_path);

    if (rgba_mat.rows!=z_mat.rows || rgba_mat.cols!=z_mat.cols)
    {
        std::string message = "Error, resolution missmatch found! Aborting merge process...";
        throw std::runtime_error(message);
    }

    data = cv::Mat_<rgbazm_>(rgba_mat.rows, rgba_mat.cols);

    if (z_mat.channels()!=1)
        throw std::runtime_error("Channels error! Z-Pass should be non-transparent grayscale image.");

    if (rgba_mat.channels()==3)
        fill_data<cv::Vec3f>(rgba_mat, z_mat, data, mode);
    else if (rgba_mat.channels()==4)
        fill_data<cv::Vec4f>(rgba_mat, z_mat, data, mode);
    else
        throw std::runtime_error("Channels error! RGB-Pass should be in RGB or RGBA mode.");
}

void
ZMergerImage::save_as_file(std::string file_name)
{
    cv::Mat_<cv::Vec<uint16_t, 4>> output(data.rows, data.cols);

    for (size_t i = 0; i<data.rows; ++i)
    {
        for (size_t j = 0; j<data.cols; ++j)
        {
            output(i, j)[0] = static_cast<uint16_t>(data(i, j)[0]*MAX_16_BIT_VALUE);
            output(i, j)[1] = static_cast<uint16_t>(data(i, j)[1]*MAX_16_BIT_VALUE);
            output(i, j)[2] = static_cast<uint16_t>(data(i, j)[2]*MAX_16_BIT_VALUE);
            output(i, j)[3] = static_cast<uint16_t>(data(i, j)[3]*MAX_16_BIT_VALUE);
        }
    }

    cv::imwrite(file_name, output);
}

void
ZMergerImage::save_as_binary_file(std::string)
{
}

