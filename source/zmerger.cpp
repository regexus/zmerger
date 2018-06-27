// // Name   :: Z-based pixel blender
// // Date   :: Februar 2018
// // Author :: Alexander Kasperovich

#include "utilities.hpp"
#include "zimage.hpp"
#include "json11.hpp"

// #define DBL_EPSILON 2.2204460492503131e-16
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <math.h>
#include <numeric>
#include <omp.h>
#include <vector>

using rgbazm_ = cv::Vec<float, 6>;

struct rgbazm
{
    uint16_t r, g, b, a;
    uint16_t z;
    unsigned char m;
};

int main(int argc, char** argv)
{

    if (argc < 5)
    {
        std::cout << "Input parameters error! Use json name path, png output file path, zpass inversion mode and zpass extension flag as parameters.";
        return 1;
    }

    auto json_file_path = std::string(argv[1]);
    auto output_png_path = std::string(argv[2]);
    bool invert_z = std::stoi(std::string(argv[3]));
    bool expand_z = std::stoi(std::string(argv[4]));

    // Get the output resolution (optional)
    int out_res_x = 0;
    int out_res_y = 0;
    if (argc == 7)
    {
        out_res_x = std::stoi(std::string(argv[5]));
        out_res_y = std::stoi(std::string(argv[6]));
    }

    // Get the data from json file
    std::string json_string;
    std::string tmp_str;
    std::ifstream json_file(json_file_path);
    while (std::getline(json_file, tmp_str))
        if (lstrip(tmp_str).substr(0, 2) != "//")
        {
            json_string += tmp_str;
        }

    std::string error_message;
    json11::Json IMAGES_DATA_INFO = json11::Json::parse(json_string, error_message);
    size_t images_count = IMAGES_DATA_INFO.array_items().size();
    if (images_count == 0)
    {
        std::cout << json_string << std::endl;
        std::cout << "Warning! No input images found, aborting..." << std::endl;
        return 1;
    }

    // Starting global time tracking
    auto start_time = get_time();

    // Starting time tracking for images reading process
    auto t1 = get_time();

    ZImageSet zimage_set;
    zimage_set.z_images.reserve(images_count);

    #pragma omp parallel for
    for (int k=0; k<images_count; ++k)
    {
        zimage_set.z_images.emplace_back(
            ZImage(
                IMAGES_DATA_INFO[k]["I"].string_value(),
                IMAGES_DATA_INFO[k]["Z"].string_value(),
                static_cast<BlendMode>(std::stoi(IMAGES_DATA_INFO[k]["M"].string_value())))
            );
    }

    // Print timing
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(get_time() - t1).count() / 1000.0;
    std::cout << "Images are loaded! Elapsed time: " << duration << std::endl;
    t1 = get_time();

    auto result = zimage_set.merge_images(invert_z, {0, 0, 0, 0});

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(get_time() - t1).count() / 1000.0;
    std::cout << "Pixel blending done! Elapsed time: " << duration << std::endl;
    t1 = get_time();

    // Rescale output image if neccessary
    if (bool(out_res_x * out_res_y))
    {
        cv::Size size(out_res_x, out_res_y);
        cv::resize(result, result, size, 0, 0, cv::INTER_CUBIC);
    }

    cv::imwrite(output_png_path, result);

    // Print timing
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(get_time() - t1).count() / 1000.0;
    std::cout << "Image saved! Elapsed time: " << duration << std::endl;

    // Print global timing
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(get_time() - start_time).count() / 1000.0;
    std::cout << "Processing done! Cumulative elapsed time: " << duration << std::endl;

}

int main__(int argc, char **argv)
{

    if (argc < 5)
    {
        std::cout << "Input parameters error! Use json name path, png output file path, zpass inversion mode and zpass extension flag as parameters.";
        return 1;
    }

    auto json_file_path = std::string(argv[1]);
    auto output_png_path = std::string(argv[2]);
    bool invert_z = std::stoi(std::string(argv[3]));
    bool expand_z = std::stoi(std::string(argv[4]));

    // Get the output resolution (optional)
    int out_res_x = 0;
    int out_res_y = 0;
    if (argc == 7)
    {
        out_res_x = std::stoi(std::string(argv[5]));
        out_res_y = std::stoi(std::string(argv[6]));
    }

    std::string json_string;
    std::string error_message;

    // Get the data from json file
    std::ifstream json_file(json_file_path);

    std::string tmp_str;
    while (std::getline(json_file, tmp_str))
        if (lstrip(tmp_str).substr(0, 2) != "//")
        {
            json_string += tmp_str;
        }

    auto IMAGES_DATA_INFO = json11::Json::parse(json_string, error_message);
    auto images_count = IMAGES_DATA_INFO.array_items().size();
    if (images_count == 0)
    {
        std::cout << json_string << std::endl;
        std::cout << "Warning! No input images found, aborting..." << std::endl;
        return 1;
    }

    // Starting global time tracking
    auto start_time = get_time();

    // Starting time tracking for images reading process
    auto t1 = get_time();

    // Reading the first image to get the resolution
    cv::Mat rgba_image = cv::imread(IMAGES_DATA_INFO[0]["I"].string_value(), CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat zpass_image = cv::imread(IMAGES_DATA_INFO[0]["Z"].string_value(), CV_LOAD_IMAGE_UNCHANGED);

    // Collect dimensions info
    auto image_height = rgba_image.size().height;
    auto zpass_height = zpass_image.size().height;
    auto image_width = rgba_image.size().width;
    auto zpass_width = zpass_image.size().width;
    auto pixels_count = image_height*image_width;

    // Checking the resolution
    if (!(image_height == zpass_height && image_width == zpass_width))
    {
        std::cout << "Resolution error! RGBA and Z-DEPTH images have different resolutions." << std::endl;
        return 1;
    }

    // Create the main buffer
    const unsigned short channels_count = rgba_image.channels() + 2; // Adding two for z-depth and mode
    if (channels_count != 6)
    {
        std::cout << "Channels error! Wrong number of channels, RGBA image should have 6 channels!" << std::endl;
        return 1;
    }
    size_t buffer_size = rgba_image.cols * rgba_image.rows * images_count;
    std::vector<rgbazm> main_buffer;
    main_buffer.reserve(buffer_size);

    // Add the data from images to the main buffer
    bool error_found = false;
    auto ellipse_kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(2, 2));
    #pragma omp parallel for
    for (int k = 0; k < images_count; ++k)
    {

        cv::Mat rgba_image = cv::imread(IMAGES_DATA_INFO[k]["I"].string_value(), CV_LOAD_IMAGE_UNCHANGED);
        if (rgba_image.depth() != CV_16U) rgba_image.convertTo(rgba_image, CV_16U, float(UINT16_MAX) / UINT8_MAX);
        auto rgb_data_as_array = reinterpret_cast<uint16_t*>(rgba_image.data);

        cv::Mat zpass_image = cv::imread(IMAGES_DATA_INFO[k]["Z"].string_value(), CV_LOAD_IMAGE_UNCHANGED);
        if (zpass_image.channels()>1) cv::cvtColor(zpass_image, zpass_image, CV_BGR2GRAY);
        if (zpass_image.depth() != CV_16U) zpass_image.convertTo(zpass_image, CV_16U, float(UINT16_MAX) / UINT8_MAX);
        if (expand_z)
        {
            if (invert_z)
                cv::erode(zpass_image, zpass_image, ellipse_kernel);
            else
                cv::dilate(zpass_image, zpass_image, ellipse_kernel);
        }
        auto z_data_as_array = reinterpret_cast<uint16_t*>(zpass_image.data);

        // Checking the resolution
        if (!
            ((image_height == rgba_image.size().height) &&
            (image_height == zpass_image.size().height) &&
            (image_width == rgba_image.size().width) &&
            (image_width == zpass_image.size().width))
            )
        {
            error_found = true;
            // break;
        }

        unsigned char mode = stoi(IMAGES_DATA_INFO[k]["M"].string_value());

        unsigned short i, j;
        size_t rgba_index, zpass_index, block_index;
        auto channels_count = rgba_image.channels();

        for (i = 0; i < image_height; ++i)
            for (j = 0; j < image_width; ++j)
            {
                // Compute indexies
                zpass_index = i*image_width + j;
                rgba_index = zpass_index*channels_count;
                block_index = pixels_count*k + zpass_index;
                //block_index = zpass_index*images_count + k;
                // RGBA Values
                main_buffer[block_index].b = rgb_data_as_array[rgba_index + 0];
                main_buffer[block_index].g = rgb_data_as_array[rgba_index + 1];
                main_buffer[block_index].r = rgb_data_as_array[rgba_index + 2];
                main_buffer[block_index].a = rgb_data_as_array[rgba_index + 3];
                // Z-Pass value
                main_buffer[block_index].z = z_data_as_array[zpass_index];
                // Mode value
                main_buffer[block_index].m = mode;
            }
    }

    if (error_found)
    {
        std::cout << "Resolution error! All images should have the same resolution. Aborting..." << std::endl;
        return 1;
    }

    // Print timing
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(get_time() - t1).count() / 1000.0;
    t1 = get_time();
    std::cout << "Images are loaded! Elapsed time: " << duration << std::endl;

    // Generate sorting map
    std::vector<std::vector<std::vector<unsigned char>>> sorting_map;
    sorting_map.resize(image_height);
    #pragma omp parallel for
    for (short i = 0; i < image_height; ++i)
    {
        size_t block_index;
        sorting_map[i].resize(image_width);

        for (unsigned short j = 0; j < image_width; ++j)
        {
            sorting_map[i][j].resize(images_count);

            // Get z-values of the stack
            std::vector<uint16_t> zvalues;
            zvalues.reserve(images_count);

            for (unsigned char k = 0; k < images_count; ++k)
            {
                block_index = pixels_count*k + (i*image_width + j);
                // Fill map with initial indexes
                sorting_map[i][j][k] = k;
                // Collect the z-values
                zvalues[k] = main_buffer[block_index].z;
            }

            if (invert_z)
                stable_sort(sorting_map[i][j].begin(), sorting_map[i][j].end(), [&zvalues](const unsigned char &i1, const unsigned char &i2) { return zvalues[i1] > zvalues[i2]; });
            else
                stable_sort(sorting_map[i][j].begin(), sorting_map[i][j].end(), [&zvalues](const unsigned char &i1, const unsigned char &i2) { return zvalues[i1] < zvalues[i2]; });
        }
    }

    // Print timing
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(get_time() - t1).count() / 1000.0;
    t1 = get_time();
    std::cout << "Sorting map generated! Elapsed time: " << duration << std::endl;

    std::vector<uint16_t> output_buffer(image_height*image_width * 4, 0);
    // Apply pixel blending
    #pragma omp parallel for
    for (short i = 0; i < image_height; ++i)
    {
        size_t pixel_index, block_index, output_index;
        float src_alpha, dst_alpha, out_alpha;
        float src_rgb, dst_rgb, out_rgb;
        float blending_function;
        unsigned char mode;
        unsigned short j;
        unsigned char k;

        for (j = 0; j < image_width; ++j)
        {
            for (k = 0; k < images_count; ++k)
            {
                pixel_index = i*image_width + j;
                block_index = pixels_count*sorting_map[i][j][k] + pixel_index;

                output_index = pixel_index * 4;

                src_alpha = output_buffer[output_index + 3] / (UINT16_MAX + 0.0f); // alpha value
                dst_alpha = main_buffer[block_index].a / (UINT16_MAX + 0.0f); // alpha value

                if (dst_alpha == 0) continue; // Early termination in case of black alpha

                out_alpha = dst_alpha + src_alpha*(1 - dst_alpha);

                mode = main_buffer[block_index].m; // mode value
                if (k == 0) mode = 0; // Reset to normal for the first image

                for (unsigned char channel = 0; channel < 3; ++channel)
                {
                    src_rgb = output_buffer[output_index + channel] / (UINT16_MAX + 0.0f); // rgb value
                    if (channel == 0)
                        dst_rgb = main_buffer[block_index].b / (UINT16_MAX + 0.0f); // rgb value

                    else if (channel == 1)
                        dst_rgb = main_buffer[block_index].g / (UINT16_MAX + 0.0f); // rgb value

                    else if (channel == 2)
                        dst_rgb = main_buffer[block_index].r / (UINT16_MAX + 0.0f); // rgb value

                    // Normal mode
                    if (mode == 0)
                        blending_function = dst_rgb;

                    // Multiply mode
                    if (mode == 1)
                        blending_function = src_rgb * dst_rgb;

                    // Screen mode
                    if (mode == 2)
                        blending_function = src_rgb + dst_rgb - src_rgb * dst_rgb;

                    out_rgb = (1 - dst_alpha / out_alpha)*src_rgb + (dst_alpha / out_alpha)*((1 - src_alpha)*dst_rgb + src_alpha*blending_function);

                    output_buffer[output_index + channel] = (uint16_t)(out_rgb*UINT16_MAX);
                }

                output_buffer[output_index + 3] = (uint16_t)(out_alpha*UINT16_MAX);
            }
        }

    }

    // Print timing
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(get_time() - t1).count() / 1000.0;
    t1 = get_time();
    std::cout << "Pixel blending done! Elapsed time: " << duration << std::endl;

    // Create output image buffer
    auto output_image = cv::Mat(output_buffer, false);
    output_image = output_image.reshape(4, image_height);

    // Rescale output image if neccessary
    if (bool(out_res_x * out_res_y))
    {
        cv::Size size(out_res_x, out_res_y);
        cv::resize(output_image, output_image, size, 0, 0, cv::INTER_CUBIC);
    }

    // Save the output image
    cv::imwrite(output_png_path, output_image);

    // Print timing
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(get_time() - t1).count() / 1000.0;
    std::cout << "Image saved! Elapsed time: " << duration << std::endl;

    // Print global timing
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(get_time() - start_time).count() / 1000.0;
    std::cout << "Processing done! Cumulative elapsed time: " << duration << std::endl;

    return 0;
}