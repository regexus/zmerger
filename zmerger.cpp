// Name :: Z-based pixel blender 
// Author :: Alexander Kasperovich
// Date :: September 2017

#include <map>
#include <omp.h>
#include <vector>
#include <chrono>
#include <fstream>
#include <numeric>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "json11.hpp"

using namespace std;
using namespace std::chrono;

struct rgbazm
{
    uint16_t r, g, b, a;
    uint16_t z;
    unsigned char m;
};

int main(int argc, char **argv)
{

    if (argc < 5)
    {
        std::cout << "Input parameters error! Use json name path, png output file path, zpass inversion mode and zpass extension flag as parameters.";
        return 1;
    }

    auto json_file_path = string(argv[1]);
    auto output_png_path = string(argv[2]);
    bool invert_z = stoi(string(argv[3]));
    bool expand_z = stoi(string(argv[4]));
    
    // Get the output resolution (optional)
    int out_res_x = 0;
    int out_res_y = 0;
    if (argc == 7) 
    {
        out_res_x = stoi(string(argv[5]));
        out_res_y = stoi(string(argv[6]));
    }

    string json_string;
    string error_message;

    // Get the data from json file
    ifstream json_file(json_file_path);

    string tmp_str;
    while (std::getline(json_file, tmp_str))
        json_string += tmp_str;

    auto IMAGES_DATA_INFO = json11::Json::parse(json_string, error_message);
    unsigned char images_count = IMAGES_DATA_INFO.array_items().size();
    if (images_count == 0)
    {
        std::cout << json_string << endl;
        std::cout << "Warning! No input images found, aborting..." << endl;
        return 1;
    }

    // Starting time tracking
    auto t1 = high_resolution_clock::now();
    auto t2 = high_resolution_clock::now();

    // Reading the first image to get the resolution
    cv::Mat rgba_image = cv::imread(IMAGES_DATA_INFO[0]["I"].string_value(), CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat zpass_image = cv::imread(IMAGES_DATA_INFO[0]["Z"].string_value(), CV_LOAD_IMAGE_UNCHANGED);

    // Checking the resolution
    const unsigned short image_height = rgba_image.size().height;
    const unsigned short zpass_height = zpass_image.size().height;
    const unsigned short image_width = rgba_image.size().width;
    const unsigned short zpass_width = zpass_image.size().width;

    if (!(image_height == zpass_height && image_width == zpass_width))
    {
        cout << "Resolution error! RGBA and Z-DEPTH images have different resolutions." << endl;
        return 1;
    }

    // Create the main buffer
    const unsigned short channels_count = rgba_image.channels() + 2; // Adding two for z-depth and mode
    if (channels_count != 6)
    {
        cout << "Channels error! Wrong number of channels, RGBA image should have 6 channels!" << endl;
        return 1;
    }
    size_t buffer_size = rgba_image.cols * rgba_image.rows * images_count;
    vector<rgbazm> main_buffer(buffer_size);

    // Add the data from images to the buffer
    bool error_found = false;
    #pragma omp parallel for
    for (short k = 0; k < images_count; ++k)
    {
        if (error_found) continue;

        cv::Mat rgba_image = cv::imread(IMAGES_DATA_INFO[k]["I"].string_value(), CV_LOAD_IMAGE_UNCHANGED);
        if (rgba_image.depth() != CV_16U) rgba_image.convertTo(rgba_image, CV_16U, float(UINT16_MAX) / UINT8_MAX);
        auto rgb_data_as_array = reinterpret_cast<uint16_t*>(rgba_image.data);

        cv::Mat zpass_image = cv::imread(IMAGES_DATA_INFO[k]["Z"].string_value(), CV_LOAD_IMAGE_UNCHANGED);
        if (zpass_image.channels()>1) cv::cvtColor(zpass_image, zpass_image, CV_BGR2GRAY);
        if (zpass_image.depth() != CV_16U) zpass_image.convertTo(zpass_image, CV_16U, float(UINT16_MAX) / UINT8_MAX);
        if (expand_z)
        {
            auto el = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(2, 2));
            if (invert_z)
                cv::erode(zpass_image, zpass_image, el);
            else
                cv::dilate(zpass_image, zpass_image, el);
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
            continue;
        }

        unsigned char mode = stoi(IMAGES_DATA_INFO[k]["M"].string_value());

        unsigned short i, j;
        size_t rgba_index, zpass_index, block_index;

        for (i = 0; i < image_height; ++i)
            for (j = 0; j < image_width; ++j)
            {
                // Compute indexies
                rgba_index = (i*image_width + j)*rgba_image.channels();
                zpass_index = (i*image_width + j);
                block_index = (i*image_width + j)*images_count + k;
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
        cout << "Resolution error! All images should have the same resolution. Aborting..." << endl;
        return 1;
    }

    // Print timing
    auto duration = duration_cast<milliseconds>(high_resolution_clock::now() - t1).count() / 1000.0;
    t1 = high_resolution_clock::now();
    std::cout << "Images are loaded! Elapsed time: " << duration << endl;

    // Generate sorting map
    vector<vector<vector<unsigned char>>> sorting_map;
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
            vector<uint16_t> zvalues(images_count);
            for (unsigned char k = 0; k < images_count; ++k)
            {
                block_index = (i*image_width + j)*images_count + k;
                // Fill map with initial indexes
                sorting_map[i][j][k] = k;
                // Collect the z-values
                zvalues[k] = main_buffer[block_index].z;
            }

            auto& sorted_indexies = sorting_map[i][j];
            if (invert_z)
                stable_sort(sorted_indexies.begin(), sorted_indexies.end(), [&zvalues] (unsigned char i1, unsigned char i2) { return zvalues[i1] > zvalues[i2]; });
            else
                stable_sort(sorted_indexies.begin(), sorted_indexies.end(), [&zvalues] (unsigned char i1, unsigned char i2) { return zvalues[i1] < zvalues[i2]; });
        }
    }

    // Print timing
    duration = duration_cast<milliseconds>(high_resolution_clock::now() - t1).count() / 1000.0;
    t1 = high_resolution_clock::now();
    std::cout << "Sorting map generated! Elapsed time: " << duration << endl;

    vector<uint16_t> output_buffer(image_height*image_width * 4, 0);
    // Apply pixel blending
    #pragma omp parallel for
    for (short i = 0; i < image_height; ++i)
    {
        size_t sorted_image_index, block_index, output_index;
        float src_alpha, dst_alpha, out_alpha;
        float src_rgb, dst_rgb, out_rgb;
        float blending_function;
        unsigned char mode;
        unsigned short j;
        unsigned char k;

        for (j = 0; j < image_width; ++j)
            for (k = 0; k < images_count; ++k)
            {
                sorted_image_index = sorting_map[i][j][k];
                block_index = (i*image_width + j)*images_count + sorted_image_index;
                output_index = (i*image_width + j) * 4;

                src_alpha = output_buffer[output_index + 3] / (UINT16_MAX + 0.0f); // alpha value
                dst_alpha = main_buffer[block_index].a / (UINT16_MAX + 0.0f); // alpha value
                                                                              // Early termination in case of black alpha
                if (dst_alpha == 0) continue;
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

    // Print timing
    duration = duration_cast<milliseconds>(high_resolution_clock::now() - t1).count() / 1000.0;
    t1 = high_resolution_clock::now();
    std::cout << "Pixel blending done! Elapsed time: " << duration << endl;

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
    duration = duration_cast<milliseconds>(high_resolution_clock::now() - t1).count() / 1000.0;
    std::cout << "Image saved! Elapsed time: " << duration << endl;

    // Print timing
    duration = duration_cast<milliseconds>(high_resolution_clock::now() - t2).count() / 1000.0;
    std::cout << "Processing done! Commulative elapsed time: " << duration << endl;

    //cv::namedWindow("Display window", cv::WINDOW_NORMAL);// Create a window for display.
    //cv::imshow("Display window", output_image);                   // Show our image inside it.
    //cv::waitKey(0);                                          // Wait for a keystroke in the window

    return 0;
}