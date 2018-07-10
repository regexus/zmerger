// Name   :: Z-based pixel merger
// Date   :: Juli 2018
// Author :: Alexander Kasperovich

#include "json11.hpp"
#include "utilities.hpp"
#include "zimage.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
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

int main(int argc, char** argv)
{

    if (argc < 5)
    {
        std::cout << "Input parameters error! Use json name path, png output file path, zpass inversion mode and zpass extension flag as parameters.";
        return 1;
    }

    auto json_file_path = std::string(argv[1]);
    auto output_image_path = std::string(argv[2]);
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

    auto json_string = read_json_string(json_file_path);
    std::string error_message;
    json11::Json IMAGES_DATA_INFO = json11::Json::parse(json_string, error_message);
    unsigned short images_count = IMAGES_DATA_INFO.array_items().size();
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

    auto zimage_set = ZImageSet(images_count);
    
    // Reading the source images
    #pragma omp parallel for
    for (int k=0; k<images_count; ++k)
    {   
        zimage_set.z_images[k] = ZImage(
                IMAGES_DATA_INFO[k]["I"].string_value(),
                IMAGES_DATA_INFO[k]["Z"].string_value(),
                static_cast<BlendMode>(std::stoi(IMAGES_DATA_INFO[k]["M"].string_value()))
        );
    }

    if (!zimage_set.resolution_check())
    {
        std::cout << "Resolution error! Input images have different resolutions." << std::endl;
        return 1;
    }

    // Expand the z-pass if needed.
    if (expand_z)
        zimage_set.expand_z(invert_z);

    // Print timing
    auto duration = (get_time() - t1).count() / 1000.0;
    std::cout << "Images are loaded! Elapsed time: " << duration << std::endl;
    t1 = get_time();

    auto result = zimage_set.merge_images(invert_z, {0, 0, 0, 0});

    duration = (get_time() - t1).count() / 1000.0;
    std::cout << "Pixel blending done! Elapsed time: " << duration << std::endl;
    t1 = get_time();

    // Rescale output image if neccessary
    if (bool(out_res_x * out_res_y))
    {
        cv::Size size(out_res_x, out_res_y);
        cv::resize(result, result, size, 0, 0, cv::INTER_CUBIC);
    }

    // Save the result
    cv::imwrite(output_image_path, result);

    // Print timing
    duration = (get_time() - t1).count() / 1000.0;
    std::cout << "Image saved! Elapsed time: " << duration << std::endl;

    // Print global timing
    duration = (get_time() - start_time).count() / 1000.0;
    std::cout << "Processing done! Cumulative elapsed time: " << duration << std::endl;
}
