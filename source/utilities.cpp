#include "utilities.hpp"

#include <opencv2/core.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

// Timing

std::chrono::time_point<std::chrono::steady_clock, std::chrono::milliseconds>
get_time()
{
    return std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now());
}

std::chrono::milliseconds
time_from(std::chrono::time_point<std::chrono::steady_clock, std::chrono::milliseconds> time_point)
{
    return get_time()-time_point;
}

// Strings

std::string
lstrip(std::string s)
{
    auto first_non_space = std::find_if(
        s.begin(), s.end(), [](int ch) {return !std::isspace(ch);}
    );
    s.erase(s.begin(), first_non_space);

    return s;
}

std::string
read_json_string(std::string json_file_path)
{
    // The function reads json with comments 
    // in 'json_file_path' file as a string.

    std::string json_string;
    std::string tmp_str;
    std::ifstream json_file(json_file_path);
    while (std::getline(json_file, tmp_str))
        if (lstrip(tmp_str).substr(0, 2) != "//")
        {
            json_string += tmp_str;
        }
    return json_string;
}

// Debugging

void
print_mat(cv::Mat mat, std::string prefix="matrix:")
{
    auto formatMat=cv::Formatter::get(cv::Formatter::FMT_PYTHON);
    formatMat->set32fPrecision(3);
    formatMat->set64fPrecision(3);
    std::cout << std::fixed << prefix << std::endl << formatMat->format(mat) << std::endl;
}