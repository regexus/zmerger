#include "utilities.hpp"

#include <opencv2/core.hpp>

#include <chrono>
#include <iostream>
#include <string>

std::chrono::time_point<std::chrono::steady_clock>
get_time()
{
    return std::chrono::steady_clock::now();
}

std::chrono::milliseconds
time_from(std::chrono::time_point<std::chrono::steady_clock> time_point)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-time_point);
}

void
print_mat(cv::Mat mat, std::string prefix="matrix:")
{
    auto formatMat=cv::Formatter::get(cv::Formatter::FMT_PYTHON);
    formatMat->set32fPrecision(3);
    formatMat->set64fPrecision(3);
    std::cout << std::fixed << prefix << std::endl << formatMat->format(mat) << std::endl;
}

std::string
lstrip(std::string s)
{
    auto first_non_space = std::find_if(
        s.begin(), s.end(), [](int ch) {return !std::isspace(ch);}
    );
    s.erase(s.begin(), first_non_space);

    return s;
}