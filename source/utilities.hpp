#pragma once

#include <opencv2/core.hpp>

#include <chrono>
#include <iostream>
#include <string>

template <typename T>
void print_(T input)
{
    std::cout << input;
}

template <typename T, typename... Args>
void print_(T input, Args... args)
{
    print_(input);
    std::cout << " ";
    print_(args...);
}

template <typename... Args>
void print(Args... args)
{
    std::cout << std::fixed;
    print_(args...);
    std::cout << std::endl;
}

std::chrono::time_point<std::chrono::steady_clock>
get_time();

std::chrono::milliseconds
time_from(std::chrono::time_point<std::chrono::steady_clock> time_point);

void print_mat(cv::Mat, std::string);

std::string lstrip(std::string s);