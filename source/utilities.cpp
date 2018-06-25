#include "utilities.h"

#include <chrono>
// #include <filesystem>
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
