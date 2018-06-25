#pragma once

#define DBL_EPSILON 2.2204460492503131e-16
#include <opencv2/core.hpp>
#include <cstdint>
#include <limits>

const uint8_t MAX_8_BIT_VALUE = std::numeric_limits<uint8_t>::max();
const uint16_t MAX_16_BIT_VALUE = std::numeric_limits<uint16_t>::max();
enum class MapType{NORMAL, DIFFUSE, REFLECTION, HEIGHT};
enum class HM_SOLVER_TYPE{MULTIRES, MULTIRES_SPARSE, SPARSE};

// Reads the image file and converts
// it's values to double in [0, 1] range.
cv::Mat
read_and_normalize(std::string image_path);

// Prepares the matrix for saving as an image, 
// does necessary values conversion
// depending on the map type.
cv::Mat
remap_to_uint16(cv::Mat source_mat, MapType map_type);

struct DirectionalLightSourceData
{
    cv::Vec3d lightDirection;
    cv::Mat_<cv::Vec3d> imageMatrix;

    DirectionalLightSourceData(cv::Vec3d lightDirection, std::string image_path);
};

enum class MODE {NORMAL, MULTIPLY, SCREEN};

using rgbazm_ = cv::Vec<float, 6>;

class ZMergerImage : public cv::Mat
{
    public:
    
    cv::Mat_<rgbazm_> data;

    ZMergerImage(size_t rows, size_t cols);
    ZMergerImage(std::string rgba_file_path, std::string z_file_path, MODE mode);
    ZMergerImage(std::string zmerger_binary_file_path);

    void
    save_as_file(std::string);

    void
    save_as_binary_file(std::string);
};