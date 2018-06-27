#pragma once

#include <cstdint>
#include <limits>

const uint8_t MAX_8_BIT_VALUE = std::numeric_limits<uint8_t>::max();
const uint16_t MAX_16_BIT_VALUE = std::numeric_limits<uint16_t>::max();
enum class BlendMode {NORMAL, MULTIPLY, SCREEN};
