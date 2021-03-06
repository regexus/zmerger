#pragma once

#include <cstdint>
#include <limits>

const uint8_t MAX_8_BIT_VALUE = std::numeric_limits<uint8_t>::max();
const float MAX_8_BIT_VALUE_F = static_cast<float>(MAX_8_BIT_VALUE);
const uint16_t MAX_16_BIT_VALUE = std::numeric_limits<uint16_t>::max();
const float MAX_16_BIT_VALUE_F = static_cast<float>(MAX_16_BIT_VALUE);

// GCC related
#if !defined DBL_EPSILON
    const double DBL_EPSILON = std::numeric_limits<double>::epsilon();
#endif
