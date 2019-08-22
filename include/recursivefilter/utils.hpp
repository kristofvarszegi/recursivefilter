#pragma once

#include <time.h>
#include <string>

namespace gpuacademy {
float to_ms(const clock_t& t);
std::string to_ms_str(const clock_t& t);
}