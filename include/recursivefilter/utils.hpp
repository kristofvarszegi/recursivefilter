#ifndef UTILS_HPP
#define UTILS_HPP

#include <time.h>
#include <string>

namespace gpuacademy {
std::string to_string(const float* data, int num_rows, int num_cols);
float to_ms(const clock_t& t);
std::string to_ms_str(const clock_t& t);
}
#endif
