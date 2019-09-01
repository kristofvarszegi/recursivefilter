#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <time.h>

namespace gpuacademy {
size_t align_size_logic2glmem(size_t size, size_t aligment_floats, size_t blockdim);
size_t align_index_logic2glmem(size_t index, size_t aligment_floats, size_t blockdim);
size_t align_index_glmem2logic(size_t index, size_t aligment_floats, size_t blockdim);
std::string to_string(const float *data, int num_rows, int num_cols);
float to_ms(const clock_t &t);
std::string to_ms_str(const clock_t &t);
} // namespace gpuacademy
#endif
