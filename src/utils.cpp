#include "utils.hpp"

#include <sstream>

namespace gpuacademy {

size_t align_size_logic2glmem(size_t size, size_t aligment_floats, size_t blockdim) {
  if (blockdim < aligment_floats) {
    return (size % blockdim == 0) ? ((size / blockdim) * aligment_floats) : ((size / blockdim + 1) * aligment_floats);
  } else if (blockdim == aligment_floats) {
    return size;
  } else {
    throw std::runtime_error("Unsupported case \"blockdim > aligment_floats\"");
  }
  return 0;
}

size_t align_index_logic2glmem(size_t index, size_t aligment_floats, size_t blockdim) {
  return index % blockdim + (index / blockdim) * aligment_floats;
}

size_t align_index_glmem2logic(size_t index, size_t aligment_floats, size_t blockdim) {
  return index % aligment_floats + (index / aligment_floats) * blockdim;
}

std::string to_string(const float *data, size_t num_rows, size_t num_cols) {
  std::stringstream ss;
  ss << std::endl;
  for (size_t i_row = 0; i_row < num_rows; ++i_row) {
    for (size_t i_col = 0; i_col < num_cols; ++i_col) {
      ss << std::to_string(data[i_col + i_row * num_cols]) << ", ";
    }
    ss << std::endl;
  }
  return ss.str();
}

float to_ms(const clock_t &t) {
  return (float)t / (float)CLOCKS_PER_SEC * 1000.0f;
}

std::string to_ms_str(const clock_t &t) { return std::to_string(to_ms(t)); }

} // namespace gpuacademy
