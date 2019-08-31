#ifndef GPUFUNCS_HPP
#define GPUFUNCS_HPP

#ifdef _WIN32
#include <device_launch_parameters.h>
#endif

#ifdef __INTELLISENSE__
void __syncthreads();
const float __ldg(const float *);
#endif

#include "CpuTable.hpp"

#include <vector>

namespace gpuacademy {

enum OUTPUT_STEP { STEP_1, STEP_2, STEP_3, STEP_4, STEP_5, STEP_6 };

namespace config {
enum BLOCK_SIZE_CLASS { SMALL, LARGE };
size_t get_blockdim_2dgrid_x(config::BLOCK_SIZE_CLASS bsc);
size_t get_blockdim_2dgrid_y(config::BLOCK_SIZE_CLASS bsc);
size_t get_blockdim_1dgrid_x(config::BLOCK_SIZE_CLASS bsc);
} // namespace config

float recursivefilter_downright_gpu(const CpuTable &input, float filter_coeff_0,
                                    float filter_coeff_1,
                                    config::BLOCK_SIZE_CLASS block_size_class,
                                    size_t num_kernel_runs,
                                    OUTPUT_STEP output_step,
                                    std::vector<CpuTable> &outputs);
} // namespace gpuacademy
#endif
