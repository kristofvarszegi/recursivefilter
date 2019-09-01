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

enum OUTPUT_STEP { STEP_1, STEP_2, STEP_3, STEP_4, STEP_5};

namespace config {
const int kBlockDim2dGridSmall = 4;
const int kBlockDim1dGridSmall = 7;
const int kBlockDim2dGridLarge = 28;
const int kBlockDim1dGridLarge = 64;
const int kNumKernelRunsFew = 1;
const int kNumKernelRunsMany = 1000;
} // namespace config

template <int BLOCKDIM_2DGRID, int BLOCKDIM_1DGRID, int NUM_KERNEL_RUNS>
float recursivefilter_downright_gpu(const CpuTable &input, float feedfwd_coeff,
                                    float feedback_coeff,
                                    OUTPUT_STEP output_step,
                                    std::vector<CpuTable> &outputs);
} // namespace gpuacademy
#endif
