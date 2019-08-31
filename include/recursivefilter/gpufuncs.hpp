#ifndef GPUFUNCS_HPP
#define GPUFUNCS_HPP

#ifdef _WIN32
#include <device_launch_parameters.h>
#endif

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include "CpuTable.hpp"

#include <vector>

namespace gpuacademy {

enum OUTPUT_STEP { STEP_1, STEP_2, STEP_3, STEP_4, STEP_5_FINAL };

namespace config {
	const int kBlockSizeSmallX = 4;
	const int kBlockSizeSmallY = kBlockSizeSmallX;
	const int kBlockSizeLargeX = 32;
	const int kBlockSizeLargeY = kBlockSizeLargeX;
}

float recursivefilter_downright_gpu(const CpuTable &input, float filter_coeff_0,
                                    float filter_coeff_1, size_t tableblockdim_x,
	size_t tableblockdim_y, size_t num_kernel_runs,
                                    OUTPUT_STEP output_step,
                                    std::vector<CpuTable> &outputs);
} // namespace gpuacademy
#endif
