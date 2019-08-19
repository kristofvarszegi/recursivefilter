#pragma once

#ifdef _WIN32
#include <device_launch_parameters.h>
#endif

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include "CpuTable.hpp"

namespace gpuacademy {
extern const int kTableBlockDimX;
extern const int kTableBlockDimY;
extern const dim3 kThreadBlockDim;
void apply_right_down_recursive_filter_gpu(const CpuTable& input_table,
	const float* filter_coeffs, int num_filter_coeffs, CpuTable& output_table);
}