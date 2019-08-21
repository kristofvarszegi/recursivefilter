#pragma once

#ifdef _WIN32
#include <device_launch_parameters.h>
#endif

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include "CpuTable.hpp"

namespace gpuacademy {
void apply_right_down_recursive_filter_gpu(const CpuTable& input_table,
	float filter_coeff_0, float filter_coeff_1, CpuTable& output_table);
}