#pragma once

#ifdef _WIN32
#include <device_launch_parameters.h>
#endif

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include "CpuTable.hpp"

namespace gpuacademy {

template <int tableblockdim_x, int tableblockdim_y>
float recursivefilter_downright_gpu(const CpuTable& input_table,
	float filter_coeff_0, float filter_coeff_1, int num_kernel_runs,
	CpuTable& output_table);

template float recursivefilter_downright_gpu<2, 2>(const CpuTable& input_table,
	float filter_coeff_0, float filter_coeff_1, int num_kernel_runs,
	CpuTable& output_table);

template float recursivefilter_downright_gpu<32, 32>(const CpuTable& input_table,
	float filter_coeff_0, float filter_coeff_1, int num_kernel_runs,
	CpuTable& output_table);

template float recursivefilter_downright_gpu<64, 64>(const CpuTable& input_table,
	float filter_coeff_0, float filter_coeff_1, int num_kernel_runs,
	CpuTable& output_table);
}