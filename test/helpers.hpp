#pragma once

#include "CpuTable.hpp"

#include <ctime>

namespace gpuacademy {

extern const float kEpsilon;
extern const float kSatFilterCoeffs[];

struct comparison_result_t {
	int equals;
	float runtime_1kernel_ms;
};

template <int tableblockdim_x, int tableblockdim_y>
comparison_result_t apply_recursivefilter_gpu_and_compare_with_cpu(const CpuTable& input,
	float filter_coeff_0, float filter_coeff_1, int num_kernel_runs,
	bool print_tables);

template comparison_result_t apply_recursivefilter_gpu_and_compare_with_cpu<2, 2>(const CpuTable& input,
	float filter_coeff_0, float filter_coeff_1, int num_kernel_runs,
	bool print_tables);
template comparison_result_t apply_recursivefilter_gpu_and_compare_with_cpu<32, 32>(const CpuTable& input,
	float filter_coeff_0, float filter_coeff_1, int num_kernel_runs,
	bool print_tables);
template comparison_result_t apply_recursivefilter_gpu_and_compare_with_cpu<64, 64>(const CpuTable& input,
	float filter_coeff_0, float filter_coeff_1, int num_kernel_runs,
	bool print_tables);
}
