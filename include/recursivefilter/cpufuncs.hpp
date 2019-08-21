#pragma once

#include "CpuTable.hpp"

namespace gpuacademy {
void calculate_summedareatable_cpu_naive(const CpuTable& input, CpuTable& output);
//void calculate_summedareatable_cpu_2dwise(const float* input_table, int num_rows, int num_cols, float* output_table);
void apply_right_down_recursive_filter_cpu(const CpuTable& input,
	float filter_coeff_0, float filter_coeff_1, CpuTable& output);
}