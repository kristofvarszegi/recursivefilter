#pragma once

#include "CpuTable.hpp"

namespace gpuacademy {
void calculate_summedareatable_cpu_naive(const CpuTable& input_table, CpuTable& output_table);
//void calculate_summedareatable_cpu_2dwise(const float* input_table, int num_rows, int num_cols, float* output_table);
void apply_right_down_recursive_filter_cpu(const CpuTable& input_table,
	const float* filter_coeffs, int num_filter_coeffs, CpuTable& output_table);
}