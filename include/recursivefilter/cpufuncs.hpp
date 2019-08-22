#pragma once

#include "CpuTable.hpp"

namespace gpuacademy {
void calculate_summedareatable_cpu_naive(const CpuTable& input, CpuTable& output);
//void calculate_summedareatable_cpu_2dwise(const float* input_table, int num_rows, int num_cols, float* output_table);
void recursivefilter_downright_cpu(const CpuTable& input,
	float filter_coeff_0, float filter_coeff_1, CpuTable& output);
}