#pragma once

namespace gpuacademy {
void calculate_summedareatable_cpu_naive(float* input_table, int num_rows, int num_cols, float* output_table);
void calculate_summedareatable_cpu_2dwise(float* input_table, int num_rows, int num_cols, float* output_table);
void apply_recursive_filter_cpu(float* input_table, int num_rows,
	int num_cols, float* filter_coeffs, int num_filter_coeffs, float* output_table);
}