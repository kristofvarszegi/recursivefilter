#pragma once

namespace gpuacademy {
void apply_recursive_filter_gpu(float* input_table, int num_rows, int num_cols,
	float* filter_coeffs, int num_filter_coeffs, float* output_table);
}