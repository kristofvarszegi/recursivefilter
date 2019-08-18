#include "cpu_funcs.hpp"

#include <cstdlib>
#include <stdexcept>
#include <time.h>
#include <iostream>

namespace gpuacademy {

void calculate_summedareatable_cpu_naive(float* input_table, int num_rows, int num_cols, float* output_table) {
	clock_t t = clock();
	for (int i_row = 0; i_row < num_rows; ++i_row) {
		for (int i_col = 0; i_col < num_cols; ++i_col) {
			for (int i_row_this = 0; i_row_this <= i_row; ++i_row_this) {
				for (int i_col_this = 0; i_col_this <= i_col; ++i_col_this) {
					output_table[i_col + i_row * num_rows] += input_table[i_col_this + i_row_this * num_rows];
				}
			}
		}
	}
	t = clock() - t;
	std::cout << std::endl << "Execution time of \"calculate_summedaretable_cpu_naive\" [ms]: "
		<< ((float)t) / (float)CLOCKS_PER_SEC * 1000.0f << std::endl;
}

void calculate_summedareatable_cpu_2dwise(float* input_table, int num_rows, int num_cols, float* output_table) {
	// In rows
	float* rowwise_sum_table = (float*)calloc(num_rows * num_cols, sizeof(float));
	for (int i_row = 0; i_row < num_rows; ++i_row) {
		for (int i_col = 0; i_col < num_cols; ++i_col) {
			for (int i_col_this = 0; i_col_this <= i_col; ++i_col_this) {
				rowwise_sum_table[i_col + i_row * num_rows] += input_table[i_col_this + i_row * num_rows];
			}
		}
	}

	// In cols
	for (int i_row = 0; i_row < num_rows; ++i_row) {
		for (int i_col = 0; i_col < num_cols; ++i_col) {
			for (int i_row_this = 0; i_row_this <= i_row; ++i_row_this) {
				output_table[i_col + i_row * num_rows] += rowwise_sum_table[i_col + i_row_this * num_rows];
			}
		}
	}
}

void apply_recursive_filter_cpu(float* input_table, int num_rows,
	int num_cols, float* filter_coeffs, int num_filter_coeffs, float* output_table) {
	if (num_filter_coeffs != 2) {
		throw std::runtime_error("Filter must have 2 coeffs");
	}

	clock_t t = clock();
	float* rowwise_sum_table = (float*)calloc(num_rows * num_cols, sizeof(float));
	for (int i_row = 0; i_row < num_rows; ++i_row) {
		for (int i_col = 0; i_col < num_cols; ++i_col) {
			rowwise_sum_table[i_col + i_row * num_rows] = filter_coeffs[0] * input_table[i_col + i_row * num_rows];
			if (i_col > 0) {
				rowwise_sum_table[i_col + i_row * num_rows] += filter_coeffs[1] * rowwise_sum_table[i_col - 1 + i_row * num_rows];
			}
			//rowwise_sum_table[i_col + i_row * num_rows] = input_table[i_col + i_row * num_rows];
		}
	}
	for (int i_row = 0; i_row < num_rows; ++i_row) {
		for (int i_col = 0; i_col < num_cols; ++i_col) {
			output_table[i_col + i_row * num_rows] = filter_coeffs[0] * rowwise_sum_table[i_col + i_row * num_rows];
			if (i_row > 0) {
				output_table[i_col + i_row * num_rows] += filter_coeffs[1] * output_table[i_col + (i_row - 1) * num_rows];
			}
			//output_table[i_col + i_row * num_rows] = rowwise_sum_table[i_col + i_row * num_rows];
		}
	}
	t = clock() - t;
	std::cout << std::endl << "Execution time of \"apply_recursive_filter_cpu\" [ms]: "
		<< ((float)t) / (float)CLOCKS_PER_SEC * 1000.0f << std::endl;
}

}