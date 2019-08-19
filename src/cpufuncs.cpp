#include "cpufuncs.hpp"
#include "CpuTable.hpp"
#include "Logger.hpp"
#include "utils.hpp"

#include <cstdlib>
#include <stdexcept>
#include <time.h>
#include <iostream>

namespace gpuacademy {

void calculate_summedareatable_cpu_naive(const CpuTable& input_table, CpuTable& output_table) {
	if (input_table.num_rows() != output_table.num_rows()) {
		throw std::runtime_error("Number of table rows must match");
	}
	if (input_table.num_cols() != output_table.num_cols()) {
		throw std::runtime_error("Number of table cols must match");
	}

	clock_t t = clock();
	for (int i_row = 0; i_row < input_table.num_rows(); ++i_row) {
		for (int i_col = 0; i_col < input_table.num_cols(); ++i_col) {
			for (int i_row_this = 0; i_row_this <= i_row; ++i_row_this) {
				for (int i_col_this = 0; i_col_this <= i_col; ++i_col_this) {
					output_table.add(i_row, i_col, input_table.get(i_row_this, i_col_this));
				}
			}
		}
	}
	t = clock() - t;
	//Logger::new_line("Execution time of \"calculate_summedaretable_cpu_naive\" [ms]: "
	//	+ to_ms_str(t));
}

/*void calculate_summedareatable_cpu_2dwise(const float* input_table, int num_rows, int num_cols, float* output_table) {
	// In rows
	float* rowwise_sum_table = (float*)calloc(num_rows * num_cols, sizeof(float));
	for (int i_row = 0; i_row < num_rows; ++i_row) {
		for (int i_col = 0; i_col < num_cols; ++i_col) {
			for (int i_col_this = 0; i_col_this <= i_col; ++i_col_this) {
				rowwise_sum_table[i_col + i_row * num_cols] += input_table[i_col_this + i_row * num_cols];
			}
		}
	}

	// In cols
	for (int i_row = 0; i_row < num_rows; ++i_row) {
		for (int i_col = 0; i_col < num_cols; ++i_col) {
			for (int i_row_this = 0; i_row_this <= i_row; ++i_row_this) {
				output_table[i_col + i_row * num_cols] += rowwise_sum_table[i_col + i_row_this * num_cols];
			}
		}
	}
}*/

void apply_right_down_recursive_filter_cpu(const CpuTable& input_table,
	const float* filter_coeffs, int num_filter_coeffs, CpuTable& output_table) {
	if (input_table.num_rows() != output_table.num_rows()) {
		throw std::runtime_error("Number of table rows must match");
	}
	if (input_table.num_cols() != output_table.num_cols()) {
		throw std::runtime_error("Number of table cols must match");
	}
	if (num_filter_coeffs != 2) {
		throw std::runtime_error("Filter must have 2 coeffs");
	}

	clock_t t = clock();
	CpuTable rowwise_sum_table(input_table.num_rows(), input_table.num_cols());
	for (int i_row = 0; i_row < input_table.num_rows(); ++i_row) {
		for (int i_col = 0; i_col < input_table.num_cols(); ++i_col) {
			rowwise_sum_table.set(i_row, i_col, filter_coeffs[0] * input_table.get(i_row, i_col));
			if (i_col > 0) {
				rowwise_sum_table.add(i_row, i_col, filter_coeffs[1] * rowwise_sum_table.get(i_row, i_col - 1));
			}
			//rowwise_sum_table[i_col + i_row * num_rows] = input_table[i_col + i_row * num_rows];
		}
	}
	for (int i_row = 0; i_row < input_table.num_rows(); ++i_row) {
		for (int i_col = 0; i_col < input_table.num_cols(); ++i_col) {
			output_table.set(i_row, i_col, filter_coeffs[0] * rowwise_sum_table.get(i_row, i_col));
			if (i_row > 0) {
				output_table.add(i_row, i_col, filter_coeffs[1] * output_table.get(i_row - 1, i_col));
			}
			//output_table[i_col + i_row * num_rows] = rowwise_sum_table[i_col + i_row * num_rows];
		}
	}
	t = clock() - t;
	//Logger::new_line("Execution time of \"apply_recursive_filter_cpu\" [ms]: "
	//	+ to_ms_str(t));
}

}