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

	for (int i_row = 0; i_row < input_table.num_rows(); ++i_row) {
		if (i_row % 50 == 0) {
			Logger::new_line("Calculating row " + std::to_string(i_row + 1)
				+ "/" + std::to_string(input_table.num_rows()) + "...");
		}
		for (int i_col = 0; i_col < input_table.num_cols(); ++i_col) {
			for (int i_row_this = 0; i_row_this <= i_row; ++i_row_this) {
				for (int i_col_this = 0; i_col_this <= i_col; ++i_col_this) {
					output_table.add(i_row, i_col, input_table.get(i_row_this, i_col_this));
				}
			}
		}
	}
}

void recursivefilter_downright_cpu(const CpuTable& input_table,
	float filter_coeff_0, float filter_coeff_1, CpuTable& output_table) {
	if (input_table.num_rows() != output_table.num_rows()) {
		throw std::runtime_error("Number of table rows must match");
	}
	if (input_table.num_cols() != output_table.num_cols()) {
		throw std::runtime_error("Number of table cols must match");
	}

	CpuTable colwise_sum_table(input_table.num_rows(), input_table.num_cols());
	for (int i_col = 0; i_col < input_table.num_cols(); ++i_col) {
		for (int i_row = 0; i_row < input_table.num_rows(); ++i_row) {
			colwise_sum_table.set(i_row, i_col, filter_coeff_0 * input_table.get(i_row, i_col));
			if (i_row > 0) {
				colwise_sum_table.add(i_row, i_col, filter_coeff_1 * colwise_sum_table.get(i_row - 1, i_col));
			}
			output_table.set(i_row, i_col, colwise_sum_table.get(i_row, i_col));
		}
	}
	for (int i_row = 0; i_row < input_table.num_rows(); ++i_row) {
		for (int i_col = 0; i_col < input_table.num_cols(); ++i_col) {
			output_table.set(i_row, i_col, filter_coeff_0 * colwise_sum_table.get(i_row, i_col));
			if (i_col > 0) {
				output_table.add(i_row, i_col, filter_coeff_1 * output_table.get(i_row, i_col - 1));
			}
		}
	}
}

}