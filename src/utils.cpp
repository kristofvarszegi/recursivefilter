#include "utils.hpp"

#include <iostream>
#include <iomanip>

namespace gpuacademy {

void fill_with_ones(float* table, int num_rows, int num_cols) {
	for (int i_row = 0; i_row < num_rows; ++i_row) {
		for (int i_col = 0; i_col < num_cols; ++i_col) {
			table[i_col + i_row * num_rows] = 1.0f;
		}
	}
}

void print_table(float* table, int num_rows, int num_cols) {
	for (int i_row = 0; i_row < num_rows; ++i_row) {
		std::cout << std::endl;
		for (int i_col = 0; i_col < num_cols; ++i_col) {
			std::cout << std::setw(6) << table[i_col + i_row * num_rows];
		}
	}
	std::cout << std::endl;
}

}