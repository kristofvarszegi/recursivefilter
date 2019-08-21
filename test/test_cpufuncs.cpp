#include "gtest/gtest.h"

#include "helpers.hpp"

#include "gpufuncs.hpp"
#include "cpufuncs.hpp"
#include "utils.hpp"
#include "Logger.hpp"

using namespace gpuacademy;

TEST(recursivefiltering, calculate_summedareatable_cpu_naive) {
	const int n_rows = 6, n_cols = 6;
	const int input_data[] = {
		1, 0, 0, 1, 1, 1,
		2, 1, 1, 1, 2, 1,
		1, 2, 1, 2, 0, 1,
		1, 1, 0, 1, 1, 2,
		0, 1, 2, 1, 2, 1,
		1, 0, 1, 0, 1, 1
	};
	CpuTable input(n_rows, n_cols);
	for (int i_row = 0; i_row < n_rows; ++i_row) {
		for (int i_col = 0; i_col < n_cols; ++i_col) {
			input.set(i_row, i_col, static_cast<float>(input_data[i_col + i_row * n_cols]));
		}
	}

	const int ground_truth_data[] = {
		 1,  1,  1,  2,  3,  4,
		 3,  4,  5,  7, 10, 12,
		 4,  7,  9, 13, 16, 19,
		 5,  9, 11, 16, 20, 25,
		 5, 10, 14, 20, 26, 32,
		 6, 11, 16, 22, 29, 36
	};
	CpuTable ground_truth(n_rows, n_cols);
	for (int i_row = 0; i_row < n_rows; ++i_row) {
		for (int i_col = 0; i_col < n_cols; ++i_col) {
			ground_truth.set(i_row, i_col, static_cast<float>(ground_truth_data[i_col + i_row * n_cols]));
		}
	}

	CpuTable summed_area_table(n_rows, n_cols);
	calculate_summedareatable_cpu_naive(input, summed_area_table);

	ASSERT_TRUE(ground_truth.equals(summed_area_table, kEpsilon));
}

TEST(recursivefiltering, apply_recursive_filter_cpu) {
	const int n_rows = 6, n_cols = 6;
	const int input_data[] = {
		1, 0, 0, 1, 1, 1,
		2, 1, 1, 1, 2, 1,
		1, 2, 1, 2, 0, 1,
		1, 1, 0, 1, 1, 2,
		0, 1, 2, 1, 2, 1,
		1, 0, 1, 0, 1, 1
	};
	CpuTable input(n_rows, n_cols);
	for (int i_row = 0; i_row < n_rows; ++i_row) {
		for (int i_col = 0; i_col < n_cols; ++i_col) {
			input.set(i_row, i_col, static_cast<float>(input_data[i_col + i_row * n_cols]));
		}
	}

	CpuTable ground_truth(n_rows, n_cols);
	calculate_summedareatable_cpu_naive(input, ground_truth);

	CpuTable summed_area_table(n_rows, n_cols);
	apply_right_down_recursive_filter_cpu(input, kSatFilterCoeffs[0],
		kSatFilterCoeffs[1], summed_area_table);

	ASSERT_TRUE(ground_truth.equals(summed_area_table, kEpsilon));
}