#include "gtest/gtest.h"

#include "helpers.hpp"

#include "gpufuncs.hpp"
#include "cpufuncs.hpp"
#include "utils.hpp"
#include "Logger.hpp"

using namespace gpuacademy;

TEST(recursivefiltering, sat_cpu_naive) {
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
	Logger::new_line("SAT-naive table:" + summed_area_table.toString());
	Logger::new_line();
	Logger::new_line();

	ASSERT_TRUE(ground_truth.equals(summed_area_table, kEpsilon));
}

TEST(recursivefiltering, sat_cpu_filtery) {
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
	recursivefilter_downright_cpu(input, kSatFilterCoeffs[0],
		kSatFilterCoeffs[1], summed_area_table);
	Logger::new_line("SAT-filtered table:" + summed_area_table.toString());
	Logger::new_line();
	Logger::new_line();

	ASSERT_TRUE(ground_truth.equals(summed_area_table, kEpsilon));
}

TEST(recursivefiltering, arbitratyfilter_cpu) {
	const float filter_coeffs[] = {0.5f, 0.25f};
	
	const int n_rows = 5, n_cols = 5;
	CpuTable input(n_rows, n_cols, 1.0f);

	const float ground_truth_data[] = {
		0.25f,	0.3125f,	0.328125f,	0.33203125f,	0.3330078125f,
		0.3125f,	0.390625f,	0.41015625f,	0.4150390625f,	0.416259765625f,
		0.328125f,	0.41015625f,	0.4306640625f,	0.435791015625f,	0.43707275390625f,
		0.33203125f,	0.4150390625f,	0.435791015625f,	0.44097900390625f,	0.442276000976562f,
		0.3330078125f,	0.416259765625f,	0.43707275390625f,	0.442276000976562f,	0.443576812744141f,
	};
	CpuTable ground_truth(n_rows, n_cols);
	for (int i_row = 0; i_row < n_rows; ++i_row) {
		for (int i_col = 0; i_col < n_cols; ++i_col) {
			ground_truth.set(i_row, i_col, ground_truth_data[i_col + i_row * n_cols]);
		}
	}

	CpuTable filtered_table(n_rows, n_cols);
	recursivefilter_downright_cpu(input, filter_coeffs[0],
		filter_coeffs[1], filtered_table);
	Logger::new_line("Filtered table:" + filtered_table.toString());
	Logger::new_line();
	Logger::new_line();

	ASSERT_TRUE(ground_truth.equals(filtered_table, kEpsilon));
}