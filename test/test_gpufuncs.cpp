#include "gtest/gtest.h"

#include "helpers.hpp"

#include "gpufuncs.hpp"
#include "cpufuncs.hpp"
#include "CpuTable.hpp"
#include "utils.hpp"
#include "Logger.hpp"

using namespace gpuacademy;

TEST(recursivefiltering, apply_recursive_filter_gpu_authorsinput) {
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
	
	ASSERT_EQ(apply_recursivefilter_gpu_and_compare_with_cpu(input, kSatFilterCoeffs[0],
		kSatFilterCoeffs[1], true), 0);
}

TEST(recursivefiltering, apply_recursive_filter_gpu_oddnumcols) {
	const int n_rows = 4, n_cols = 5;
	const int input_data[] = {
		1, 0, 0, 1, 1,
		2, 1, 1, 1, 2,
		1, 2, 1, 2, 0,
		1, 1, 0, 1, 1
	};
	CpuTable input(n_rows, n_cols);
	for (int i_row = 0; i_row < n_rows; ++i_row) {
		for (int i_col = 0; i_col < n_cols; ++i_col) {
			input.set(i_row, i_col, static_cast<float>(input_data[i_col + i_row * n_cols]));
		}
	}

	ASSERT_EQ(apply_recursivefilter_gpu_and_compare_with_cpu(input, kSatFilterCoeffs[0],
		kSatFilterCoeffs[1], true), 0);
}

TEST(recursivefiltering, apply_recursive_filter_gpu_oddnumrows) {
	const int n_rows = 5, n_cols = 4;
	const int input_data[] = {
		1, 0, 0, 1,
		2, 1, 1, 1,
		1, 2, 1, 2,
		1, 1, 0, 1,
		0, 1, 2, 1
	};
	CpuTable input(n_rows, n_cols);
	for (int i_row = 0; i_row < n_rows; ++i_row) {
		for (int i_col = 0; i_col < n_cols; ++i_col) {
			input.set(i_row, i_col, static_cast<float>(input_data[i_col + i_row * n_cols]));
		}
	}

	ASSERT_EQ(apply_recursivefilter_gpu_and_compare_with_cpu(input, kSatFilterCoeffs[0],
		kSatFilterCoeffs[1], true), 0);
}

TEST(recursivefiltering, apply_recursive_filter_gpu_oddnumcolsnumrows) {
	const int n_rows = 5, n_cols = 5;
	const int input_data[] = {
		1, 0, 0, 1, 1,
		2, 1, 1, 1, 2,
		1, 2, 1, 2, 0,
		1, 1, 0, 1, 1,
		0, 1, 2, 1, 2,
	};
	CpuTable input(n_rows, n_cols);
	for (int i_row = 0; i_row < n_rows; ++i_row) {
		for (int i_col = 0; i_col < n_cols; ++i_col) {
			input.set(i_row, i_col, static_cast<float>(input_data[i_col + i_row * n_cols]));
		}
	}

	ASSERT_EQ(apply_recursivefilter_gpu_and_compare_with_cpu(input, kSatFilterCoeffs[0],
		kSatFilterCoeffs[1], true), 0);
}

TEST(recursivefiltering, apply_recursive_filter_gpu_fractionfloats) {
	const int n_rows = 13, n_cols = 11;
	CpuTable input(n_rows, n_cols);
	input.setIncreasing(0.0f, 0.0015f);

	ASSERT_EQ(apply_recursivefilter_gpu_and_compare_with_cpu(input, kSatFilterCoeffs[0],
		kSatFilterCoeffs[1], false), 0);
}

TEST(recursivefiltering, apply_recursive_filter_gpu_bigtable) {
	const int n_rows = 1024, n_cols = 2048;
	CpuTable input(n_rows, n_cols);
	input.setIncreasing(0.0f, 0.0015f);

	Logger::new_line("Big table size: (" + std::to_string(n_cols) + ", " + std::to_string(n_rows) + ")");
	Logger::new_line("Filter coeffs: " + std::to_string(kSatFilterCoeffs[0]) + ", " + std::to_string(kSatFilterCoeffs[1]));
	ASSERT_EQ(apply_recursivefilter_gpu_and_compare_with_cpu(input, kSatFilterCoeffs[0],
		kSatFilterCoeffs[1], false), 0);
}
