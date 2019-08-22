#include "gtest/gtest.h"

#include "helpers.hpp"

#include "gpufuncs.hpp"
#include "cpufuncs.hpp"
#include "CpuTable.hpp"
#include "utils.hpp"
#include "Logger.hpp"

using namespace gpuacademy;

TEST(recursivefiltering, checkmath_gpu_authorsinput) {
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
	
	const comparison_result_t comparison_result = apply_recursivefilter_gpu_and_compare_with_cpu<2, 2>(
		input, kSatFilterCoeffs[0], kSatFilterCoeffs[1], 1, true);
	ASSERT_TRUE(comparison_result.equals);
}

TEST(recursivefiltering, checkmath_gpu_oddnumcols) {
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

	const comparison_result_t comparison_result = apply_recursivefilter_gpu_and_compare_with_cpu<2, 2>
		(input, kSatFilterCoeffs[0], kSatFilterCoeffs[1], 1, true);
	ASSERT_TRUE(comparison_result.equals);
}

TEST(recursivefiltering, checkmath_gpu_oddnumrows) {
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

	const comparison_result_t comparison_result = apply_recursivefilter_gpu_and_compare_with_cpu<2, 2>
		(input, kSatFilterCoeffs[0], kSatFilterCoeffs[1], 1, true);
	ASSERT_TRUE(comparison_result.equals);
}

TEST(recursivefiltering, checkmath_gpu_oddnumcolsnumrows) {
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

	const comparison_result_t comparison_result = apply_recursivefilter_gpu_and_compare_with_cpu<2, 2>
		(input, kSatFilterCoeffs[0], kSatFilterCoeffs[1], 1, true);
	ASSERT_TRUE(comparison_result.equals);
}

TEST(recursivefiltering, checkmath_gpu_fractionfloats) {
	const int n_rows = 13, n_cols = 11;
	CpuTable input(n_rows, n_cols);
	input.setSawTooth(3.4f, 5);

	const comparison_result_t comparison_result = apply_recursivefilter_gpu_and_compare_with_cpu<2, 2>
		(input, kSatFilterCoeffs[0], kSatFilterCoeffs[1], 1, false);
	ASSERT_TRUE(comparison_result.equals);
}

TEST(recursivefiltering, checkmath_gpu_authorstablesize) {
	const int n_rows = 1024, n_cols = 1024;
	CpuTable input(n_rows, n_cols);
	input.setSawTooth(1.8f, 29);

	Logger::new_line("Authors' table size: (" + std::to_string(n_cols) + ", " + std::to_string(n_rows) + ")");
	Logger::new_line("Filter coeffs: " + std::to_string(kSatFilterCoeffs[0]) + ", " + std::to_string(kSatFilterCoeffs[1]));

	const comparison_result_t comparison_result = apply_recursivefilter_gpu_and_compare_with_cpu<64, 64>
		(input, kSatFilterCoeffs[0], kSatFilterCoeffs[1], 1, false);
	ASSERT_TRUE(comparison_result.equals);
}


TEST(recursivefiltering, checkmath_gpu_bigtable) {
	const int n_rows = 1000, n_cols = 1200;
	CpuTable input(n_rows, n_cols);
	input.setSawTooth(1.8f, 29);

	Logger::new_line("Big table size: (" + std::to_string(n_cols) + ", " + std::to_string(n_rows) + ")");
	Logger::new_line("Filter coeffs: " + std::to_string(kSatFilterCoeffs[0]) + ", " + std::to_string(kSatFilterCoeffs[1]));

	const comparison_result_t comparison_result = apply_recursivefilter_gpu_and_compare_with_cpu<64, 64>
		(input, kSatFilterCoeffs[0], kSatFilterCoeffs[1], 1, false);
	ASSERT_TRUE(comparison_result.equals);
}

TEST(recursivefiltering, checkmath_gpu_hugetable) {
	const int n_rows = 2000, n_cols = 4000;
	//const int n_rows = 1024, n_cols = 1024;
	CpuTable input(n_rows, n_cols);
	input.setSawTooth(1.8f, 29);

	Logger::new_line("Huge table size: (" + std::to_string(n_cols) + ", " + std::to_string(n_rows) + ")");
	Logger::new_line("Filter coeffs: " + std::to_string(kSatFilterCoeffs[0]) + ", " + std::to_string(kSatFilterCoeffs[1]));

	const comparison_result_t comparison_result = apply_recursivefilter_gpu_and_compare_with_cpu<64, 64>
		(input, kSatFilterCoeffs[0], kSatFilterCoeffs[1], 1, false);
	ASSERT_TRUE(comparison_result.equals);
}

TEST(recursivefiltering, measuretime_gpu_authorstablesize) {
	const int n_rows = 1024, n_cols = 1024;
	CpuTable input(n_rows, n_cols);
	input.setSawTooth(1.8f, 29);

	Logger::new_line("Authors' table size: (" + std::to_string(n_cols) + ", " + std::to_string(n_rows) + ")");
	Logger::new_line("Filter coeffs: " + std::to_string(kSatFilterCoeffs[0]) + ", " + std::to_string(kSatFilterCoeffs[1]));

	const comparison_result_t comparison_result = apply_recursivefilter_gpu_and_compare_with_cpu<64, 64>
		(input, kSatFilterCoeffs[0], kSatFilterCoeffs[1], 1, false);
	EXPECT_TRUE(comparison_result.equals);

	Logger::new_line("Execution time of recursive filtering kernel for "
		+ std::to_string(input.num_cols()) + "x" + std::to_string(input.num_rows())
		+ " [ms]: " + std::to_string(comparison_result.runtime_1kernel_ms) + "\n");
	ASSERT_LT(comparison_result.runtime_1kernel_ms, 0.5f);
}

TEST(recursivefiltering, measuretime_gpu_bigtable) {
	const int n_rows = 1000, n_cols = 1200;
	CpuTable input(n_rows, n_cols);
	input.setSawTooth(1.8f, 29);

	Logger::new_line("Big table size: (" + std::to_string(n_cols) + ", " + std::to_string(n_rows) + ")");
	Logger::new_line("Filter coeffs: " + std::to_string(kSatFilterCoeffs[0]) + ", " + std::to_string(kSatFilterCoeffs[1]));

	const comparison_result_t comparison_result = apply_recursivefilter_gpu_and_compare_with_cpu<64, 64>
		(input, kSatFilterCoeffs[0], kSatFilterCoeffs[1], 1, false);
	EXPECT_TRUE(comparison_result.equals);

	Logger::new_line("Execution time of recursive filtering kernel for "
		+ std::to_string(input.num_cols()) + "x" + std::to_string(input.num_rows())
		+ " [ms]: " + std::to_string(comparison_result.runtime_1kernel_ms) + "\n");
	ASSERT_LT(comparison_result.runtime_1kernel_ms, 0.5f);
}

TEST(recursivefiltering, measuretime_gpu_hugetable) {
	const int n_rows = 2000, n_cols = 4000;
	CpuTable input(n_rows, n_cols);
	input.setSawTooth(1.8f, 29);

	Logger::new_line("Huge table size: (" + std::to_string(n_cols) + ", " + std::to_string(n_rows) + ")");
	Logger::new_line("Filter coeffs: " + std::to_string(kSatFilterCoeffs[0]) + ", " + std::to_string(kSatFilterCoeffs[1]));

	const comparison_result_t comparison_result = apply_recursivefilter_gpu_and_compare_with_cpu<64, 64>
		(input, kSatFilterCoeffs[0], kSatFilterCoeffs[1], 1, false);
	EXPECT_TRUE(comparison_result.equals);

	Logger::new_line("Execution time of recursive filtering kernel for "
		+ std::to_string(input.num_cols()) + "x" + std::to_string(input.num_rows())
		+ " [ms]: " + std::to_string(comparison_result.runtime_1kernel_ms) + "\n");
	ASSERT_LT(comparison_result.runtime_1kernel_ms, 5.0f);
}
