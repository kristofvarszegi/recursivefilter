#include "gtest/gtest.h"

#include "helpers.hpp"

#include "Logger.hpp"

using namespace gpuacademy;

TEST(recursivefiltering, CpuTable_setIncreasing) {
	const int n_rows = 3, n_cols = 4;
	CpuTable input(n_rows, n_cols);
	input.setIncreasing(-0.003f, 0.0015f);

	const float ground_truth_data[] = {
		 -0.003f, -0.0015f,  0.000f,  0.0015f,
		  0.003f,  0.0045f,  0.006f,  0.0075f,
		  0.009f,  0.0105f,  0.012f,  0.0135f
	};
	CpuTable ground_truth(n_rows, n_cols);
	for (int i_row = 0; i_row < n_rows; ++i_row) {
		for (int i_col = 0; i_col < n_cols; ++i_col) {
			ground_truth.set(i_row, i_col, static_cast<float>(ground_truth_data[i_col + i_row * n_cols]));
		}
	}
	Logger::new_line(ground_truth.toString());
	Logger::new_line(input.toString());

	ASSERT_TRUE(ground_truth.equals(input, kEpsilon));
}