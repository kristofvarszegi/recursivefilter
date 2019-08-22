#include "gtest/gtest.h"

#include "helpers.hpp"

#include "Logger.hpp"

using namespace gpuacademy;

TEST(recursivefiltering, CpuTable_setSawTooth) {
	const int n_rows = 3, n_cols = 4;
	CpuTable input(n_rows, n_cols);
	const float amplitude = 1.2f;
	const int period = 3;
	input.setSawTooth(amplitude, period);
	Logger::new_line("Sawtooth data:" + input.toString() + "\n");

	const float ground_truth_data[] = {
		-amplitude, 0.0f, amplitude, -amplitude,
		0.0f, amplitude, -amplitude, 0.0f,
		amplitude, -amplitude, 0.0f, amplitude
	};
	CpuTable ground_truth(n_rows, n_cols);
	for (int i_row = 0; i_row < n_rows; ++i_row) {
		for (int i_col = 0; i_col < n_cols; ++i_col) {
			ground_truth.set(i_row, i_col, static_cast<float>(ground_truth_data[i_col + i_row * n_cols]));
		}
	}
	Logger::new_line(ground_truth.toString());
	Logger::new_line(input.toString());
	Logger::new_line();
	Logger::new_line();

	ASSERT_TRUE(ground_truth.equals(input, kEpsilon));
}