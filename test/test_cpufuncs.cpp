#include "gtest/gtest.h"

#include "gpufuncs.hpp"
#include "cpufuncs.hpp"
#include "utils.hpp"
#include "Logger.hpp"

using namespace gpuacademy;

const static float kEpsilon = 0.000001f;
const static float kSatFilterCoeffs[] = { 1.0f, 1.0f };

TEST(recursivefiltering, calculate_summedareatable_cpu_naive) {
	clock_t t;
	const int n_rows = 3;
	const int n_cols = 4;
	const CpuTable input_table(n_rows, n_cols, 1.0f);
	const float reference_table_data[] = { 1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 4.0f, 6.0f, 8.0f, 3.0f, 6.0f, 9.0f, 12.0f, 4.0f, 8.0f, 12.0f, 16.0f };
	CpuTable reference_table(n_rows, n_cols);
	for (int i_row = 0; i_row < n_rows; ++i_row) {
		for (int i_col = 0; i_col < n_cols; ++i_col) {
			reference_table.set(i_row, i_col, reference_table_data[i_col + i_row * n_cols]);
		}
	}
	t = clock();
	CpuTable summed_area_table(n_rows, n_cols);
	calculate_summedareatable_cpu_naive(input_table, summed_area_table);
	//Logger::new_line(to_string(summed_area_table, n_rows, n_cols));
	//Logger::new_line("Time spent on \"calculate_summedareatable_cpu_naive\" [ms]: " + to_ms_str(t));
	//Logger::new_line();
	ASSERT_TRUE(reference_table.equals(summed_area_table, kEpsilon));
}

TEST(recursivefiltering, apply_recursive_filter_cpu) {
	clock_t t;
	int n_rows = 128;
	int n_cols = 64;
	const CpuTable input_table(n_rows, n_cols, 1.0f);
	CpuTable reference_table(n_rows, n_cols);
	calculate_summedareatable_cpu_naive(input_table, reference_table);

	t = clock();
	CpuTable summed_area_table(n_rows, n_cols);
	apply_right_down_recursive_filter_cpu(input_table, kSatFilterCoeffs,
		sizeof(kSatFilterCoeffs) / sizeof(float), summed_area_table);
	//Logger::new_line("Time spent on \"apply_recursive_filter_cpu\" [ms]: " + to_ms_str(t));

	ASSERT_TRUE(reference_table.equals(summed_area_table, kEpsilon));
}