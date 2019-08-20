#include "gtest/gtest.h"

#include "gpufuncs.hpp"
#include "cpufuncs.hpp"
#include "CpuTable.hpp"
#include "utils.hpp"
#include "Logger.hpp"

using namespace gpuacademy;

const static float kEpsilon = 0.000001f;

TEST(recursivefiltering, apply_recursive_filter_gpu) {
	clock_t t;
	const int n_rows = 11, n_cols = 7;
	//const int n_rows = 1024, n_cols = 2048;
	const CpuTable input_table(n_rows, n_cols, 1.0f);
	
	CpuTable summed_area_table(n_rows, n_cols);
	float sat_filter_coeffs[] = { 1.0f, 1.0f };
	apply_right_down_recursive_filter_gpu(input_table, sat_filter_coeffs,
		summed_area_table);

	t = clock();
	CpuTable reference_table(n_rows, n_cols);
	//Logger::new_line("Calculating SAT CPU Naive for reference...");
	//calculate_summedareatable_cpu_naive(input_table, reference_table);
	Logger::new_line("Calculating SAT CPU for reference...");
	apply_right_down_recursive_filter_cpu(input_table, sat_filter_coeffs,
		reference_table);
	//Logger::new_line("Time spent on \"calculate_summedareatable_cpu_naive\" [ms]: " + to_ms_str(t));
	//Logger::new_line();
	if (n_cols <= 12 && n_rows <= 12) {
		Logger::new_line(input_table.toString());
		Logger::new_line(reference_table.toString());
		Logger::new_line(summed_area_table.toString());
		Logger::new_line();
	}

	Logger::new_line();
	ASSERT_TRUE(reference_table.equals(summed_area_table, kEpsilon));
}
