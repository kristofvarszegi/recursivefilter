#include "gpu_funcs.hpp"
#include "cpu_funcs.hpp"
#include "utils.hpp"

#include <time.h>
#include <iostream>

using namespace gpuacademy;

int main(int argc, char* argv[]) {
	std::cout << "sizeof(float): " << sizeof(float) << std::endl;
	std::cout << "sizeof(int): " << sizeof(int) << std::endl;
	clock_t t;

    int n_rows = 512;
    int n_cols = 512;
	float* input_table = (float*)calloc(n_rows * n_cols, sizeof(float));
	t = clock();
	fill_with_ones(input_table, n_rows, n_cols);
	t = clock() - t;
	std::cout << std::endl << "Time spent [ms]: " << ((float)t) / (float)CLOCKS_PER_SEC * 1000.0f << std::endl;

	//std::cout << std::endl << std::endl << "Input table:" << std::endl;
	//print_table(input_table, n_rows, n_cols);
	
	float* summed_area_table;
	float sat_filter_coeffs[] = { 1.0f, 1.0f };

#if 0
	summed_area_table = (float*)calloc(n_rows * n_cols, sizeof(float));
	calculate_summedareatable_cpu_naive(input_table, n_rows, n_cols, summed_area_table);
	//std::cout << std::endl << std::endl << "SAT - CPU, naive:";
    //print_table(summed_area_table, n_rows, n_cols);
#endif

#if 0
	summed_area_table = (float*)calloc(n_rows * n_cols, sizeof(float));
	calculate_summedaretable_cpu_2dwise(input_table, n_rows, n_cols, summed_area_table);
	std::cout << std::endl << std::endl << "SAT - 2D-wise:";
	print_table(summed_area_table, n_rows, n_cols);
#endif

	summed_area_table = (float*)calloc(n_rows * n_cols, sizeof(float));
	apply_recursive_filter_cpu(input_table, n_rows, n_cols, sat_filter_coeffs,
		sizeof(sat_filter_coeffs) / sizeof(float), summed_area_table);
	//std::cout << std::endl << std::endl << "SAT - CPU, filtery:";
	//print_table(summed_area_table, n_rows, n_cols);

	summed_area_table = (float*)calloc(n_rows * n_cols, sizeof(float));
	apply_recursive_filter_gpu(input_table, n_rows, n_cols, sat_filter_coeffs,
		sizeof(sat_filter_coeffs) / sizeof(float), summed_area_table);
	//std::cout << std::endl << std::endl << "SAT - GPU:";
	//print_table(summed_area_table, n_rows, n_cols);

    return 0;
}