#include <iostream>
#include <iomanip>

void calculate_summedaretable_cpu_naive(float* input_table, int num_rows, int num_cols, float* output_table) {
    for (int i_row = 0; i_row < num_rows; ++i_row) {
        for (int i_col = 0; i_col < num_cols; ++i_col) {
			for (int i_row_this = 0; i_row_this <= i_row; ++i_row_this) {
				for (int i_col_this = 0; i_col_this <= i_col; ++i_col_this) {
					output_table[i_col + i_row * num_rows] += input_table[i_col_this + i_row_this * num_rows];
				}
			}
        }
    }
}

void calculate_summedaretable_cpu_2dwise(float* input_table, int num_rows, int num_cols, float* output_table) {
	// In rows
	float* rowwise_sum_table = (float*)calloc(num_rows * num_cols, sizeof(float));
	for (int i_row = 0; i_row < num_rows; ++i_row) {
		for (int i_col = 0; i_col < num_cols; ++i_col) {
			for (int i_col_this = 0; i_col_this <= i_col; ++i_col_this) {
				rowwise_sum_table[i_col + i_row * num_rows] += input_table[i_col_this + i_row * num_rows];
			}
		}
	}

	// In cols
	for (int i_row = 0; i_row < num_rows; ++i_row) {
		for (int i_col = 0; i_col < num_cols; ++i_col) {
			for (int i_row_this = 0; i_row_this <= i_row; ++i_row_this) {
				output_table[i_col + i_row * num_rows] += rowwise_sum_table[i_col + i_row_this * num_rows];
			}
		}
	}
}

void calculate_summedaretable_cpu_filtery(float* input_table, int num_rows, int num_cols, float* output_table) {
	float filter[] = { 1.0f, 1.0f };

	// In rows
	float* rowwise_sum_table = (float*)calloc(num_rows * num_cols, sizeof(float));
	for (int i_row = 0; i_row < num_rows; ++i_row) {
		for (int i_col = 0; i_col < num_cols; ++i_col) {
			rowwise_sum_table[i_col + i_row * num_rows] = filter[0] * input_table[i_col + i_row * num_rows];
			if (i_col > 0) {
				rowwise_sum_table[i_col + i_row * num_rows] += filter[1] * rowwise_sum_table[i_col - 1 + i_row * num_rows];
			}
			//rowwise_sum_table[i_col + i_row * num_rows] = input_table[i_col + i_row * num_rows];
		}
	}

	// In cols
	for (int i_row = 0; i_row < num_rows; ++i_row) {
		for (int i_col = 0; i_col < num_cols; ++i_col) {
			output_table[i_col + i_row * num_rows] = filter[0] * rowwise_sum_table[i_col + i_row * num_rows];
			if (i_row > 0) {
				output_table[i_col + i_row * num_rows] += filter[1] * output_table[i_col + (i_row - 1) * num_rows];
			}
			//output_table[i_col + i_row * num_rows] = rowwise_sum_table[i_col + i_row * num_rows];
		}
	}
}

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
}

int main(int argc, char* argv[]) {
	std::cout << "sizeof(float): " << sizeof(float) << std::endl;
	std::cout << "sizeof(int): " << sizeof(int) << std::endl;

    //run_gpu_imageprocessing_example();

    int n_table_rows = 10;
    int n_table_cols = 10;
	float* input_table = (float*)calloc(n_table_rows * n_table_cols, sizeof(float));
	fill_with_ones(input_table, n_table_rows, n_table_cols);

	std::cout << std::endl << std::endl << "Input table:" << std::endl;
	print_table(input_table, n_table_rows, n_table_cols);
	
	float* summed_area_table;

	summed_area_table = (float*)calloc(n_table_rows * n_table_cols, sizeof(float));
	calculate_summedaretable_cpu_naive(input_table, n_table_rows, n_table_cols, summed_area_table);
	std::cout << std::endl << std::endl << "SAT - Naive:";
    print_table(summed_area_table, n_table_rows, n_table_cols);

	summed_area_table = (float*)calloc(n_table_rows * n_table_cols, sizeof(float));
	calculate_summedaretable_cpu_2dwise(input_table, n_table_rows, n_table_cols, summed_area_table);
	std::cout << std::endl << std::endl << "SAT - 2D-wise:";
	print_table(summed_area_table, n_table_rows, n_table_cols);

	summed_area_table = (float*)calloc(n_table_rows * n_table_cols, sizeof(float));
	calculate_summedaretable_cpu_filtery(input_table, n_table_rows, n_table_cols, summed_area_table);
	std::cout << std::endl << std::endl << "SAT - Filtery:";
	print_table(summed_area_table, n_table_rows, n_table_cols);

    return 0;
}