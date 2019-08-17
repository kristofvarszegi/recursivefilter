#include <iostream>
#include <iomanip>

void calculate_summedaretable_cpu(float* table, int num_rows, int num_cols, float* summed_area_table) {
    for (int i_row = 0; i_row < num_rows; ++i_row) {
        for (int i_col = 0; i_col < num_cols; ++i_col) {
			for (int i_row_this = 0; i_row_this <= i_row; ++i_row_this) {
				for (int i_col_this = 0; i_col_this <= i_col; ++i_col_this) {
					summed_area_table[i_col + i_row * num_rows] += table[i_col_this + i_row_this * num_rows];
				}
			}
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
	float* table = (float*)calloc(n_table_rows * n_table_cols, sizeof(float));
	fill_with_ones(table, n_table_rows, n_table_cols);

	std::cout << std::endl << "Input table:" << std::endl;
	print_table(table, n_table_rows, n_table_cols);

	float* summed_area_table = (float*)calloc(n_table_rows * n_table_cols, sizeof(float));
	calculate_summedaretable_cpu(table, n_table_rows, n_table_cols, summed_area_table);

	std::cout << std::endl << "SAT:" << std::endl;
    print_table(summed_area_table, n_table_rows, n_table_cols);

    return 0;
}