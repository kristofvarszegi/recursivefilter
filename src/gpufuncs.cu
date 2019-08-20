#include "gpufuncs.hpp"
#include "Logger.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>

namespace gpuacademy {

const int kNumRuns = 5;
const int kTableBlockDimX = 2, kTableBlockDimY = 2;
//const int kTableBlockDimX = 32, kTableBlockDimY = 32;
const dim3 kThreadBlockDimInRows = dim3(1, kTableBlockDimY);
const dim3 kThreadBlockDimInCols = dim3(kTableBlockDimX , 1);
const dim3 kThreadBlockDimInColsBlocky = dim3(kTableBlockDimX, 1);
const dim3 kThreadBlockDimInColsInRowsBlocky = dim3(kTableBlockDimX, 1);

__global__ void copy(const float* input_table, int num_rows,
	int num_cols, float* output_table) {
	int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
	int thread_id_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (thread_id_x < num_cols && thread_id_y < num_rows) {
		output_table[thread_id_x + thread_id_y * num_cols] = input_table[thread_id_x + thread_id_y * num_cols];
	}
}

// One thread calculates one row
__global__ void apply_recursive_filter_right(const float* input_table, int num_rows,
	int num_cols, float filter_coeff_0, float filter_coeff_1, float* output_table) {
	const int global_tid = blockIdx.y * blockDim.y + threadIdx.y;

	if (global_tid < num_rows) {
		float filtered_val, prev_filtered_val;
#pragma unroll
		for (int i_col = 0; i_col < num_cols; ++i_col) {
			filtered_val = filter_coeff_0 * input_table[i_col + global_tid * num_cols];
			if (i_col > 0) {
				filtered_val += filter_coeff_1 * prev_filtered_val;
			}
			output_table[i_col + global_tid * num_cols] = filtered_val;
			prev_filtered_val = filtered_val;
		}
	}
}

// One thread calculates one col
__global__ void apply_recursive_filter_down(const float* input_table, int num_rows,
	int num_cols, float filter_coeff_0, float filter_coeff_1, float* output_table) {
	const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (global_tid < num_cols) {
		float filtered_val, prev_filtered_val;
#pragma unroll
		for (int i_row = 0; i_row < num_rows; ++i_row) {
			filtered_val = filter_coeff_0 * input_table[global_tid + i_row * num_cols];
			if (i_row > 0) {
				filtered_val += filter_coeff_1 * prev_filtered_val;
			}
			output_table[global_tid + i_row * num_cols] = filtered_val;
			prev_filtered_val = filtered_val;
		}
	}
}

#if 0
// One thread calculates one row within a table block
__global__ void apply_recursive_filter_right_blocky(const float* input_table, int num_rows,
	int num_cols, const float* filter_coeffs, float* output_table) {
	const int global_tid = blockIdx.y * blockDim.z + threadIdx.z;
	
	const int tableblock_start_x_global = blockIdx.x * kTableBlockDimOneSide;
	const int y_in_tableblock = global_tid % kTableBlockDimOneSide;
	__shared__ float rowwise_sums_tableblock[kTableBlockDimOneSide * kTableBlockDimOneSide];
	//printf("\n%d, %d, %d, %d", blockIdx.x, blockDim.x, threadIdx.x, tableblock_start_x);
	if (global_tid < num_rows) {
		#pragma unroll
		for (int x_in_tableblock = 0; x_in_tableblock < kTableBlockDimOneSide; ++x_in_tableblock) {
			if (tableblock_start_x_global + x_in_tableblock < num_cols) {
				const int id_in_tableblock = x_in_tableblock + y_in_tableblock * kTableBlockDimOneSide;
				rowwise_sums_tableblock[id_in_tableblock] =
					filter_coeffs[0] * input_table[tableblock_start_x_global + x_in_tableblock + global_tid * num_cols];
				if (x_in_tableblock > 0) {
					rowwise_sums_tableblock[id_in_tableblock] +=
						filter_coeffs[1] * rowwise_sums_tableblock[(x_in_tableblock - 1) + y_in_tableblock * kTableBlockDimOneSide];
				}
			}
		}
		__syncthreads();
		for (int x_in_tableblock = 0; x_in_tableblock < kTableBlockDimOneSide; ++x_in_tableblock) {
			if (tableblock_start_x_global + x_in_tableblock < num_cols) {
				output_table[tableblock_start_x_global + x_in_tableblock + global_tid * num_cols] =
					rowwise_sums_tableblock[x_in_tableblock + y_in_tableblock * kTableBlockDimOneSide];
			}
		}
	}
}
#endif

// One thread calculates one col within a table block
__global__ void apply_recursive_filter_down_blocky(const float* input_table, int num_rows,
	int num_cols, float filter_coeff_0, float filter_coeff_1, float* incolsum_table) {
	const int global_tid_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (global_tid_x < num_cols) {
		float colwise_sum, prev_colwise_sum;
		const int tableblock_start_y_global = blockIdx.y * kTableBlockDimY;
		//const int x_in_tableblock = global_tid_x % kTableBlockDimInColsX;
		const int x_in_tableblock = global_tid_x & (kTableBlockDimX - 1);	// Modulo with power of 2

		#pragma unroll
		for (int y_in_tableblock = 0; y_in_tableblock < kTableBlockDimY; ++y_in_tableblock) {
			if (tableblock_start_y_global + y_in_tableblock < num_rows) {
				colwise_sum = filter_coeff_0 * input_table[global_tid_x + (tableblock_start_y_global + y_in_tableblock) * num_cols];
				if (y_in_tableblock > 0) {
					colwise_sum += filter_coeff_1 * prev_colwise_sum;
				}
				prev_colwise_sum = colwise_sum;
				incolsum_table[global_tid_x + blockIdx.y * num_cols] = colwise_sum;
				//output_table[global_tid_x + (tableblock_start_y_global + y_in_tableblock) * num_cols] = colwise_sum;
			}
		}
	}
}

__global__ void apply_recursive_filter_downright_blocky_part1(
	const float* input_table, int num_rows, int num_cols, float filter_coeff_0,
	float filter_coeff_1, float* blockwise_colwise_sums,
	float* blockwise_rowwise_sums) {
	const int global_tid_x = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float colwise_sums[kTableBlockDimX * kTableBlockDimY];
	__shared__ float rowwise_sums[kTableBlockDimY];

	const int tableblock_start_y_global = blockIdx.y * kTableBlockDimY;
	if (global_tid_x < num_cols) {
		//const int x_in_tableblock = global_tid_x % kTableBlockDimInColsX;
		//const int x_in_tableblock = global_tid_x & (kTableBlockDimX - 1);	// Modulo with power of 2
		const int x_in_tableblock = threadIdx.x;

#pragma unroll
		for (int y_in_tableblock = 0; y_in_tableblock < kTableBlockDimY; ++y_in_tableblock) {
			if (tableblock_start_y_global + y_in_tableblock <= num_rows) {
				const int id_in_tableblock = x_in_tableblock + y_in_tableblock * kTableBlockDimX;
				colwise_sums[id_in_tableblock] = 0.0f;
				if (tableblock_start_y_global + y_in_tableblock < num_rows) {
					colwise_sums[id_in_tableblock] = filter_coeff_0
						* input_table[global_tid_x + (tableblock_start_y_global + y_in_tableblock) * num_cols];
				}
				if (y_in_tableblock > 0) {
					colwise_sums[id_in_tableblock] += filter_coeff_1 *
						colwise_sums[x_in_tableblock + (y_in_tableblock - 1) * kTableBlockDimX];
				}
			}
		}
		__syncthreads();
		blockwise_colwise_sums[global_tid_x + blockIdx.y * num_cols] =
			colwise_sums[threadIdx.x + (kTableBlockDimY - 1) * kTableBlockDimX];
	}

	if (threadIdx.x < kTableBlockDimY) {
		const int tableblock_start_x_global = blockIdx.x * kTableBlockDimX;
		const int y_in_tableblock = threadIdx.x;	// Each thread does a row
		for (int x_in_tableblock = 0; x_in_tableblock < kTableBlockDimX; ++x_in_tableblock) {
			if (tableblock_start_x_global + x_in_tableblock <= num_cols) {
				if (tableblock_start_x_global + x_in_tableblock < num_cols) {
					rowwise_sums[y_in_tableblock] = filter_coeff_0
						* colwise_sums[x_in_tableblock + y_in_tableblock * kTableBlockDimX];
				}
				if (x_in_tableblock > 0) {
					rowwise_sums[y_in_tableblock] += filter_coeff_1 *
						colwise_sums[(x_in_tableblock - 1) + y_in_tableblock * kTableBlockDimX];
				}
			}
		}
		__syncthreads();
		blockwise_rowwise_sums[blockIdx.x + (tableblock_start_y_global + threadIdx.x) * gridDim.x] =
			rowwise_sums[threadIdx.x];
	}
}

void apply_right_down_recursive_filter_gpu(const CpuTable& input_table,
	const float* filter_coeffs, CpuTable& output_table) {
	if (input_table.num_rows() != output_table.num_rows()) {
		throw std::runtime_error("Number of table rows must match");
	}
	if (input_table.num_cols() != output_table.num_cols()) {
		throw std::runtime_error("Number of table cols must match");
	}

	Logger::new_line("Input table dims: (" + std::to_string(input_table.num_cols())
		+ ", " + std::to_string(input_table.num_rows()) + ")");

	float* d_input_table_data;
	cudaMalloc((void**)(&d_input_table_data), input_table.num_rows() * input_table.num_cols() * sizeof(float));
	cudaMemcpy(d_input_table_data, input_table.data(), input_table.num_cols() * input_table.num_rows() * sizeof(float), cudaMemcpyHostToDevice);

	float* d_blockwise_colwise_sums;
	const int n_blockwise_colwise_rows = input_table.num_rows() % kTableBlockDimY == 0 ?
		input_table.num_rows() / kTableBlockDimY : input_table.num_rows() / kTableBlockDimY + 1;
	cudaMalloc((void**)(&d_blockwise_colwise_sums), input_table.num_cols() * n_blockwise_colwise_rows * sizeof(float));
	Logger::new_line("Blockwise-colwise table dims: (" + std::to_string(input_table.num_cols())
		+ ", " + std::to_string(n_blockwise_colwise_rows) + ")");

	float* d_blockwise_rowwise_sums;
	const int n_d_blockwise_rowwise_cols = input_table.num_cols() % kTableBlockDimX == 0 ?
		input_table.num_cols() / kTableBlockDimX : input_table.num_cols() / kTableBlockDimX + 1;
	cudaMalloc((void**)(&d_blockwise_rowwise_sums), n_d_blockwise_rowwise_cols * input_table.num_rows() * sizeof(float));
	Logger::new_line("Blockwise-rowwise table dims: (" + std::to_string(n_d_blockwise_rowwise_cols)
		+ ", " + std::to_string(input_table.num_rows()) + ")");
	
	float* d_output_table_data;
	cudaMalloc((void**)(&d_output_table_data), input_table.num_rows() * input_table.num_cols() * sizeof(float));
	
	//dim3 thread_grid_dim_inrows(1, input_table.num_rows() / kThreadBlockDimInRows.y);
	//if (input_table.num_rows() % kThreadBlockDimInRows.y != 0) { thread_grid_dim_inrows.y += 1; }
	//dim3 thread_grid_dim_incols(input_table.num_cols() / kThreadBlockDimInCols.x, 1);
	//if (input_table.num_cols() % kThreadBlockDimInCols.x != 0) { thread_grid_dim_incols.x += 1; }
	//dim3 thread_grid_dim_incols_blocky(input_table.num_cols() / kTableBlockDimX,
	//	input_table.num_rows() / kTableBlockDimY);
	//if (input_table.num_cols() % kTableBlockDimX != 0) { thread_grid_dim_incols_blocky.x += 1; }
	//if (input_table.num_rows() % kTableBlockDimY != 0) { thread_grid_dim_incols_blocky.y += 1; }

	dim3 thread_grid_dim_incolsinrows_blocky(input_table.num_cols() / kThreadBlockDimInColsInRowsBlocky.x,
		input_table.num_rows() / kThreadBlockDimInColsInRowsBlocky.x);	// Same array of threads do the cols and the rows as well
	if (input_table.num_cols() % kThreadBlockDimInColsInRowsBlocky.x != 0) { thread_grid_dim_incolsinrows_blocky.x += 1; }
	if (input_table.num_rows() % kThreadBlockDimInColsInRowsBlocky.x != 0) { thread_grid_dim_incolsinrows_blocky.y += 1; }

	//Logger::new_line("#blocks (in-rows filter): (" + std::to_string(thread_grid_dim_inrows.x)
	//	+ ", " + std::to_string(thread_grid_dim_inrows.y) + ")");
	//Logger::new_line("#threads in block (in-rows filter): (" + std::to_string(kThreadBlockDimInRows.x)
	//	+ ", " + std::to_string(kThreadBlockDimInRows.y) + ", " + std::to_string(kThreadBlockDimInRows.z) + ")");
	//Logger::new_line("#blocks (in-cols filter): (" + std::to_string(thread_grid_dim_incols.x)
	//	+ ", " + std::to_string(thread_grid_dim_incols.y) + ")");
	//Logger::new_line("#blocks (in-cols blocky filter): (" + std::to_string(thread_grid_dim_incols_blocky.x)
	//	+ ", " + std::to_string(thread_grid_dim_incols_blocky.y) + ")");
	//Logger::new_line("#threads in block (in-cols blocky filter): (" + std::to_string(kThreadBlockDimInColsBlocky.x)
	//	+ ", " + std::to_string(kThreadBlockDimInColsBlocky.y) + ", " + std::to_string(kThreadBlockDimInColsBlocky.z) + ")");
	Logger::new_line("#blocks (in-cols in-rows blocky filter): (" + std::to_string(thread_grid_dim_incolsinrows_blocky.x)
		+ ", " + std::to_string(thread_grid_dim_incolsinrows_blocky.y) + ")");
	Logger::new_line("#threads in block (in-cols in-rows blocky filter): (" + std::to_string(kThreadBlockDimInColsInRowsBlocky.x)
		+ ", " + std::to_string(kThreadBlockDimInColsInRowsBlocky.y) + ", " + std::to_string(kThreadBlockDimInColsInRowsBlocky.z) + ")");
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	for (int i_run = 0; i_run < kNumRuns; ++i_run) {
		apply_recursive_filter_downright_blocky_part1
			<<<thread_grid_dim_incolsinrows_blocky, kThreadBlockDimInColsInRowsBlocky>>>
				(d_input_table_data, input_table.num_rows(), input_table.num_cols(),
				filter_coeffs[0], filter_coeffs[1], d_blockwise_colwise_sums, d_blockwise_rowwise_sums);
		//apply_recursive_filter_right<<<thread_grid_dim_inrows, kThreadBlockDimInRows>>>
		//	(d_output_table_data, input_table.num_rows(), input_table.num_cols(), filter_coeffs[0], filter_coeffs[1], d_output_table_data);
		//apply_recursive_filter_down<<<thread_grid_dim_incols, kThreadBlockDim>>>
		//	(d_output_table_data, input_table.num_rows(), input_table.num_cols(), filter_coeffs[0], filter_coeffs[1], d_output_table_data);
	}
	//copy<<<n_thread_blocks, n_threads_in_block>>>
	//	(d_input_table_data, input_table.num_rows(), input_table.num_cols(), d_output_table_data);
	//cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float time_elapsed_ms = 0;
	cudaEventElapsedTime(&time_elapsed_ms, start, stop);
	Logger::new_line("Kernel execution time for "
		+ std::to_string(input_table.num_cols()) + "x" + std::to_string(input_table.num_rows())
		+ " [ms]: " + std::to_string(time_elapsed_ms / float(kNumRuns))
		+ " (average of " + std::to_string(kNumRuns) + " runs)");

	float* h_output_table_data = (float*)malloc(input_table.num_rows() * input_table.num_cols() * sizeof(float));
	cudaMemcpy(h_output_table_data, d_output_table_data, input_table.num_rows()
		* input_table.num_cols() * sizeof(float), cudaMemcpyDeviceToHost);
	output_table.set(h_output_table_data);

	float* h_blockwise_colwise_sums = (float*)malloc(n_blockwise_colwise_rows * input_table.num_cols() * sizeof(float));
	cudaMemcpy(h_blockwise_colwise_sums, d_blockwise_colwise_sums, n_blockwise_colwise_rows
		* input_table.num_cols() * sizeof(float), cudaMemcpyDeviceToHost);
	CpuTable blockwise_colwise_table(n_blockwise_colwise_rows, input_table.num_cols(), h_blockwise_colwise_sums);
	if (n_blockwise_colwise_rows <= 12 && input_table.num_cols() <= 12) {
		Logger::new_line("Blockwise-colwise table:\n" + blockwise_colwise_table.toString());
	}

	float* h_blockwise_rowwise_sums = (float*)malloc(input_table.num_rows() * n_d_blockwise_rowwise_cols * sizeof(float));
	cudaMemcpy(h_blockwise_rowwise_sums, d_blockwise_rowwise_sums, input_table.num_rows()
		* n_d_blockwise_rowwise_cols * sizeof(float), cudaMemcpyDeviceToHost);
	CpuTable blockwise_rowwise_table(input_table.num_rows(), n_d_blockwise_rowwise_cols, h_blockwise_rowwise_sums);
	if (input_table.num_rows() <= 12 && n_d_blockwise_rowwise_cols <= 12) {
		Logger::new_line("Blockwise-rowwise table:\n" + blockwise_rowwise_table.toString());
	}

	free(h_output_table_data);
	cudaFree(d_input_table_data);
	cudaFree(d_output_table_data);
}

}