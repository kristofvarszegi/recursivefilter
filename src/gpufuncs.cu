#include "gpufuncs.hpp"
#include "Logger.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>

namespace gpuacademy {

const int kNumRuns = 5;
//const int kTableBlockDimX = 2, kTableBlockDimY = 2;
const int kTableBlockDimX = 64, kTableBlockDimY = 64;
const dim3 kThreadBlockDimInRows = dim3(1, kTableBlockDimY);
const dim3 kThreadBlockDimInCols = dim3(kTableBlockDimX , 1);
const dim3 kThreadBlockDimInColsBlocky = dim3(kTableBlockDimX, 1);
const dim3 kThreadBlockDimInColsInRowsBlocky = dim3(std::max(kTableBlockDimX, kTableBlockDimY), 1);

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
		//const int x_in_tableblock = global_tid_x & (kTableBlockDimX - 1);	// Modulo with power of 2

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

__global__ void apply_recursive_filter_downright_blocky(
	const float* input, int num_rows, int num_cols, float filter_coeff_0,
	float filter_coeff_1, float* blockwise_colwise_sums,
	float* blockwise_rowwise_sums, float* aggregated_colwise_sums,
	float* blockwise_rowwise_aggregatedcolsums, float* aggregated_rowwise_sums,
	float* output) {
	const int global_tid_x = blockIdx.x * kTableBlockDimX + threadIdx.x;
	const int global_tid_y = blockIdx.y * kTableBlockDimY + threadIdx.x;
		// Yes, blockDim.x and threadIdx.x (not .y), as we have a 1D thread array within a thread block

	__shared__ float colwise_sums_thisblock[kTableBlockDimX * kTableBlockDimY];
	__shared__ float rowwise_sums_thisblock[kTableBlockDimY];
	__shared__ float aggregated_colwise_sums_thisblock[kTableBlockDimX * kTableBlockDimY];

	const int thisblock_start_x_global = blockIdx.x * kTableBlockDimX;
	const int thisblock_start_y_global = blockIdx.y * kTableBlockDimY;
	if (global_tid_x < num_cols) {
		const int x_in_thisblock = threadIdx.x;

#pragma unroll
		for (int y_in_thisblock = 0; y_in_thisblock < kTableBlockDimY; ++y_in_thisblock) {
			if (thisblock_start_y_global + y_in_thisblock <= num_rows) {
				const int id_in_tableblock = x_in_thisblock + y_in_thisblock * kTableBlockDimX;
				colwise_sums_thisblock[id_in_tableblock] = 0.0f;
				if (thisblock_start_y_global + y_in_thisblock < num_rows) {
					colwise_sums_thisblock[id_in_tableblock] = filter_coeff_0
						* input[global_tid_x + (thisblock_start_y_global + y_in_thisblock) * num_cols];
				}
				if (y_in_thisblock > 0) {
					colwise_sums_thisblock[id_in_tableblock] += filter_coeff_1 *
						colwise_sums_thisblock[x_in_thisblock + (y_in_thisblock - 1) * kTableBlockDimX];
				}
			}
		}
		__syncthreads();
		blockwise_colwise_sums[global_tid_x + blockIdx.y * num_cols] =
			colwise_sums_thisblock[threadIdx.x + (kTableBlockDimY - 1) * kTableBlockDimX];
	}

	if (global_tid_y < num_rows && threadIdx.x < kTableBlockDimY) {
		const int y_in_thisblock = threadIdx.x;	// Each thread does a row
		float aggregated_sum, prev_aggregated_sum = 0.0f;
		for (int x_in_thisblock = 0; x_in_thisblock < kTableBlockDimX; ++x_in_thisblock) {
			if (thisblock_start_x_global + x_in_thisblock < num_cols) {
				aggregated_sum = filter_coeff_0 * colwise_sums_thisblock[x_in_thisblock + y_in_thisblock * kTableBlockDimX]
					+ filter_coeff_1 * prev_aggregated_sum;
				rowwise_sums_thisblock[y_in_thisblock] = aggregated_sum;
			}
			prev_aggregated_sum = aggregated_sum;
		}
		__syncthreads();
		blockwise_rowwise_sums[blockIdx.x + (thisblock_start_y_global + threadIdx.x) * gridDim.x] =
			rowwise_sums_thisblock[threadIdx.x];
	}

	if (global_tid_x < num_cols) {
		float aggregated_sum, prev_aggregated_sum = 0.0f;
		for (int y_in_grid = 0; y_in_grid < gridDim.y; ++y_in_grid) {
			aggregated_sum = filter_coeff_0 * blockwise_colwise_sums[global_tid_x + y_in_grid * num_cols]
				+ filter_coeff_1 * prev_aggregated_sum;
			prev_aggregated_sum = aggregated_sum;
			aggregated_colwise_sums[global_tid_x + y_in_grid * num_cols] = aggregated_sum;
		}
	}

	if (global_tid_y < num_rows && threadIdx.x == 0) {
		float aggregated_sum, prev_aggregated_sum = 0.0f;
		for (int x_in_blockrow = 0; x_in_blockrow < kTableBlockDimX ; ++x_in_blockrow) {
			if (blockIdx.x * kTableBlockDimX + x_in_blockrow < num_cols) {
				aggregated_sum = filter_coeff_0
					* aggregated_colwise_sums[(blockIdx.x * kTableBlockDimX + x_in_blockrow) + blockIdx.y * num_cols]
					+ filter_coeff_1 * prev_aggregated_sum;
				prev_aggregated_sum = aggregated_sum;
			}
		}
		blockwise_rowwise_aggregatedcolsums[blockIdx.x + blockIdx.y * gridDim.x] = aggregated_sum;
	}

	if (global_tid_y < num_rows) {
		float aggregated_sum, prev_aggregated_sum = 0.0f;
		for (int x_in_grid = 0; x_in_grid < gridDim.x; ++x_in_grid) {
			aggregated_sum = filter_coeff_0 * blockwise_rowwise_sums[x_in_grid + global_tid_y * gridDim.x]
				+ filter_coeff_1 * prev_aggregated_sum;
			if (blockIdx.y > 0) {
				aggregated_sum += filter_coeff_1 * blockwise_rowwise_aggregatedcolsums[x_in_grid + (blockIdx.y - 1) * gridDim.x];
			}
			prev_aggregated_sum = aggregated_sum;
			aggregated_rowwise_sums[x_in_grid + global_tid_y * gridDim.x] = aggregated_sum;
		}
	}

	if (global_tid_x < num_cols) {
		float aggregated_sum, prev_aggregated_sum = 0.0f;
		if (blockIdx.y > 0) {
			prev_aggregated_sum = aggregated_colwise_sums[global_tid_x + (blockIdx.y - 1) * num_cols];
		}
		const int y_in_thisblock_upper = (thisblock_start_y_global + kTableBlockDimY >= num_rows)
			? (num_rows - thisblock_start_y_global) : kTableBlockDimY;
		for (int y_in_thisblock = 0; y_in_thisblock < y_in_thisblock_upper; ++y_in_thisblock) {
			aggregated_sum = filter_coeff_0 * input[global_tid_x + (thisblock_start_y_global + y_in_thisblock) * num_cols]
				+ filter_coeff_1 * prev_aggregated_sum;
			prev_aggregated_sum = aggregated_sum;
			aggregated_colwise_sums_thisblock[threadIdx.x + y_in_thisblock * kTableBlockDimX] =
				aggregated_sum;
			//output[global_tid_x + (thisblock_start_y_global + y_in_thisblock) * num_cols] =
			//	aggregated_colwise_sums_thisblock[threadIdx.x + y_in_thisblock * kTableBlockDimX];
		}
		__syncthreads();
	}

	if (global_tid_y < num_rows) {
		float aggregated_sum, prev_aggregated_sum = 0.0f;
		if (blockIdx.x > 0) {
			prev_aggregated_sum = aggregated_rowwise_sums[(blockIdx.x - 1) + global_tid_y * gridDim.x];
		}
		const int x_in_thisblock_upper = (thisblock_start_x_global + kTableBlockDimX >= num_cols)
			? (num_cols - thisblock_start_x_global) : kTableBlockDimX;
		for (int x_in_thisblock = 0; x_in_thisblock < x_in_thisblock_upper; ++x_in_thisblock) {
			aggregated_sum = filter_coeff_0 * aggregated_colwise_sums_thisblock[x_in_thisblock + threadIdx.x * kTableBlockDimX]
				+ filter_coeff_1 * prev_aggregated_sum;	// Yes, threadIdx.x (not .y)
			prev_aggregated_sum = aggregated_sum;
			output[(thisblock_start_x_global + x_in_thisblock) + global_tid_y * num_cols] = aggregated_sum;
		}
	}
}

void apply_right_down_recursive_filter_gpu(const CpuTable& input,
	float filter_coeff_0, float filter_coeff_1, CpuTable& output) {
	if (input.num_rows() != output.num_rows()) {
		throw std::runtime_error("Number of table rows must match");
	}
	if (input.num_cols() != output.num_cols()) {
		throw std::runtime_error("Number of table cols must match");
	}

	Logger::new_line("Input table dims: (" + std::to_string(input.num_cols())
		+ ", " + std::to_string(input.num_rows()) + ")");

	float* d_input;
	cudaMalloc((void**)(&d_input), input.num_rows() * input.num_cols() * sizeof(float));
	cudaMemcpy(d_input, input.data(), input.num_cols() * input.num_rows() * sizeof(float), cudaMemcpyHostToDevice);

	float* d_blockwise_colwise_sums;
	const int n_blockwise_colwise_rows = input.num_rows() % kTableBlockDimY == 0 ?
		input.num_rows() / kTableBlockDimY : input.num_rows() / kTableBlockDimY + 1;
	cudaMalloc((void**)(&d_blockwise_colwise_sums), input.num_cols() * n_blockwise_colwise_rows * sizeof(float));
	Logger::new_line("Blockwise-colwise table dims: (" + std::to_string(input.num_cols())
		+ ", " + std::to_string(n_blockwise_colwise_rows) + ")");

	float* d_blockwise_rowwise_sums;
	const int n_blockwise_rowwise_cols = input.num_cols() % kTableBlockDimX == 0 ?
		input.num_cols() / kTableBlockDimX : input.num_cols() / kTableBlockDimX + 1;
	cudaMalloc((void**)(&d_blockwise_rowwise_sums), n_blockwise_rowwise_cols * input.num_rows() * sizeof(float));
	Logger::new_line("Blockwise-rowwise table dims: (" + std::to_string(n_blockwise_rowwise_cols)
		+ ", " + std::to_string(input.num_rows()) + ")");

	float* d_aggregated_colwise_sums;
	const int n_aggregated_colwise_rows = input.num_rows() % kTableBlockDimY == 0 ?
		input.num_rows() / kTableBlockDimY : input.num_rows() / kTableBlockDimY + 1;
	cudaMalloc((void**)(&d_aggregated_colwise_sums), input.num_cols() * n_aggregated_colwise_rows * sizeof(float));
	Logger::new_line("Aggregated colwise table dims: (" + std::to_string(input.num_cols())
		+ ", " + std::to_string(n_aggregated_colwise_rows) + ")");

	float* d_blockwise_rowwise_aggregatedcolsums;
	const int n_blockwise_rowwise_aggregatedcolsums_rows = input.num_rows() % kTableBlockDimY == 0 ?
		input.num_rows() / kTableBlockDimY : input.num_rows() / kTableBlockDimY + 1;
	const int n_blockwise_rowwise_aggregatedcolsums_cols = input.num_cols() % kTableBlockDimX == 0 ?
		input.num_cols() / kTableBlockDimX : input.num_cols() / kTableBlockDimX + 1;
	cudaMalloc((void**)(&d_blockwise_rowwise_aggregatedcolsums),
		n_blockwise_rowwise_aggregatedcolsums_cols * n_blockwise_rowwise_aggregatedcolsums_rows * sizeof(float));
	Logger::new_line("Blockwise-rowwise aggregatedcolsum table dims: ("
		+ std::to_string(n_blockwise_rowwise_aggregatedcolsums_cols) + ", "
		+ std::to_string(n_blockwise_rowwise_aggregatedcolsums_rows) + ")");
	
	float* d_aggregated_rowwise_sums;
	const int n_aggregated_rowwise_cols = input.num_cols() % kTableBlockDimX == 0 ?
		input.num_cols() / kTableBlockDimX : input.num_cols() / kTableBlockDimX + 1;
	cudaMalloc((void**)(&d_aggregated_rowwise_sums), n_aggregated_rowwise_cols * input.num_rows() * sizeof(float));
	Logger::new_line("Aggregated rowwise sum table dims: (" + std::to_string(n_aggregated_rowwise_cols)
		+ ", " + std::to_string(input.num_rows()) + ")");

	float* d_output;
	cudaMalloc((void**)(&d_output), input.num_rows() * input.num_cols() * sizeof(float));
	
	//dim3 thread_grid_dim_inrows(1, input_table.num_rows() / kThreadBlockDimInRows.y);
	//if (input_table.num_rows() % kThreadBlockDimInRows.y != 0) { thread_grid_dim_inrows.y += 1; }
	//dim3 thread_grid_dim_incols(input_table.num_cols() / kThreadBlockDimInCols.x, 1);
	//if (input_table.num_cols() % kThreadBlockDimInCols.x != 0) { thread_grid_dim_incols.x += 1; }
	//dim3 thread_grid_dim_incols_blocky(input_table.num_cols() / kTableBlockDimX,
	//	input_table.num_rows() / kTableBlockDimY);
	//if (input_table.num_cols() % kTableBlockDimX != 0) { thread_grid_dim_incols_blocky.x += 1; }
	//if (input_table.num_rows() % kTableBlockDimY != 0) { thread_grid_dim_incols_blocky.y += 1; }

	dim3 thread_grid_dim_incolsinrows_blocky(input.num_cols() / kThreadBlockDimInColsInRowsBlocky.x,
		input.num_rows() / kThreadBlockDimInColsInRowsBlocky.x);	// Same array of threads do the cols and the rows as well
	if (input.num_cols() % kThreadBlockDimInColsInRowsBlocky.x != 0) { thread_grid_dim_incolsinrows_blocky.x += 1; }
	if (input.num_rows() % kThreadBlockDimInColsInRowsBlocky.x != 0) { thread_grid_dim_incolsinrows_blocky.y += 1; }

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
		apply_recursive_filter_downright_blocky
			<<<thread_grid_dim_incolsinrows_blocky, kThreadBlockDimInColsInRowsBlocky>>>
				(d_input, input.num_rows(), input.num_cols(),
				filter_coeff_0, filter_coeff_1, d_blockwise_colwise_sums,
				d_blockwise_rowwise_sums, d_aggregated_colwise_sums,
				d_blockwise_rowwise_aggregatedcolsums, d_aggregated_rowwise_sums,
				d_output);
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
	Logger::new_line("\nKernel execution time for "
		+ std::to_string(input.num_cols()) + "x" + std::to_string(input.num_rows())
		+ " [ms]: " + std::to_string(time_elapsed_ms / float(kNumRuns))
		+ " (average of " + std::to_string(kNumRuns) + " runs)");

	float* h_blockwise_colwise_sums = (float*)malloc(n_blockwise_colwise_rows * input.num_cols() * sizeof(float));
	cudaMemcpy(h_blockwise_colwise_sums, d_blockwise_colwise_sums, n_blockwise_colwise_rows
		* input.num_cols() * sizeof(float), cudaMemcpyDeviceToHost);
	CpuTable blockwise_colwise_sums(n_blockwise_colwise_rows, input.num_cols(), h_blockwise_colwise_sums);
	if (n_blockwise_colwise_rows <= 12 && input.num_cols() <= 12) {
		Logger::new_line("\nBlockwise-colwise table (light blue):\n" + blockwise_colwise_sums.toString());
	}

	float* h_blockwise_rowwise_sums = (float*)malloc(input.num_rows() * n_blockwise_rowwise_cols * sizeof(float));
	cudaMemcpy(h_blockwise_rowwise_sums, d_blockwise_rowwise_sums, input.num_rows()
		* n_blockwise_rowwise_cols * sizeof(float), cudaMemcpyDeviceToHost);
	CpuTable blockwise_rowwise_sums(input.num_rows(), n_blockwise_rowwise_cols, h_blockwise_rowwise_sums);
	if (input.num_rows() <= 12 && n_blockwise_rowwise_cols <= 12) {
		Logger::new_line("\nBlockwise-rowwise table: (light green)\n" + blockwise_rowwise_sums.toString());
	}

	float* h_aggregated_colwise_sums = (float*)malloc(
		n_aggregated_colwise_rows * input.num_cols() * sizeof(float));
	cudaMemcpy(h_aggregated_colwise_sums, d_aggregated_colwise_sums, n_aggregated_colwise_rows
		* input.num_cols() * sizeof(float), cudaMemcpyDeviceToHost);
	CpuTable aggregated_blockwise_colwise_sums(n_aggregated_colwise_rows, input.num_cols(), h_aggregated_colwise_sums);
	if (n_aggregated_colwise_rows <= 12 && input.num_cols() <= 12) {
		Logger::new_line("\nAggregated blockwise-colwise table (dark blue):\n" + aggregated_blockwise_colwise_sums.toString());
	}

	float* h_blockwise_rowwise_aggregatedcolsums = (float*)malloc(
		n_blockwise_rowwise_aggregatedcolsums_rows * n_blockwise_rowwise_aggregatedcolsums_cols * sizeof(float));
	cudaMemcpy(h_blockwise_rowwise_aggregatedcolsums, d_blockwise_rowwise_aggregatedcolsums, n_blockwise_rowwise_aggregatedcolsums_rows
		* n_blockwise_rowwise_aggregatedcolsums_cols * sizeof(float), cudaMemcpyDeviceToHost);
	CpuTable blockwise_rowwise_aggregatedcolsums(
		n_blockwise_rowwise_aggregatedcolsums_rows, n_blockwise_rowwise_aggregatedcolsums_cols,
		h_blockwise_rowwise_aggregatedcolsums);
	if (n_blockwise_rowwise_aggregatedcolsums_rows <= 12 && n_blockwise_rowwise_aggregatedcolsums_cols <= 12) {
		Logger::new_line("\nBlockwise-rowwise aggregatedcolsum table (red):\n" + blockwise_rowwise_aggregatedcolsums.toString());
	}

	float* h_aggregated_rowwise_sums = (float*)malloc(input.num_rows() * n_aggregated_rowwise_cols * sizeof(float));
	cudaMemcpy(h_aggregated_rowwise_sums, d_aggregated_rowwise_sums, input.num_rows()
		* n_aggregated_rowwise_cols * sizeof(float), cudaMemcpyDeviceToHost);
	CpuTable aggregated_rowwise_sums(input.num_rows(), n_aggregated_rowwise_cols, h_aggregated_rowwise_sums);
	if (input.num_rows() <= 12 && n_aggregated_rowwise_cols <= 12) {
		Logger::new_line("\nAggregated rowwise table (dark green):\n" + aggregated_rowwise_sums.toString());
	}

	float* h_output = (float*)malloc(input.num_rows() * input.num_cols() * sizeof(float));
	cudaMemcpy(h_output, d_output, input.num_rows()
		* input.num_cols() * sizeof(float), cudaMemcpyDeviceToHost);
	output.set(h_output);
	if (output.num_rows() <= 12 && output.num_cols() <= 12) {
		Logger::new_line("\nOutput table:\n" + output.toString());
	}

	cudaFree(d_input);
	cudaFree(d_blockwise_colwise_sums);
	free(h_blockwise_colwise_sums);
	cudaFree(d_blockwise_rowwise_sums);
	free(h_blockwise_rowwise_sums);
	cudaFree(d_aggregated_colwise_sums);
	free(h_aggregated_colwise_sums);
	cudaFree(d_blockwise_rowwise_aggregatedcolsums);
	free(h_blockwise_rowwise_aggregatedcolsums);
	cudaFree(d_aggregated_rowwise_sums);
	free(h_aggregated_rowwise_sums);
	cudaFree(d_output);
	free(h_output);
}

}