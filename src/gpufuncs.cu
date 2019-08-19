#include "gpufuncs.hpp"
#include "Logger.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>

namespace gpuacademy {

const int kTableBlockDimX = 5;
const int kTableBlockDimY = 3;
const dim3 kThreadBlockDim = dim3(1, 1, std::max(kTableBlockDimX, kTableBlockDimY));

__global__ void copy(const float* input_table, int num_rows,
	int num_cols, float* output_table) {
	int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
	int thread_id_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (thread_id_x < num_cols && thread_id_y < num_rows) {
		output_table[thread_id_x + thread_id_y * num_cols] = input_table[thread_id_x + thread_id_y * num_cols];
	}
}

// One thread calculates one row within a table block
__global__ void apply_recursive_filter_right(const float* input_table, int num_rows,
	int num_cols, const float* filter_coeffs, int num_filter_coeffs, float* output_table) {
	const int global_tid = blockIdx.y * blockDim.z + threadIdx.z;
	
	const int tableblock_start_x_global = blockIdx.x * kTableBlockDimX;
	const int y_in_tableblock = global_tid % kTableBlockDimY;
	__shared__ float rowwise_sums_tableblock[kTableBlockDimX * kTableBlockDimY];
	//printf("\n%d, %d, %d, %d", blockIdx.x, blockDim.x, threadIdx.x, tableblock_start_x);
	if (global_tid < num_rows) {
		#pragma unroll
		for (int x_in_tableblock = 0; x_in_tableblock < kTableBlockDimX; ++x_in_tableblock) {
			if (tableblock_start_x_global + x_in_tableblock < num_cols) {
				const int id_in_tableblock = x_in_tableblock + y_in_tableblock * kTableBlockDimX;
				rowwise_sums_tableblock[id_in_tableblock] =
					filter_coeffs[0] * input_table[tableblock_start_x_global + x_in_tableblock + global_tid * num_cols];
				if (x_in_tableblock > 0) {
					rowwise_sums_tableblock[id_in_tableblock] +=
						filter_coeffs[1] * rowwise_sums_tableblock[(x_in_tableblock - 1) + y_in_tableblock * kTableBlockDimX];
				}
			}
		}
		__syncthreads();
		for (int x_in_tableblock = 0; x_in_tableblock < kTableBlockDimX; ++x_in_tableblock) {
			if (tableblock_start_x_global + x_in_tableblock < num_cols) {
				output_table[tableblock_start_x_global + x_in_tableblock + global_tid * num_cols] =
					rowwise_sums_tableblock[x_in_tableblock + y_in_tableblock * kTableBlockDimX];
			}
		}
	}
}

// One thread calculates one col within a table block
__global__ void apply_recursive_filter_down(const float* input_table, int num_rows,
	int num_cols, const float* filter_coeffs, int num_filter_coeffs, float* output_table) {
	const int global_tid = blockIdx.x * blockDim.z + threadIdx.z;

	const int tableblock_start_y_global = blockIdx.y * kTableBlockDimY;
	const int x_in_tableblock = global_tid % kTableBlockDimX;
	__shared__ float colwise_sums_tableblock[kTableBlockDimX * kTableBlockDimY];
	if (global_tid < num_cols) {
		#pragma unroll
		for (int y_in_tableblock = 0; y_in_tableblock < kTableBlockDimY; ++y_in_tableblock) {
			if (tableblock_start_y_global + y_in_tableblock < num_rows) {
				const int id_in_tableblock = x_in_tableblock + y_in_tableblock * kTableBlockDimX;
				colwise_sums_tableblock[id_in_tableblock] =
					filter_coeffs[0] * input_table[global_tid + (tableblock_start_y_global + y_in_tableblock) * num_cols];
				if (y_in_tableblock > 0) {
					colwise_sums_tableblock[id_in_tableblock] +=
						filter_coeffs[1] * colwise_sums_tableblock[x_in_tableblock + (y_in_tableblock - 1) * kTableBlockDimX];
				}
			}
		}
		__syncthreads();
		for (int y_in_tableblock = 0; y_in_tableblock < kTableBlockDimY; ++y_in_tableblock) {
			if (tableblock_start_y_global + y_in_tableblock < num_rows) {
				output_table[global_tid + (tableblock_start_y_global + y_in_tableblock) * num_cols] =
					colwise_sums_tableblock[x_in_tableblock + y_in_tableblock * kTableBlockDimX];
			}
		}
	}
}

void apply_right_down_recursive_filter_gpu(const CpuTable& input_table,
	const float* filter_coeffs, int num_filter_coeffs, CpuTable& output_table) {
	if (input_table.num_rows() != output_table.num_rows()) {
		throw std::runtime_error("Number of table rows must match");
	}
	if (input_table.num_cols() != output_table.num_cols()) {
		throw std::runtime_error("Number of table cols must match");
	}

	float* d_input_table_data;
	cudaMalloc((void**)(&d_input_table_data), input_table.num_rows() * input_table.num_cols() * sizeof(float));
	cudaMemcpy(d_input_table_data, input_table.data(), input_table.num_rows() * input_table.num_cols() * sizeof(float), cudaMemcpyHostToDevice);

	float* d_filter_coeffs;
	cudaMalloc((void**)(&d_filter_coeffs), num_filter_coeffs * sizeof(float));
	cudaMemcpy(d_filter_coeffs, filter_coeffs, num_filter_coeffs * sizeof(float), cudaMemcpyHostToDevice);

	float* d_output_table_data;
	float* output_table_data = (float*)calloc(input_table.num_rows() * input_table.num_cols(), sizeof(float));
	cudaMalloc((void**)(&d_output_table_data), input_table.num_rows() * input_table.num_cols() * sizeof(float));
	cudaMemcpy(d_output_table_data, output_table_data, input_table.num_rows() * input_table.num_cols() * sizeof(float), cudaMemcpyHostToDevice);

	dim3 thread_grid_dim(input_table.num_cols() / kTableBlockDimX,
		input_table.num_rows() / kTableBlockDimY);
	if (input_table.num_cols() % kTableBlockDimX != 0) {
		thread_grid_dim.x += 1;
	}
	if (input_table.num_rows() % kTableBlockDimY != 0) {
		thread_grid_dim.y += 1;
	}
	/*dim3 thread_grid_dim_rightfilter(input_table.num_cols() / kThreadBlockDim.x, input_table.num_rows() / kThreadBlockDim.y);
	if (input_table.num_cols() % kThreadBlockDim.x != 0) {
		thread_grid_dim_rightfilter.x += 1;
	}
	if (input_table.num_rows() % kThreadBlockDim.y != 0) {
		thread_grid_dim_rightfilter.y += 1;
	}
	dim3 thread_grid_dim_downfilter(input_table.num_cols() / kThreadBlockDim.y, input_table.num_rows() / kThreadBlockDim.x);
	if (input_table.num_cols() % kThreadBlockDim.y != 0) {
		thread_grid_dim_downfilter.x += 1;
	}
	if (input_table.num_rows() % kThreadBlockDim.x != 0) {
		thread_grid_dim_downfilter.y += 1;
	}*/
	Logger::new_line("#blocks, #threads in block: (" + std::to_string(thread_grid_dim.x)
		+ ", " + std::to_string(thread_grid_dim.y) + "), (" + std::to_string(kThreadBlockDim.x)
		+ ", " + std::to_string(kThreadBlockDim.y) + ", " + std::to_string(kThreadBlockDim.z) + ")");
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	apply_recursive_filter_right<<<thread_grid_dim, kThreadBlockDim>>>
		(d_input_table_data, input_table.num_rows(), input_table.num_cols(), d_filter_coeffs, num_filter_coeffs, d_output_table_data);
	apply_recursive_filter_down<<<thread_grid_dim, kThreadBlockDim>>>
		(d_output_table_data, input_table.num_rows(), input_table.num_cols(), d_filter_coeffs, num_filter_coeffs, d_output_table_data);
	//copy<<<n_thread_blocks, n_threads_in_block>>>
	//	(d_input_table_data, input_table.num_rows(), input_table.num_cols(), d_output_table_data);
	//cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaMemcpy(output_table_data, d_output_table_data, input_table.num_rows() * input_table.num_cols() * sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	float time_elapsed_ms = 0;
	cudaEventElapsedTime(&time_elapsed_ms, start, stop);
	Logger::new_line("Kernel executione time of \"apply_recursive_filter\" [ms]: " + std::to_string(time_elapsed_ms));

	output_table.set(output_table_data);
	free(output_table_data);
	cudaFree(d_input_table_data);
	cudaFree(d_filter_coeffs);
	cudaFree(d_output_table_data);
}

}