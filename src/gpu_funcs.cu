#include "gpu_funcs.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef _WIN32
	#include <device_launch_parameters.h>
#endif

#include <iostream>

namespace gpuacademy {

__global__ void apply_recursive_filter(float* input_table, int num_rows,
	int num_cols, float* filter, int num_filter_coeffs, float* output_table) {
	int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
	int thread_id_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (thread_id_x < num_cols && thread_id_y < num_rows) {
		output_table[thread_id_x + thread_id_y * num_cols] = input_table[thread_id_x + thread_id_y * num_cols];
	}
}

void apply_recursive_filter_gpu(float* input_table, int num_rows, int num_cols,
	float* filter_coeffs, int num_filter_coeffs, float* output_table) {
	float* d_input_table;
	cudaMalloc((void**)(&d_input_table), num_cols * num_rows * sizeof(float));
	cudaMemcpy(d_input_table, input_table, num_cols * num_rows * sizeof(float), cudaMemcpyHostToDevice);

	float* d_filter_coeffs;
	cudaMalloc((void**)(&d_filter_coeffs), num_filter_coeffs * sizeof(float));
	cudaMemcpy(d_filter_coeffs, filter_coeffs, num_filter_coeffs * sizeof(float), cudaMemcpyHostToDevice);

	float* d_output_table;
	cudaMalloc((void**)(&d_output_table), num_cols * num_rows * sizeof(float));
	cudaMemcpy(d_output_table, output_table, num_cols * num_rows * sizeof(float), cudaMemcpyHostToDevice);

	const int div = 2;
	const dim3 n_thread_blocks(num_cols / div, num_rows / div);
	const dim3 n_threads_in_block(div, div);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	apply_recursive_filter<<<n_thread_blocks, n_threads_in_block>>>
		(d_input_table, num_rows, num_cols, d_filter_coeffs, num_filter_coeffs, d_output_table);
	//cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaMemcpy(output_table, d_output_table, num_cols * num_rows * sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	float time_elapsed_ms = 0;
	cudaEventElapsedTime(&time_elapsed_ms, start, stop);
	std::cout << std::endl << "Kernel executione time of \"apply_recursive_filter\" [ms]: " << time_elapsed_ms << std::endl;
}

}