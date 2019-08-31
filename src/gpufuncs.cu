#include "Logger.hpp"
#include "gpufuncs.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <stdexcept>

#define SHMEM_PAD_X 1
#define PRINT_LIMIT_X 8
#define PRINT_LIMIT_Y 8

namespace gpuacademy {

inline void chk_cu_err(CUresult code) {
  if (code != CUDA_SUCCESS) {
    const char *buf;
    cuGetErrorString(code, &buf);
    throw std::runtime_error("Error: " + std::string(buf) + "\n");
  }
}

inline void chk_cu_err(cudaError_t code) {
  if (code != cudaSuccess) {
    throw std::runtime_error("Error: " + std::string(cudaGetErrorString(code)) +
                             "\n");
  }
}

#if 0
template <int EXPONENT>
__device__ float float_to_the_power_int(float base) {
	float accum = 1.0f;
#pragma unroll
	for (int i = 0; i < EXPONENT; ++i) {
		accum *= base;
	}
	return accum;
}
template __device__ float float_to_the_power_int<2>(float base);
template __device__ float float_to_the_power_int<32>(float base);
#endif

__global__ void copy(const float *input_table, int num_rows, int num_cols,
                     float *output_table) {
  int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_id_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (thread_id_x < num_cols && thread_id_y < num_rows) {
    output_table[thread_id_x + thread_id_y * num_cols] =
        input_table[thread_id_x + thread_id_y * num_cols];
  }
}

// One thread calculates one col
__global__ void recursivefilter_down(const float *input_table, int num_rows,
                                     int num_cols, float filter_coeff_0,
                                     float filter_coeff_1,
                                     float *output_table) {
  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_tid < num_cols) {
    float filtered_val, prev_filtered_val;
#pragma unroll
    for (int i_row = 0; i_row < num_rows; ++i_row) {
      filtered_val =
          filter_coeff_0 * input_table[global_tid + i_row * num_cols];
      if (i_row > 0) {
        filtered_val += filter_coeff_1 * prev_filtered_val;
      }
      output_table[global_tid + i_row * num_cols] = filtered_val;
      prev_filtered_val = filtered_val;
    }
  }
}

// One thread calculates one row
__global__ void recursivefilter_right(const float *input_table, int num_rows,
                                      int num_cols, float filter_coeff_0,
                                      float filter_coeff_1,
                                      float *output_table) {
  const int global_tid = blockIdx.y * blockDim.y + threadIdx.y;

  if (global_tid < num_rows) {
    float filtered_val, prev_filtered_val;
#pragma unroll
    for (int i_col = 0; i_col < num_cols; ++i_col) {
      filtered_val =
          filter_coeff_0 * input_table[i_col + global_tid * num_cols];
      if (i_col > 0) {
        filtered_val += filter_coeff_1 * prev_filtered_val;
      }
      output_table[i_col + global_tid * num_cols] = filtered_val;
      prev_filtered_val = filtered_val;
    }
  }
}

__global__ void recursivefilter_step1_inblocksdownright(
    const float *input, int num_rows, int num_cols, int tableblockdim_x,
    int tableblockdim_y, float filter_coeff_0, float filter_coeff_1,
    float *blockwise_colwise_sums, float *blockwise_rowwise_sums) {
  const int global_tid_x = blockIdx.x * tableblockdim_x + threadIdx.x;
  const int global_tid_y = blockIdx.y * tableblockdim_y + threadIdx.x;
  // Yes, threadIdx.x (not .y), as we have a 1D thread array within a thread
  // block

  extern __shared__ float
      colwise_sums_thisblock[]; // [tableblockdim_x * tableblockdim_y];

  if (global_tid_x < num_cols) {
    float aggregated_sum, prev_aggregated_sum = 0.0f;
#pragma unroll
    for (int y_in_thisblock = 0; y_in_thisblock < tableblockdim_y;
         ++y_in_thisblock) {
      if (blockIdx.y * tableblockdim_y + y_in_thisblock < num_rows) {
        aggregated_sum =
            filter_coeff_0 *
            input[global_tid_x +
                  (blockIdx.y * tableblockdim_y + y_in_thisblock) * num_cols];
        aggregated_sum += filter_coeff_1 * prev_aggregated_sum;
      }
      colwise_sums_thisblock[threadIdx.x +
                             y_in_thisblock * (tableblockdim_x + SHMEM_PAD_X)] =
          aggregated_sum;
      prev_aggregated_sum = aggregated_sum;
    }
    __syncthreads();
    blockwise_colwise_sums[global_tid_x + blockIdx.y * num_cols] =
        aggregated_sum;
  }

  if (global_tid_y < num_rows && threadIdx.x < tableblockdim_y) {
    float aggregated_sum, prev_aggregated_sum = 0.0f;
#pragma unroll
    for (int x_in_thisblock = 0; x_in_thisblock < tableblockdim_x;
         ++x_in_thisblock) {
      if (blockIdx.x * tableblockdim_x + x_in_thisblock < num_cols) {
        aggregated_sum = filter_coeff_0 *
                         colwise_sums_thisblock[x_in_thisblock +
                                                threadIdx.x * (tableblockdim_x +
                                                               SHMEM_PAD_X)];
        aggregated_sum += filter_coeff_1 * prev_aggregated_sum;
      }
      prev_aggregated_sum = aggregated_sum;
    }
    blockwise_rowwise_sums[(blockIdx.y * tableblockdim_y + threadIdx.x) +
                           blockIdx.x * num_rows] = aggregated_sum;
    // Transposed to coalesce global memory access
    // blockwise_rowwise_sums[blockIdx.x + (blockIdx.y * tableblockdim_y +
    // threadIdx.x) * gridDim.x] = aggregated_sum;
  }
}

__global__ void recursivefilter_step2_overblocksdown(
    int num_aggregated_rows, int num_cols, int tableblockdim_y,
    float filter_coeff_0, float filter_coeff_1, float *blockwise_colwise_sums,
    float *aggregated_colwise_sums) {
  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_tid < num_cols) {
    float aggregated_sum, prev_aggregated_sum = 0.0f;
    for (int y_in_grid = 0; y_in_grid < num_aggregated_rows; ++y_in_grid) {
      aggregated_sum =
          blockwise_colwise_sums[global_tid + y_in_grid * num_cols] +
          powf(filter_coeff_1, tableblockdim_y) * prev_aggregated_sum;
      prev_aggregated_sum = aggregated_sum;
      aggregated_colwise_sums[global_tid + y_in_grid * num_cols] =
          aggregated_sum;
    }
  }
}

__global__ void recursivefilter_step3_inoverblockscolsummedblocksright(
    int num_aggregated_rows, int num_cols, int num_aggregated_cols,
    int tableblockdim_x, float filter_coeff_0, float filter_coeff_1,
    float *aggregated_colwise_sums,
    float *blockwise_rowwise_aggregatedcolsums) {
  const int global_tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (global_tid_x < num_aggregated_cols &&
      global_tid_y < num_aggregated_rows) {
    float aggregated_sum, prev_aggregated_sum = 0.0f;
#pragma unroll
    for (int x_in_blockrow = 0; x_in_blockrow < tableblockdim_x;
         ++x_in_blockrow) {
      const int global_x_offset =
          global_tid_x * tableblockdim_x + x_in_blockrow;
      if (global_x_offset < num_cols) {
        aggregated_sum =
            filter_coeff_0 *
            aggregated_colwise_sums[global_x_offset + global_tid_y * num_cols];
        aggregated_sum += filter_coeff_1 * prev_aggregated_sum;
      }
      prev_aggregated_sum = aggregated_sum;
    }
    blockwise_rowwise_aggregatedcolsums[global_tid_y +
                                        global_tid_x * num_aggregated_rows] =
        aggregated_sum;
    // Transposed to coalesce global memory access
    // blockwise_rowwise_aggregatedcolsums[global_tid_x + global_tid_y *
    // num_aggregated_cols] = aggregated_sum;
  }
}

__global__ void recursivefilter_step4_overblocksright(
    int num_rows, int num_aggregated_cols, int num_aggregated_rows,
    int tableblockdim_x, float filter_coeff_0, float filter_coeff_1,
    float *blockwise_rowwise_sums, float *blockwise_rowwise_aggregatedcolsums,
    float *aggregated_rowwise_sums) {
  const int global_tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (global_tid_y < num_rows) {
    float aggregated_sum, prev_aggregated_sum = 0.0f;
    for (int x_in_row = 0; x_in_row < num_aggregated_cols; ++x_in_row) {
      aggregated_sum =
          blockwise_rowwise_sums[global_tid_y + x_in_row * num_rows] +
          powf(filter_coeff_1, tableblockdim_x) * prev_aggregated_sum;
      // Transposed to coalesce global memory access
      // aggregated_sum = blockwise_rowwise_sums[x_in_row + global_tid_y *
      // num_aggregated_cols] + powf(filter_coeff_1, tableblockdim_x) *
      // prev_aggregated_sum;
      if (blockIdx.y > 0) {
        aggregated_sum +=
            powf(filter_coeff_1, threadIdx.y + 1) *
            blockwise_rowwise_aggregatedcolsums[(blockIdx.y - 1) +
                                                x_in_row * num_aggregated_rows];
        // Transposed to coalesce global memory access
        // aggregated_sum += powf(filter_coeff_1, threadIdx.y + 1) *
        // blockwise_rowwise_aggregatedcolsums[x_in_row + (blockIdx.y - 1) *
        // num_aggregated_cols];
      }
      prev_aggregated_sum = aggregated_sum;
      aggregated_rowwise_sums[global_tid_y + x_in_row * num_rows] =
          aggregated_sum;
      // Transposed to coalesce global memory access
      // aggregated_rowwise_sums[x_in_row + global_tid_y * num_aggregated_cols]
      // = aggregated_sum;
    }
  }
}

__global__ void recursivefilter_step5_inblocksdownright(
    const float *input, int num_rows, int num_cols, int tableblockdim_x,
    int tableblockdim_y, float filter_coeff_0, float filter_coeff_1,
    float *aggregated_colwise_sums, float *aggregated_rowwise_sums,
    float *final_sums) {
  const int global_tid_x = blockIdx.x * tableblockdim_x + threadIdx.x;
  const int global_tid_y = blockIdx.y * tableblockdim_y + threadIdx.x;
  // Yes, blockDim.x and threadIdx.x (not .y), as we have a 1D thread array
  // within a thread block

  extern __shared__ float
      aggregated_colwise_sums_thisblock[]; // [tableblockdim_x *
                                           // tableblockdim_y];

  const int thisblock_start_x_global = blockIdx.x * tableblockdim_x;
  const int thisblock_start_y_global = blockIdx.y * tableblockdim_y;

  if (global_tid_x < num_cols) {
    float aggregated_sum, prev_aggregated_sum = 0.0f;
    if (blockIdx.y > 0) {
      prev_aggregated_sum =
          aggregated_colwise_sums[global_tid_x + (blockIdx.y - 1) * num_cols];
    }
    const int y_in_thisblock_upper =
        (thisblock_start_y_global + tableblockdim_y >= num_rows)
            ? (num_rows - thisblock_start_y_global)
            : tableblockdim_y;
    for (int y_in_thisblock = 0; y_in_thisblock < y_in_thisblock_upper;
         ++y_in_thisblock) {
      aggregated_sum =
          filter_coeff_0 *
              input[global_tid_x +
                    (thisblock_start_y_global + y_in_thisblock) * num_cols] +
          filter_coeff_1 * prev_aggregated_sum;
      prev_aggregated_sum = aggregated_sum;
      aggregated_colwise_sums_thisblock[threadIdx.x +
                                        y_in_thisblock *
                                            (tableblockdim_x + SHMEM_PAD_X)] =
          aggregated_sum;
      // output[global_tid_x + (thisblock_start_y_global + y_in_thisblock) *
      // num_cols] = 	aggregated_colwise_sums_thisblock[threadIdx.x +
      // y_in_thisblock * kTableBlockDimX];
    }
    __syncthreads();
  }

  if (global_tid_y < num_rows) {
    float aggregated_sum, prev_aggregated_sum = 0.0f;
    if (blockIdx.x > 0) {
      prev_aggregated_sum =
          aggregated_rowwise_sums[global_tid_y + (blockIdx.x - 1) * num_rows];
      // Transposed to coalesce global memory access
      // prev_aggregated_sum = aggregated_rowwise_sums[(blockIdx.x - 1) +
      // global_tid_y * gridDim.x];
    }
    const int x_in_thisblock_upper =
        (thisblock_start_x_global + tableblockdim_x >= num_cols)
            ? (num_cols - thisblock_start_x_global)
            : tableblockdim_x;
    for (int x_in_thisblock = 0; x_in_thisblock < x_in_thisblock_upper;
         ++x_in_thisblock) {
      aggregated_sum =
          filter_coeff_0 *
              aggregated_colwise_sums_thisblock[x_in_thisblock +
                                                threadIdx.x * (tableblockdim_x +
                                                               SHMEM_PAD_X)] +
          filter_coeff_1 * prev_aggregated_sum; // Yes, threadIdx.x (not .y)
      prev_aggregated_sum = aggregated_sum;
      final_sums[(thisblock_start_x_global + x_in_thisblock) +
                 global_tid_y * num_cols] = aggregated_sum;
    }
  }
}

float recursivefilter_downright_gpu(const CpuTable &input, float filter_coeff_0,
                                    float filter_coeff_1, int tableblockdim_x,
                                    int tableblockdim_y, int num_kernel_runs,
                                    OUTPUT_STEP output_step,
                                    std::vector<CpuTable> &outputs) {
  if (input.num_rows() < 2) {
    throw std::runtime_error("Number of input rows must be at least 2");
  }
  if (input.num_cols() < 2) {
    throw std::runtime_error("Number of input cols must be at least 2");
  }
  if (tableblockdim_x < 2) {
    throw std::runtime_error("Table block dim x must be at least 2");
  }
  if (tableblockdim_y < 2) {
    throw std::runtime_error("Table block dim y must be at least 2");
  }
  if (num_kernel_runs < 1) {
    throw std::runtime_error("Number of kernel runs must be at least 1");
  }

  Logger::new_line("Input table dims: (" + std::to_string(input.num_cols()) +
                   ", " + std::to_string(input.num_rows()) + ")\n");

  chk_cu_err(cuInit(0));
  int device_count = -1;
  chk_cu_err(cuDeviceGetCount(&device_count));
  Logger::new_line("CUDA device count: " + std::to_string(device_count));
  CUdevice device;
  chk_cu_err(cuDeviceGet(&device, 0));
  CUcontext cudaContext;
  chk_cu_err(cuCtxCreate(&cudaContext, CU_CTX_SCHED_AUTO, device));

  if (input.num_rows() <= PRINT_LIMIT_Y && input.num_cols() <= PRINT_LIMIT_X) {
    Logger::new_line("\nInput:\n" + input.toString());
  }
  float *h_input =
      (float *)malloc(input.num_rows() * input.num_cols() * sizeof(float));
  for (int i_row = 0; i_row < input.num_rows(); ++i_row) {
    for (int i_col = 0; i_col < input.num_cols(); ++i_col) {
      h_input[i_col + i_row * input.num_cols()] = input.get(i_row, i_col);
    }
  }

  float *d_input;
  // chk_cu_err(cudaMalloc(&d_input, input.num_rows() * input.num_cols() *
  // sizeof(float)));
  chk_cu_err(cudaMalloc((void **)(&d_input),
                        input.num_rows() * input.num_cols() * sizeof(float)));
  chk_cu_err(cudaMemcpy(d_input, h_input,
                        input.num_rows() * input.num_cols() * sizeof(float),
                        cudaMemcpyHostToDevice));

  float *d_blockwise_colwise_sums;
  const int n_blockwise_colwise_rows =
      input.num_rows() % tableblockdim_y == 0
          ? input.num_rows() / tableblockdim_y
          : input.num_rows() / tableblockdim_y + 1;
  chk_cu_err(
      cudaMalloc((void **)(&d_blockwise_colwise_sums),
                 input.num_cols() * n_blockwise_colwise_rows * sizeof(float)));
  Logger::new_line("Blockwise-colwise table dims: (" +
                   std::to_string(input.num_cols()) + ", " +
                   std::to_string(n_blockwise_colwise_rows) + ")");

  float
      *d_blockwise_rowwise_sums; // Transposed to coalesce global memory access
  const int n_step1_inblocksdownright_rows =
      input.num_rows() % tableblockdim_y == 0
          ? input.num_rows() / tableblockdim_y
          : input.num_rows() / tableblockdim_y + 1;
  const int n_step1_inblocksdownright_cols =
      input.num_cols() % tableblockdim_x == 0
          ? input.num_cols() / tableblockdim_x
          : input.num_cols() / tableblockdim_x + 1;
  chk_cu_err(cudaMalloc((void **)(&d_blockwise_rowwise_sums),
                        input.num_rows() * n_step1_inblocksdownright_cols *
                            sizeof(float)));
  Logger::new_line("Blockwise-rowwise table dims (transposed!): (" +
                   std::to_string(input.num_rows()) + ", " +
                   std::to_string(n_step1_inblocksdownright_cols) + ")");

  float *d_aggregated_colwise_sums;
  const int n_step2_overblocksdown_rows =
      input.num_rows() % tableblockdim_y == 0
          ? input.num_rows() / tableblockdim_y
          : input.num_rows() / tableblockdim_y + 1;
  chk_cu_err(cudaMalloc((void **)(&d_aggregated_colwise_sums),
                        input.num_cols() * n_step2_overblocksdown_rows *
                            sizeof(float)));
  Logger::new_line("Aggregated colwise table dims: (" +
                   std::to_string(input.num_cols()) + ", " +
                   std::to_string(n_step2_overblocksdown_rows) + ")");

  float *d_blockwise_rowwise_aggregatedcolsums;
  const int n_step3_inoverblockscolsummedblocksright_rows =
      input.num_rows() % tableblockdim_y == 0
          ? input.num_rows() / tableblockdim_y
          : input.num_rows() / tableblockdim_y + 1;
  const int n_step3_inoverblockscolsummedblocksright_cols =
      input.num_cols() % tableblockdim_x == 0
          ? input.num_cols() / tableblockdim_x
          : input.num_cols() / tableblockdim_x + 1;
  chk_cu_err(cudaMalloc((void **)(&d_blockwise_rowwise_aggregatedcolsums),
                        n_step3_inoverblockscolsummedblocksright_cols *
                            n_step3_inoverblockscolsummedblocksright_rows *
                            sizeof(float)));
  Logger::new_line(
      "Blockwise-rowwise aggregatedcolsum table dims: (" +
      std::to_string(n_step3_inoverblockscolsummedblocksright_cols) + ", " +
      std::to_string(n_step3_inoverblockscolsummedblocksright_rows) + ")");

  float *d_aggregated_rowwise_sums;
  const int n_recursivefilter_step4_overblocksright_cols =
      input.num_cols() % tableblockdim_x == 0
          ? input.num_cols() / tableblockdim_x
          : input.num_cols() / tableblockdim_x + 1;
  chk_cu_err(cudaMalloc((void **)(&d_aggregated_rowwise_sums),
                        n_recursivefilter_step4_overblocksright_cols *
                            input.num_rows() * sizeof(float)));
  Logger::new_line(
      "Aggregated rowwise sum table dims: (" +
      std::to_string(n_recursivefilter_step4_overblocksright_cols) + ", " +
      std::to_string(input.num_rows()) + ")");

  float *d_finalsums;
  chk_cu_err(cudaMalloc((void **)(&d_finalsums),
                        input.num_rows() * input.num_cols() * sizeof(float)));

  const int threadgriddim_x = input.num_cols() % tableblockdim_x == 0
                                  ? input.num_cols() / tableblockdim_x
                                  : input.num_cols() / tableblockdim_x + 1;
  const int threadgriddim_y = input.num_rows() % tableblockdim_y == 0
                                  ? input.num_rows() / tableblockdim_y
                                  : input.num_rows() / tableblockdim_y + 1;
  const int threadgriddim2_x = threadgriddim_x % tableblockdim_x == 0
                                   ? threadgriddim_x / tableblockdim_x
                                   : threadgriddim_x / tableblockdim_x + 1;
  const int threadgriddim2_y = threadgriddim_y % tableblockdim_y == 0
                                   ? threadgriddim_y / tableblockdim_y
                                   : threadgriddim_y / tableblockdim_y + 1;

  Logger::new_line("#blocks: (" + std::to_string(threadgriddim_x) + ", " +
                   std::to_string(threadgriddim_y) + ")");
  Logger::new_line("#blocks 2: (" + std::to_string(threadgriddim2_x) + ", " +
                   std::to_string(threadgriddim2_y) + ")");
  const int tableblock_shmem_size_bytes =
      (tableblockdim_x + SHMEM_PAD_X) * tableblockdim_y * sizeof(float);
  float run_time_allruns_ms = -1.0f;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i_run = 0; i_run < num_kernel_runs; ++i_run) {
    recursivefilter_step1_inblocksdownright<<<
        dim3(threadgriddim_x, threadgriddim_y),
        dim3(std::max(tableblockdim_x, tableblockdim_y)),
        tableblock_shmem_size_bytes>>>(
        d_input, input.num_rows(), input.num_cols(), tableblockdim_x,
        tableblockdim_y, filter_coeff_0, filter_coeff_1,
        d_blockwise_colwise_sums, d_blockwise_rowwise_sums);
    recursivefilter_step2_overblocksdown<<<dim3(threadgriddim_x),
                                           dim3(tableblockdim_x)>>>(
        n_step2_overblocksdown_rows, input.num_cols(), tableblockdim_y,
        filter_coeff_0, filter_coeff_1, d_blockwise_colwise_sums,
        d_aggregated_colwise_sums);
    recursivefilter_step3_inoverblockscolsummedblocksright<<<
        dim3(threadgriddim2_x, threadgriddim2_y),
        dim3(tableblockdim_x, tableblockdim_y)>>>(
        n_step3_inoverblockscolsummedblocksright_rows, input.num_cols(),
        n_step3_inoverblockscolsummedblocksright_cols, tableblockdim_x,
        filter_coeff_0, filter_coeff_1, d_aggregated_colwise_sums,
        d_blockwise_rowwise_aggregatedcolsums);
    recursivefilter_step4_overblocksright<<<dim3(1, threadgriddim_y),
                                            dim3(1, tableblockdim_y)>>>(
        input.num_rows(), n_recursivefilter_step4_overblocksright_cols,
        n_step3_inoverblockscolsummedblocksright_rows, tableblockdim_x,
        filter_coeff_0, filter_coeff_1, d_blockwise_rowwise_sums,
        d_blockwise_rowwise_aggregatedcolsums, d_aggregated_rowwise_sums);
    recursivefilter_step5_inblocksdownright<<<
        dim3(threadgriddim_x, threadgriddim_y),
        dim3(std::max(tableblockdim_x, tableblockdim_y)),
        tableblock_shmem_size_bytes>>>(
        d_input, input.num_rows(), input.num_cols(), tableblockdim_x,
        tableblockdim_y, filter_coeff_0, filter_coeff_1,
        d_aggregated_colwise_sums, d_aggregated_rowwise_sums, d_finalsums);
    // recursivefilter_downright_blocky<tableblockdim_x, tableblockdim_y>
    //	<<<threadgriddim, dim3(std::max(tableblockdim_x, tableblockdim_y))>>>
    //		(d_input, input.num_rows(), input.num_cols(),
    //		filter_coeff_0, filter_coeff_1, d_blockwise_colwise_sums,
    //		d_blockwise_rowwise_sums, d_aggregated_colwise_sums,
    //		d_blockwise_rowwise_aggregatedcolsums,
    // d_aggregated_rowwise_sums, 		d_output);
  }
  // cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&run_time_allruns_ms, start, stop);
  const float run_time_1run_ms = run_time_allruns_ms / float(num_kernel_runs);
  Logger::new_line(
      "\nKernel execution time for " + std::to_string(input.num_cols()) + "x" +
      std::to_string(input.num_rows()) +
      " [ms]: " + std::to_string(run_time_1run_ms) + " (average of " +
      std::to_string(num_kernel_runs) + " runs)");

  float *h_blockwise_colwise_sums = (float *)malloc(
      n_blockwise_colwise_rows * input.num_cols() * sizeof(float));
  chk_cu_err(
      cudaMemcpy(h_blockwise_colwise_sums, d_blockwise_colwise_sums,
                 n_blockwise_colwise_rows * input.num_cols() * sizeof(float),
                 cudaMemcpyDeviceToHost));
  CpuTable blockwise_colwise_sums(n_blockwise_colwise_rows, input.num_cols(),
                                  h_blockwise_colwise_sums);
  if (n_blockwise_colwise_rows <= 12 && input.num_cols() <= 12) {
    Logger::new_line("\nBlockwise-colwise table (light blue):\n" +
                     blockwise_colwise_sums.toString());
  }
  float *h_step1_inblocksdownright = (float *)malloc(
      n_step1_inblocksdownright_cols * input.num_rows() * sizeof(float));
  chk_cu_err(cudaMemcpy(h_step1_inblocksdownright, d_blockwise_rowwise_sums,
                        n_step1_inblocksdownright_cols * input.num_rows() *
                            sizeof(float),
                        cudaMemcpyDeviceToHost));
  CpuTable blockwise_rowwise_sums(n_step1_inblocksdownright_cols,
                                  input.num_rows(), h_step1_inblocksdownright);
  blockwise_rowwise_sums.transpose();
  if (input.num_rows() <= 12 && n_step1_inblocksdownright_cols <= 12) {
    Logger::new_line("\nBlockwise-rowwise table (light green):\n" +
                     blockwise_rowwise_sums.toString());
  }

  float *h_step2_overblocksdown = (float *)malloc(
      n_step2_overblocksdown_rows * input.num_cols() * sizeof(float));
  chk_cu_err(
      cudaMemcpy(h_step2_overblocksdown, d_aggregated_colwise_sums,
                 n_step2_overblocksdown_rows * input.num_cols() * sizeof(float),
                 cudaMemcpyDeviceToHost));
  CpuTable aggregated_blockwise_colwise_sums(
      n_step2_overblocksdown_rows, input.num_cols(), h_step2_overblocksdown);
  if (n_step2_overblocksdown_rows <= 12 && input.num_cols() <= 12) {
    Logger::new_line("\nAggregated blockwise-colwise table (dark blue):\n" +
                     aggregated_blockwise_colwise_sums.toString());
  }

  float *h_step3_inoverblockscolsummedblocksright = (float *)malloc(
      n_step3_inoverblockscolsummedblocksright_cols *
      n_step3_inoverblockscolsummedblocksright_rows * sizeof(float));
  chk_cu_err(cudaMemcpy(h_step3_inoverblockscolsummedblocksright,
                        d_blockwise_rowwise_aggregatedcolsums,
                        n_step3_inoverblockscolsummedblocksright_cols *
                            n_step3_inoverblockscolsummedblocksright_rows *
                            sizeof(float),
                        cudaMemcpyDeviceToHost));
  CpuTable blockwise_rowwise_aggregatedcolsums(
      n_step3_inoverblockscolsummedblocksright_cols,
      n_step3_inoverblockscolsummedblocksright_rows,
      h_step3_inoverblockscolsummedblocksright);
  blockwise_rowwise_aggregatedcolsums.transpose();
  if (n_step3_inoverblockscolsummedblocksright_rows <= 12 &&
      n_step3_inoverblockscolsummedblocksright_cols <= 12) {
    Logger::new_line("\nBlockwise-rowwise aggregatedcolsum table (red):\n" +
                     blockwise_rowwise_aggregatedcolsums.toString());
  }

  float *h_step4_overblocksright =
      (float *)malloc(n_recursivefilter_step4_overblocksright_cols *
                      input.num_rows() * sizeof(float));
  chk_cu_err(cudaMemcpy(h_step4_overblocksright, d_aggregated_rowwise_sums,
                        n_recursivefilter_step4_overblocksright_cols *
                            input.num_rows() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CpuTable aggregated_rowwise_sums(n_recursivefilter_step4_overblocksright_cols,
                                   input.num_rows(), h_step4_overblocksright);
  aggregated_rowwise_sums.transpose();
  if (input.num_rows() <= 12 &&
      n_recursivefilter_step4_overblocksright_cols <= 12) {
    Logger::new_line("\nAggregated rowwise table (dark green):\n" +
                     aggregated_rowwise_sums.toString());
  }

  float *h_final_sums =
      (float *)malloc(input.num_rows() * input.num_cols() * sizeof(float));
  chk_cu_err(cudaMemcpy(h_final_sums, d_finalsums,
                        input.num_rows() * input.num_cols() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CpuTable final_sums(input.num_rows(), input.num_cols(), h_final_sums);
  if (final_sums.num_rows() <= PRINT_LIMIT_Y &&
      final_sums.num_cols() <= PRINT_LIMIT_X) {
    Logger::new_line("\nFinal sums:\n" + final_sums.toString());
  }

  switch (output_step) {
  case STEP_1: {
    outputs[0].reset(n_step1_inblocksdownright_rows, input.num_cols(),
                     h_blockwise_colwise_sums);
    outputs[1].reset(n_step1_inblocksdownright_cols, input.num_rows(),
                     h_step1_inblocksdownright);
    outputs[1].transpose();
    break;
  }
  case STEP_2: {
    outputs[0].reset(n_step2_overblocksdown_rows, input.num_cols(),
                     h_step2_overblocksdown);
    break;
  }
  case STEP_3: {
    outputs[0].reset(n_step3_inoverblockscolsummedblocksright_cols,
                     n_step3_inoverblockscolsummedblocksright_rows,
                     h_step3_inoverblockscolsummedblocksright);
    outputs[0].transpose();
    break;
  }
  case STEP_4: {
    outputs[0].reset(n_recursivefilter_step4_overblocksright_cols,
                     input.num_rows(), h_step4_overblocksright);
    outputs[0].transpose();
    break;
  }
  case STEP_5_FINAL: {
    outputs[0].reset(input.num_rows(), input.num_cols(), h_final_sums);
    break;
  }
  }

  chk_cu_err(cudaFree(d_input));
  free(h_input);
  chk_cu_err(cudaFree(d_blockwise_colwise_sums));
  free(h_blockwise_colwise_sums);
  chk_cu_err(cudaFree(d_blockwise_rowwise_sums));
  free(h_step1_inblocksdownright);
  chk_cu_err(cudaFree(d_aggregated_colwise_sums));
  free(h_step2_overblocksdown);
  chk_cu_err(cudaFree(d_blockwise_rowwise_aggregatedcolsums));
  free(h_step3_inoverblockscolsummedblocksright);
  chk_cu_err(cudaFree(d_aggregated_rowwise_sums));
  free(h_step4_overblocksright);
  chk_cu_err(cudaFree(d_finalsums));
  free(h_final_sums);

  return run_time_1run_ms;
}

} // namespace gpuacademy
