#include "Logger.hpp"
#include "gpufuncs.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <stdexcept>

#define SHMEM_PAD_X 1
#define PRINT_LIMIT_X 8
#define PRINT_LIMIT_Y 8
#define SAVE_TABLES_TO_CSV false

namespace gpuacademy {

namespace config {

size_t get_blockdim_2dgrid_x(config::BLOCK_SIZE_CLASS bsc) {
  switch (bsc) {
  case config::BLOCK_SIZE_CLASS::SMALL:
    return 4;
    break;
  case config::BLOCK_SIZE_CLASS::LARGE:
    return 28;
    break;
  default:
    throw std::runtime_error("Invalid block size class: " +
                             std::to_string(bsc));
  }
}

size_t get_blockdim_2dgrid_y(config::BLOCK_SIZE_CLASS bsc) {
  switch (bsc) {
  case config::BLOCK_SIZE_CLASS::SMALL:
    return 4;
    break;
  case config::BLOCK_SIZE_CLASS::LARGE:
    return 28;
    break;
  default:
    throw std::runtime_error("Invalid block size class: " +
                             std::to_string(bsc));
  }
}

size_t get_blockdim_1dgrid_x(config::BLOCK_SIZE_CLASS bsc) {
  switch (bsc) {
  case config::BLOCK_SIZE_CLASS::SMALL:
    return 7;
    break;
  case config::BLOCK_SIZE_CLASS::LARGE:
    return 64;
    break;
  default:
    throw std::runtime_error("Invalid block size class: " +
                             std::to_string(bsc));
  }
}

} // namespace config

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

__global__ void recursivefilter_step1_inblocksdownright(
    const float* __restrict__ input, int num_rows, int num_cols, float filter_coeff_0,
    float filter_coeff_1, float* __restrict__ blockwise_colwise_sums,
    float* __restrict__ blockwise_rowwise_sums) {
  const int global_tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_tid_y = blockIdx.y * blockDim.x + threadIdx.x;
  // Yes, threadIdx.x (not .y), as we have a 1D thread array within a thread
  // block
  extern __shared__ float colwisesums_thisblock[];

  if (global_tid_x < num_cols) {
    float aggregated_sum, prev_aggregated_sum = 0.0f;
#pragma unroll
    for (int y_in_thisblock = 0; y_in_thisblock < blockDim.x;
         ++y_in_thisblock) {
      if (blockIdx.y * blockDim.x + y_in_thisblock < num_rows) {
        aggregated_sum =
            filter_coeff_0 *
            __ldg(
                (const float *)&input[global_tid_x + (blockIdx.y * blockDim.x +
                                                      y_in_thisblock) *
                                                         num_cols]);
        aggregated_sum += filter_coeff_1 * prev_aggregated_sum;
      }
      colwisesums_thisblock[threadIdx.x +
                            y_in_thisblock * (blockDim.x + SHMEM_PAD_X)] =
          aggregated_sum;
      prev_aggregated_sum = aggregated_sum;
    }
    __syncthreads();
    blockwise_colwise_sums[global_tid_x + blockIdx.y * num_cols] =
        aggregated_sum;
  }

  if (global_tid_y < num_rows && threadIdx.x < blockDim.x) {
    float aggregated_sum, prev_aggregated_sum = 0.0f;
#pragma unroll
    for (int x_in_thisblock = 0; x_in_thisblock < blockDim.x;
         ++x_in_thisblock) {
      if (blockIdx.x * blockDim.x + x_in_thisblock < num_cols) {
        aggregated_sum =
            filter_coeff_0 *
            colwisesums_thisblock[x_in_thisblock +
                                  threadIdx.x * (blockDim.x + SHMEM_PAD_X)];
        aggregated_sum += filter_coeff_1 * prev_aggregated_sum;
      }
      prev_aggregated_sum = aggregated_sum;
    }
    blockwise_rowwise_sums[(blockIdx.y * blockDim.x + threadIdx.x) +
                           blockIdx.x * num_rows] = aggregated_sum;
    // Transposed to coalesce global memory access
  }
}

//__device__ inline int ilog2(int x) {
//	return INT_BIT_SIZE - __clz(x);
//}

__global__ void recursivefilter_step2_overblocksdown(
    int num_aggregated_rows, int num_cols, float filter_coeff_1_toblockdimypow,
    const float* __restrict__ blockwise_colwise_sums, float* __restrict__ aggregated_colwise_sums) {
  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_tid < num_cols) {
    float aggregated_sum, prev_aggregated_sum = 0.0f;
    for (int y_in_grid = 0; y_in_grid < num_aggregated_rows; ++y_in_grid) {
      aggregated_sum =
          __ldg((const float *)&blockwise_colwise_sums[global_tid +
                                                       y_in_grid * num_cols]) +
          filter_coeff_1_toblockdimypow * prev_aggregated_sum;
      prev_aggregated_sum = aggregated_sum;
      aggregated_colwise_sums[global_tid + y_in_grid * num_cols] =
          aggregated_sum;
    }
  }
}

__global__ void recursivefilter_step3_inoverblockscolsummedblocksright(
    int num_aggregated_rows, int num_cols, int num_aggregated_cols,
    int tableblockdim_x, float filter_coeff_0, float filter_coeff_1,
    const float* __restrict__ aggregated_colwise_sums,
    float* __restrict__ blockwise_rowwise_aggregatedcolsums) {
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
        aggregated_sum = filter_coeff_0 *
                         __ldg((const float *)&aggregated_colwise_sums
                                   [global_x_offset + global_tid_y * num_cols]);
        aggregated_sum += filter_coeff_1 * prev_aggregated_sum;
      }
      prev_aggregated_sum = aggregated_sum;
    }
    blockwise_rowwise_aggregatedcolsums[global_tid_y +
                                        global_tid_x * num_aggregated_rows] =
        aggregated_sum;
    // Transposed to coalesce global memory access
  }
}

__global__ void recursivefilter_step4_overblocksright(
    int num_rows, int num_aggregated_cols, int num_aggregated_rows,
	int num_rows_in2dblock,
    float filter_coeff_1, float filter_coeff_1_toblockdimxpow,
    const float* __restrict__ blockwise_rowwise_sums,
    const float* __restrict__ blockwise_rowwise_aggregatedcolsums,
    float* __restrict__ aggregated_rowwise_sums) {
  const int global_tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (global_tid_y < num_rows) {
    float aggregated_sum, prev_aggregated_sum = 0.0f;
	const int bwrwaggcs_row_id = global_tid_y / num_rows_in2dblock;
    for (int x_in_row = 0; x_in_row < num_aggregated_cols; ++x_in_row) {
      aggregated_sum =
		  __ldg((const float*)&blockwise_rowwise_sums[global_tid_y + x_in_row * num_rows]) +
          filter_coeff_1_toblockdimxpow * prev_aggregated_sum;
      // Transposed to coalesce global memory access
      if (bwrwaggcs_row_id > 0) {
        aggregated_sum +=
            powf(filter_coeff_1, (global_tid_y % num_rows_in2dblock) + 1) *
			__ldg((const float*)&blockwise_rowwise_aggregatedcolsums[(bwrwaggcs_row_id - 1) +
                                                x_in_row * num_aggregated_rows]);
        // Transposed to coalesce global memory access
      }
      prev_aggregated_sum = aggregated_sum;
      aggregated_rowwise_sums[global_tid_y + x_in_row * num_rows] =
          aggregated_sum;
      // Transposed to coalesce global memory access
    }
  }
}

__global__ void recursivefilter_step5_inblocksdownright(
    const float* __restrict__ input, int num_rows, int num_cols, float filter_coeff_0,
    float filter_coeff_1, const float* __restrict__ aggregated_colwise_sums,
    const float* __restrict__ aggregated_rowwise_sums, float* __restrict__ final_sums) {
  const int global_tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_tid_y = blockIdx.y * blockDim.x + threadIdx.x;
  // Yes, blockDim.x and threadIdx.x (not .y), as we have a 1D thread array
  // within a thread block
  const int thisblock_globalstart_x = blockIdx.x * blockDim.x;
  const int thisblock_globalstart_y = blockIdx.y * blockDim.x;
  extern __shared__ float aggregated_sums_thisblock[];
  
  const int y_in_thisblock_upper =
	  (thisblock_globalstart_y + blockDim.x >= num_rows)
	  ? (num_rows - thisblock_globalstart_y)
	  : blockDim.x;

  if (global_tid_x < num_cols) {
    float aggregated_sum, prev_aggregated_sum = 0.0f;
    if (blockIdx.y > 0) {
      prev_aggregated_sum =
		  __ldg((const float*)&aggregated_colwise_sums[global_tid_x + (blockIdx.y - 1) * num_cols]);
    }
    for (int y_in_thisblock = 0; y_in_thisblock < y_in_thisblock_upper;
         ++y_in_thisblock) {
      aggregated_sum =
          filter_coeff_0 *
		  __ldg((const float*)&input[global_tid_x +
                    (thisblock_globalstart_y + y_in_thisblock) * num_cols]) +
          filter_coeff_1 * prev_aggregated_sum;
      prev_aggregated_sum = aggregated_sum;
      aggregated_sums_thisblock[threadIdx.x +
                                        y_in_thisblock *
                                            (blockDim.x + SHMEM_PAD_X)] =
          aggregated_sum;
    }
    __syncthreads();
  }

  if (global_tid_y < num_rows) {
    float aggregated_sum, prev_aggregated_sum = 0.0f;
    if (blockIdx.x > 0) {
      prev_aggregated_sum =
		  __ldg((const float*)&aggregated_rowwise_sums[global_tid_y + (blockIdx.x - 1) * num_rows]);
      // Transposed to coalesce global memory access
    }
    const int x_in_thisblock_upper =
        (thisblock_globalstart_x + blockDim.x >= num_cols)
            ? (num_cols - thisblock_globalstart_x)
            : blockDim.x;
    for (int x_in_thisblock = 0; x_in_thisblock < x_in_thisblock_upper;
         ++x_in_thisblock) {
      aggregated_sum =
          filter_coeff_0 *
              aggregated_sums_thisblock[x_in_thisblock +
                                                threadIdx.x * (blockDim.x +
                                                               SHMEM_PAD_X)] +
          filter_coeff_1 * prev_aggregated_sum; // Yes, threadIdx.x (not .y)
      prev_aggregated_sum = aggregated_sum;
	  aggregated_sums_thisblock[x_in_thisblock + threadIdx.x * (blockDim.x + SHMEM_PAD_X)] = aggregated_sum;
    }
  }

  if (global_tid_x < num_cols) {
	  for (int y_in_thisblock = 0; y_in_thisblock < y_in_thisblock_upper; ++y_in_thisblock) {
		  final_sums[global_tid_x + (thisblock_globalstart_y + y_in_thisblock) * num_cols] = 
			  aggregated_sums_thisblock[threadIdx.x + y_in_thisblock * (blockDim.x + SHMEM_PAD_X)];
	  }
  }
}

float recursivefilter_downright_gpu(const CpuTable &input, float filter_coeff_0,
                                    float filter_coeff_1,
                                    config::BLOCK_SIZE_CLASS block_size_class,
                                    size_t num_kernel_runs,
                                    OUTPUT_STEP output_step,
                                    std::vector<CpuTable> &outputs) {
  if (input.num_rows() < 2) {
    throw std::runtime_error("Number of input rows must be at least 2");
  }
  if (input.num_cols() < 2) {
    throw std::runtime_error("Number of input cols must be at least 2");
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
  //texture<float, 2, cudaReadModeElementType> input_texture;
  float *d_input;
  chk_cu_err(cudaMalloc((void **)(&d_input),
                        input.num_rows() * input.num_cols() * sizeof(float)));
  chk_cu_err(cudaMemcpy(d_input, h_input,
                        input.num_rows() * input.num_cols() * sizeof(float),
                        cudaMemcpyHostToDevice));
 
  const size_t blockdim_2dgrid_x = get_blockdim_2dgrid_x(block_size_class);
  const size_t blockdim_2dgrid_y = get_blockdim_2dgrid_y(block_size_class);
  const size_t blockdim_1dgrid_x = get_blockdim_1dgrid_x(block_size_class);

  const dim3 blockdim_step1 =
      dim3(int(std::max(blockdim_2dgrid_x, blockdim_2dgrid_y)), 1, 1);
  // const dim3 threadblockdim_step2 = dim3(1, int(threadgriddim_localscale_y),
  // 1);
  const dim3 blockdim_step2 = dim3(int(blockdim_1dgrid_x), 1, 1);
  const dim3 blockdim_step3 = dim3(int(blockdim_1dgrid_x), 1, 1);
  const dim3 blockdim_step4 = dim3(1, int(blockdim_1dgrid_x), 1);
  const dim3 blockdim_step5 =
      dim3(int(std::max(blockdim_2dgrid_x, blockdim_2dgrid_y)), 1, 1);
  Logger::new_line("#threads in step 1's block: (" +
                   std::to_string(blockdim_step1.x) + ", " +
                   std::to_string(blockdim_step1.y) + ")");
  Logger::new_line("#threads in step 2's block: (" +
                   std::to_string(blockdim_step2.x) + ", " +
                   std::to_string(blockdim_step2.y) + ")");
  Logger::new_line("#threads in step 3's block: (" +
                   std::to_string(blockdim_step3.x) + ", " +
                   std::to_string(blockdim_step3.y) + ")");
  Logger::new_line("#threads in step 4's block: (" +
                   std::to_string(blockdim_step4.x) + ", " +
                   std::to_string(blockdim_step4.y) + ")");
  Logger::new_line("#threads in step 5's block: (" +
                   std::to_string(blockdim_step5.x) + ", " +
                   std::to_string(blockdim_step5.y) + ")");

  const size_t griddim_for2dblock_x =
      input.num_cols() % blockdim_2dgrid_x == 0
          ? input.num_cols() / blockdim_2dgrid_x
          : input.num_cols() / blockdim_2dgrid_x + 1;
  const size_t griddim_for2dblock_y =
      input.num_rows() % blockdim_2dgrid_y == 0
          ? input.num_rows() / blockdim_2dgrid_y
          : input.num_rows() / blockdim_2dgrid_y + 1;
  const size_t griddim_for1dblock_x =
      input.num_cols() % blockdim_1dgrid_x == 0
          ? input.num_cols() / blockdim_1dgrid_x
          : input.num_cols() / blockdim_1dgrid_x + 1;
  const size_t griddim_for1dblock_y =
	  input.num_rows() % blockdim_1dgrid_x == 0
	  ? input.num_rows() / blockdim_1dgrid_x
	  : input.num_rows() / blockdim_1dgrid_x + 1;
  
  const dim3 griddim_step1(int(griddim_for2dblock_x), int(griddim_for2dblock_y),
                           1);
  // const dim3 threadgriddim_step2(int(input.num_cols()), 1, 1);
  const dim3 griddim_step2(int(griddim_for1dblock_x), 1, 1);
  const dim3 griddim_step3(griddim_step1.x % blockdim_step3.x == 0
                               ? griddim_step1.x / blockdim_step3.x
                               : griddim_step1.x / blockdim_step3.x + 1,
                           int(griddim_for2dblock_y), 1);
  const dim3 griddim_step4(1, int(griddim_for1dblock_y), 1);
  const dim3 griddim_step5(int(griddim_for2dblock_x), int(griddim_for2dblock_y),
                           1);
  Logger::new_line("#blocks in step 1: (" + std::to_string(griddim_step1.x) +
                   ", " + std::to_string(griddim_step1.y) + ")");
  Logger::new_line("#blocks in step 2: (" + std::to_string(griddim_step2.x) +
                   ", " + std::to_string(griddim_step2.y) + ")");
  Logger::new_line("#blocks in step 3: (" + std::to_string(griddim_step3.x) +
                   ", " + std::to_string(griddim_step3.y) + ")");
  Logger::new_line("#blocks in step 4: (" + std::to_string(griddim_step4.x) +
                   ", " + std::to_string(griddim_step4.y) + ")");
  Logger::new_line("#blocks in step 5: (" + std::to_string(griddim_step5.x) +
                   ", " + std::to_string(griddim_step5.y) + ")");

  const size_t shmemsizebytes_step1 =
      (blockdim_step1.x + SHMEM_PAD_X) * blockdim_step1.x *
      sizeof(float); // Yes, .x as step1 has a 1D thread block spanning 2
                     // directions
  // const size_t shmemsizebytes_step2 = (tableblockdim_x + SHMEM_PAD_X) *
  //                                    threadgriddim_localscale_y *
  //                                    sizeof(float);
  const size_t shmemsizebytes_step5 =
      (blockdim_step5.x + SHMEM_PAD_X) * blockdim_step5.x *
      sizeof(float); // Yes, .x as step1 has a 1D thread block spanning 2
                     // directions

  const float filter_coeff_1_toblockdimypow =
      powf(filter_coeff_1, static_cast<float>(blockdim_2dgrid_y));
  const float filter_coeff_1_toblockdimxpow =
      powf(filter_coeff_1, static_cast<float>(blockdim_2dgrid_x));
  // const int n_step2_scansteps =
  // static_cast<int>(log2(threadblockdim_step2.y));

  float *d_step1_blockwise_colwise_sums;
  const size_t n_step1_inblocksdown_rows = griddim_step1.y;
  chk_cu_err(
      cudaMalloc((void **)(&d_step1_blockwise_colwise_sums),
                 input.num_cols() * n_step1_inblocksdown_rows * sizeof(float)));
  Logger::new_line("Blockwise-colwise table dims: (" +
                   std::to_string(input.num_cols()) + ", " +
                   std::to_string(n_step1_inblocksdown_rows) + ")");

  float
      *d_step1_blockwise_rowwise_sums; // Transposed to coalesce global memory accesses
  const size_t n_step1_inblocksdownright_cols = griddim_step1.x;
  chk_cu_err(cudaMalloc((void **)(&d_step1_blockwise_rowwise_sums),
                        input.num_rows() * n_step1_inblocksdownright_cols *
                            sizeof(float)));
  Logger::new_line("Blockwise-rowwise table dims (transposed!): (" +
                   std::to_string(input.num_rows()) + ", " +
                   std::to_string(n_step1_inblocksdownright_cols) + ")");

  float *d_step2_aggregated_colwise_sums;
  const size_t n_step2_overblocksdown_rows = n_step1_inblocksdown_rows;
  chk_cu_err(cudaMalloc((void **)(&d_step2_aggregated_colwise_sums),
                        input.num_cols() * n_step2_overblocksdown_rows *
                            sizeof(float)));
  Logger::new_line("Aggregated colwise table dims: (" +
                   std::to_string(input.num_cols()) + ", " +
                   std::to_string(n_step2_overblocksdown_rows) + ")");

  float *d_step3_blockwise_rowwise_aggregatedcolsums;	// Transposed to coalesce global memory accesses
  const size_t n_step3_inoverblockscolsummedblocksright_rows =
      n_step2_overblocksdown_rows;
  const size_t n_step3_inoverblockscolsummedblocksright_cols =
      n_step1_inblocksdownright_cols;
  chk_cu_err(cudaMalloc((void **)(&d_step3_blockwise_rowwise_aggregatedcolsums),
                        n_step3_inoverblockscolsummedblocksright_cols *
                            n_step3_inoverblockscolsummedblocksright_rows *
                            sizeof(float)));
  Logger::new_line(
      "Blockwise-rowwise aggregatedcolsum table dims: (" +
      std::to_string(n_step3_inoverblockscolsummedblocksright_cols) + ", " +
      std::to_string(n_step3_inoverblockscolsummedblocksright_rows) + ")");

  float *d_step4_aggregated_rowwise_sums;	// Transposed to coalesce global memory accesses
  const size_t n_recursivefilter_step4_overblocksright_cols =
      n_step1_inblocksdownright_cols;
  chk_cu_err(cudaMalloc((void **)(&d_step4_aggregated_rowwise_sums),
                        n_recursivefilter_step4_overblocksright_cols *
                            input.num_rows() * sizeof(float)));
  Logger::new_line(
      "Aggregated rowwise sum table dims: (" +
      std::to_string(n_recursivefilter_step4_overblocksright_cols) + ", " +
      std::to_string(input.num_rows()) + ")");

  float *d_step5_finalsums_rowmajor;
  chk_cu_err(cudaMalloc((void **)(&d_step5_finalsums_rowmajor), input.num_rows() * input.num_cols() * sizeof(float)));

  float run_time_allruns_ms = -1.0f;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (size_t i_run = 0; i_run < num_kernel_runs; ++i_run) {
    recursivefilter_step1_inblocksdownright<<<griddim_step1, blockdim_step1,
                                              shmemsizebytes_step1>>>(
        d_input, int(input.num_rows()), int(input.num_cols()), filter_coeff_0,
        filter_coeff_1, d_step1_blockwise_colwise_sums, d_step1_blockwise_rowwise_sums);
    recursivefilter_step2_overblocksdown<<<griddim_step2, blockdim_step2>>>(
        int(n_step2_overblocksdown_rows), int(input.num_cols()),
        filter_coeff_1_toblockdimypow, d_step1_blockwise_colwise_sums,
        d_step2_aggregated_colwise_sums);
    recursivefilter_step3_inoverblockscolsummedblocksright<<<griddim_step3,
                                                             blockdim_step3>>>(
        int(n_step3_inoverblockscolsummedblocksright_rows),
        int(input.num_cols()),
        int(n_step3_inoverblockscolsummedblocksright_cols),
        int(blockdim_2dgrid_x), filter_coeff_0, filter_coeff_1,
        d_step2_aggregated_colwise_sums, d_step3_blockwise_rowwise_aggregatedcolsums);
    recursivefilter_step4_overblocksright<<<griddim_step4, blockdim_step4>>>(
        int(input.num_rows()),
        int(n_recursivefilter_step4_overblocksright_cols),
        int(n_step3_inoverblockscolsummedblocksright_rows), blockdim_step1.x, filter_coeff_1,
        filter_coeff_1_toblockdimxpow, d_step1_blockwise_rowwise_sums,
        d_step3_blockwise_rowwise_aggregatedcolsums, d_step4_aggregated_rowwise_sums);
    recursivefilter_step5_inblocksdownright<<<griddim_step5, blockdim_step5,
                                              shmemsizebytes_step5>>>(
        d_input, int(input.num_rows()), int(input.num_cols()), filter_coeff_0,
        filter_coeff_1, d_step2_aggregated_colwise_sums, d_step4_aggregated_rowwise_sums,
        d_step5_finalsums_rowmajor);
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
      n_step1_inblocksdown_rows * input.num_cols() * sizeof(float));
  chk_cu_err(
      cudaMemcpy(h_blockwise_colwise_sums, d_step1_blockwise_colwise_sums,
                 n_step1_inblocksdown_rows * input.num_cols() * sizeof(float),
                 cudaMemcpyDeviceToHost));
  CpuTable blockwise_colwise_sums(n_step1_inblocksdown_rows, input.num_cols(),
                                  h_blockwise_colwise_sums);
  if (n_step1_inblocksdown_rows <= 12 && input.num_cols() <= 12) {
    Logger::new_line("\nBlockwise-colwise table (light blue):\n" +
                     blockwise_colwise_sums.toString());
  }
  float *h_step1_inblocksdownright = (float *)malloc(
      n_step1_inblocksdownright_cols * input.num_rows() * sizeof(float));
  chk_cu_err(cudaMemcpy(h_step1_inblocksdownright, d_step1_blockwise_rowwise_sums,
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
      cudaMemcpy(h_step2_overblocksdown, d_step2_aggregated_colwise_sums,
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
                        d_step3_blockwise_rowwise_aggregatedcolsums,
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
  chk_cu_err(cudaMemcpy(h_step4_overblocksright, d_step4_aggregated_rowwise_sums,
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

  float *h_step5_finalsums_rowmajor =
      (float *)malloc(input.num_rows() * input.num_cols() * sizeof(float));
  chk_cu_err(cudaMemcpy(h_step5_finalsums_rowmajor, d_step5_finalsums_rowmajor,
                        input.num_rows() * input.num_cols() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CpuTable finalsums_rowmajor(input.num_rows(), input.num_cols(), h_step5_finalsums_rowmajor);
  if (finalsums_rowmajor.num_rows() <= PRINT_LIMIT_Y &&
	  finalsums_rowmajor.num_cols() <= PRINT_LIMIT_X) {
	  Logger::new_line("\nFinal sums:\n" + finalsums_rowmajor.toString());
  }

  if (SAVE_TABLES_TO_CSV) {
	  blockwise_colwise_sums.saveToCsv("step1_blockwise_colwise_sums.csv");
	  blockwise_rowwise_sums.saveToCsv("step1_blockwise_rowwise_sums.csv");
	  aggregated_blockwise_colwise_sums.saveToCsv(
		  "step2_aggregated_blockwise_colwise_sums.csv");
	  blockwise_rowwise_aggregatedcolsums.saveToCsv(
		  "step2_blockwise_rowwise_aggregatedcolsums.csv");
	  aggregated_rowwise_sums.saveToCsv("step4_aggregated_rowwise_sums.csv");
	  finalsums_rowmajor.saveToCsv("step5_finalsums_rowmajor.csv");
  }

  switch (output_step) {
  case STEP_1: {
    outputs[0].reset(n_step1_inblocksdown_rows, input.num_cols(),
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
  case STEP_5: {
	outputs[0].reset(input.num_rows(), input.num_cols(), h_step5_finalsums_rowmajor);
    break;
  }
  default:
	  throw std::runtime_error("Invalid output step requested: " + std::to_string(output_step));
  }

  chk_cu_err(cudaFree(d_input));
  free(h_input);
  chk_cu_err(cudaFree(d_step1_blockwise_colwise_sums));
  free(h_blockwise_colwise_sums);
  chk_cu_err(cudaFree(d_step1_blockwise_rowwise_sums));
  free(h_step1_inblocksdownright);
  chk_cu_err(cudaFree(d_step2_aggregated_colwise_sums));
  free(h_step2_overblocksdown);
  chk_cu_err(cudaFree(d_step3_blockwise_rowwise_aggregatedcolsums));
  free(h_step3_inoverblockscolsummedblocksright);
  chk_cu_err(cudaFree(d_step4_aggregated_rowwise_sums));
  free(h_step4_overblocksright);
  chk_cu_err(cudaFree(d_step5_finalsums_rowmajor));
  free(h_step5_finalsums_rowmajor);

  return run_time_1run_ms;
}

} // namespace gpuacademy
