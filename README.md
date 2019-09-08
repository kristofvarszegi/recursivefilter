# TODO

* Texture mem for input
* Surface mem for output
* Turn on sync

# Mathematical specification

Given a table of floating point numbers, apply the filter "y_i = feedfwd_coeff * x_i + feedback_coeff * y_i-1" along columns then along rows. Support arbitrary table sizes.

# How to build and run

To build: ./build.sh
To run: ./run.sh

# Platform requirements

* Ubuntu 18.04
* CUDA 10.1

# Tried and helped

* Making shared memory accesses bank conflict-free by padding
* Coalescing global memory accesses
* const __restrict__ on global arrays
* Fill final sum table as col-major and transpose afterwards with cuBLAS
* Fill final sum table with tiled transpose within step5

# Tried but didn't help

* Doing steps 2, 3, 4 with parallel-scan
 * One thread block per column: led to low occupancy
 * One thread block for N columns: inherent shared memory bank conflicts between the scan strips
 * Using warp shuffle functions: the shared memory writes after the scan offset the pros because of bank conflicts
* Reading input-only global arrays with __ldg(.)
* Unrolling "for" loops in kernels by templating
* Aligning thread block dim to 128byte for global memory accesses in 2dgrid kernels: optimum also considering between block limitation and shared memory limitation was not divisor of 128
* Aligning tables to 128bytes for global memory accesses
* Using single precision intrinsics for calculations: better to leave it to the compiler
* Implementing specific fast pow(float, int)

# Further runtime optimization possibilities

* Use half precision number format - depends on the requirements of the domain of usage
* Don't calculate the unused last row/col of the aggregated tables
* Find parallelizable kernel sections, and parallelize them using streams

# Troubleshooting

## '/RTC1' and '/O2' command-line options are incompatible

Remove the "-O3" flag from CUDA_NVCC_FLAGS in the corresponding CMakeLists.txt.

## nvcc not found

export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Links

https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html
