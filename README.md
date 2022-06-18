
# 2D recursive filter implementation on GPU (C++, CUDA)

Given a table of floating point numbers, apply the filter "y_i = feedfwd_coeff * x_i + feedback_coeff * y_i-1" along columns then along rows, for arbitrary table sizes. For the details of the algorithm, see [Nehab et al.: GPU-Efficient Recursive Filtering and Summed-Area Tables](https://github.com/kristofvarszegi/recursivefilter/blob/master/GPU-Efficient%20Recursive%20Filtering%20and%20Summed-Area%20Tables.pdf).

This repo also contains the authors' implementation ("gpufilter") as a submodule for the purpose of benchmarking.

<p align="center">
  <img src="https://user-images.githubusercontent.com/15958029/174431842-23bb3900-b552-43d8-9d64-926bab9c6ef2.png" alt="recursive-filter-problem-figure" width="480"/>
  <br>Figure taken from the article (describing the authors' algorithm) linked above
</p>

## How to build

./build.sh

## How to run

./run.sh

## How run on arbitrary image

Replace arbitrary.png with your image and run.

## Platform requirements

* Ubuntu 18.04
* CUDA 10.1
* GTest
* OpenCV

## Troubleshooting

### '/RTC1' and '/O2' command-line options are incompatible

Remove the "-O3" flag from CUDA_NVCC_FLAGS in the corresponding CMakeLists.txt.

### nvcc not found

export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

## Miscellaneous notes

### Tried and helped

* Making shared memory accesses bank conflict-free by padding
* Coalescing global memory accesses
* const __restrict__ on global arrays
* Fill final sum table as col-major and transpose afterwards with cuBLAS
* Fill final sum table with tiled transpose within step5

### Tried but didn't help

* Doing steps 2, 3, 4 with parallel-scan
 * One thread block per column: led to low occupancy
 * One thread block for N columns: inherent shared memory bank conflicts between the scan strips
 * Using warp shuffle functions: the shared memory writes after the scan offset the pros because of bank conflicts
* Increasing occupancy for steps 1 and 5: limited by shared memory
* Reading input-only global arrays with __ldg(.)
* Unrolling "for" loops in kernels by templating
* Aligning thread block dim to 128byte for global memory accesses in 2dgrid kernels: optimum also considering between block limitation and shared memory limitation was not divisor of 128
* Aligning tables to 128bytes for global memory accesses
* Using single precision intrinsics for calculations: better to leave it to the compiler
* Using texture memory for input: global memory accesses are already optimized for cache (same for using surface objects)
* Implementing specific fast pow(float, int)

(See repository branches for some of the abandoned ways.)

### Further runtime optimization possibilities

* Use half precision number format - depends on the requirements of the domain of usage
* Don't calculate the unused last row/col of the aggregated tables
* Find parallelizable kernel sections, and parallelize them using streams

### Useful links

https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html
