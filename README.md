# TODO

* Use __ldg
* Texture mem for input
* Use intrinsic functions for calculations
* Template so you can unroll the loops
* Pad global memory
* Switch to half precision
* Don't calculate last row/col of aggreg values (unused)

# Tried and helped

* Making shared memory accesses bank conflict-free by padding
* Coalescing global memory accesses

# Tried but didn't help

* Doing steps 2, 3, 4 with parallel-scan
 * One thread block per column: bad occupancy
 * One thread block for N columns: inherent shared memory bank conflicts between the scan strips
* Implementing specific fast pow(float, int)

# Troubleshooting

## '/RTC1' and '/O2' command-line options are incompatible

Remove the "-O3" flag from CUDA_NVCC_FLAGS in the corresponding CMakeLists.txt.

## nvcc not found

export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Links

https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html