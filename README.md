# TODO

* Template so you can unroll the loops
* Use intrinsic functions for calculations
* Coeff pows to constant memory
* Coeff pows by mult-aggreg
* Align global memory accesses to 128byte
* Pad global memory
* cudaHostMalloc() for pinned area
* Texture mem for input
* Switch to half precision
* Don't calculate last row/col of aggreg values (unused)

# Tried and helped

* Making shared memory accesses bank conflict-free by padding
* Coalescing global memory accesses
* const __restrict__ on global arrays
* Fill final sum table as col-major and transpose afterwards with cuBLAS
* Fill final sum table with tiled transpose within step5

# Tried but didn't help

* Doing steps 2, 3, 4 with parallel-scan
 * One thread block per column: bad occupancy
 * One thread block for N columns: inherent shared memory bank conflicts between the scan strips
* Reading input-only global arrays with __ldg(.)
* Implementing specific fast pow(float, int)

# Troubleshooting

## '/RTC1' and '/O2' command-line options are incompatible

Remove the "-O3" flag from CUDA_NVCC_FLAGS in the corresponding CMakeLists.txt.

## nvcc not found

export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Links

https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html