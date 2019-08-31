# TODO

* Scan in step2 using shared memory
* Scan in step3 using shared memory
* Scan in step4 using shared memory
* Profile and fix occupancy: min 64 threads/block
* Remove colwise sum table from step1 globals
* Parallelize kernels with streams
* Use __ldg
* Scan at in-block aggregs
* Texture mem for input
* Use intrinsic functions for calculations
* Template so you can unroll the loops
* Pad global memory
* Don't calculate last row/col of aggreg values (unused)

# Tried and helped

* Making shared memory accesses bank conflict-free by padding
* Coalescing global memory accesses

# Tried but didn't help

* Implementing specific fast pow(float, int)

# Troubleshooting

## '/RTC1' and '/O2' command-line options are incompatible

Remove the "-O3" flag from CUDA_NVCC_FLAGS in the corresponding CMakeLists.txt.

## nvcc not found

export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
