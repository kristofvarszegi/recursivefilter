# TODO

* Parallelize kernels with streams
* Profile and fix occupancy
* Template so you can unroll the loops
* Don't calculate last row/col of aggreg values (unused)

# Tried and helped

* Making shared memory accesses bank conflict-free by padding
* Coalescing global memory accesses

# Tried but didn't help

* Implementing specific fast pow(float, int)

# Troubleshooting

## '/RTC1' and '/O2' command-line options are incompatible

Remove the "-O3" flag from CUDA_NVCC_FLAGS in the corresponding CMakeLists.txt.
