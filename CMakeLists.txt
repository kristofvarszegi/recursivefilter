cmake_minimum_required(VERSION 2.8)
project(gpuacademy_sat)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

cuda_add_executable(sat
    src/main.cu
)

install(TARGETS sat DESTINATION ${CMAKE_CURRENT_LIST_DIR}/bin)