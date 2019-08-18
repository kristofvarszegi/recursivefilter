cmake_minimum_required(VERSION 2.8)
project(gpuacademy_recursivefiltering)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

set(EXECUTABLE_NAME filter)
cuda_add_executable(${EXECUTABLE_NAME}
    src/main.cpp
	src/cpu_funcs.hpp src/cpu_funcs.cpp
	src/gpu_funcs.hpp src/gpu_funcs.cu
	src/utils.hpp src/utils.cpp
)

install(TARGETS ${EXECUTABLE_NAME} DESTINATION ${CMAKE_CURRENT_LIST_DIR}/bin)