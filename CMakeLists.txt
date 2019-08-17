cmake_minimum_required(VERSION 2.8)
project(gpuacademy_sat)

#set(CUDA_TOOLKIT_ROOT_DIR )
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_20,code=sm_20)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

cuda_add_executable(sat
    src/main.cu
)

target_link_libraries(sat ${OpenCV_LIBS})

install(TARGETS sat DESTINATION ${CMAKE_CURRENT_LIST_DIR}/bin)