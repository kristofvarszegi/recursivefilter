cmake_minimum_required(VERSION 2.8)
project(recursivefiltering)

enable_testing()

################################################################################

find_package(CUDA REQUIRED)
message("CUDA_NVCC_FLAGS before:" ${CUDA_NVCC_FLAGS})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -use_fast_math)

set(LIB_NAME ${CMAKE_PROJECT_NAME})

set(HEADERS
    include/recursivefiltering/cpufuncs.hpp
    include/recursivefiltering/CpuTable.hpp
    include/recursivefiltering/gpufuncs.hpp
    include/recursivefiltering/Logger.hpp
    include/recursivefiltering/utils.hpp
)
set(SOURCES
    src/cpufuncs.cpp
    src/CpuTable.cpp
    src/gpufuncs.cu
    src/Logger.cpp
    src/utils.cpp
)
cuda_add_library(${LIB_NAME} ${HEADERS} ${SOURCES})
target_include_directories(${LIB_NAME} PUBLIC include/recursivefiltering)
set_target_properties(${LIB_NAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")

install(TARGETS ${LIB_NAME} DESTINATION ${PROJECT_SOURCE_DIR}/bin)

install(TARGETS EXPORT ${LIB_NAME}Config
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION include)

#install(EXPORT ${LIB_NAME}Config DESTINATION share/${LIB_NAME}/cmake)
#export(TARGETS ${LIB_NAME} FILE ${LIB_NAME}Config.cmake)

################################################################################

find_package(GTest REQUIRED)

set(TEST_NAME ${LIB_NAME}_tests)

add_executable(${TEST_NAME}
    ${HEADERS}
    test/main.cpp
    test/test_cpufuncs.cpp
    test/test_gpufuncs.cpp
)
target_include_directories(${TEST_NAME} PUBLIC include/recursivefiltering)
target_link_libraries(${TEST_NAME}
    ${LIB_NAME}
    GTest::Main
)

add_test(
    ${TEST_NAME}
    COMMAND ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/${TEST_NAME}
)