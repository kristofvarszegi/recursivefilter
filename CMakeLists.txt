cmake_minimum_required(VERSION 2.8.12)

project(recursivefilter)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

################################################################################


find_package(CUDA REQUIRED)
message("CUDA_DIR: \"${CUDA_DIR}\"")
message("CUDA_TOOLKIT_ROOT_DIR: \"${CUDA_TOOLKIT_ROOT_DIR}\"")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -use_fast_math --ptxas-options=-v")

set(LIB_NAME ${CMAKE_PROJECT_NAME})

set(HEADERS
    include/${LIB_NAME}/cpufuncs.hpp
    include/${LIB_NAME}/CpuTable.hpp
    include/${LIB_NAME}/gpufuncs.hpp
    include/${LIB_NAME}/Logger.hpp
    include/${LIB_NAME}/utils.hpp
)
set(SOURCES
    src/cpufuncs.cpp
    src/CpuTable.cpp
    src/gpufuncs.cu
    src/Logger.cpp
    src/utils.cpp
)

include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
include_directories(BEFORE include/${LIB_NAME})

cuda_add_library(${LIB_NAME} ${HEADERS} ${SOURCES})
#add_library(${LIB_NAME} ${HEADERS} ${SOURCES})
#target_include_directories(${LIB_NAME} PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/include)
#target_include_directories(${LIB_NAME} PUBLIC include/${LIB_NAME})
get_property(incl_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
message("incl_dirs: \"${incl_dirs}\"")
link_directories(${LIB_NAME} ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
target_link_libraries(${LIB_NAME} cuda)
set_target_properties(${LIB_NAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")

message("CMAKE_INSTALL_PREFIX: \"${CMAKE_INSTALL_PREFIX}\"")
message("CMAKE_INSTALL_LIBDIR: \"${CMAKE_INSTALL_LIBDIR}\"")
message("CMAKE_INSTALL_BINDIR: \"${CMAKE_INSTALL_BINDIR}\"")
message("CMAKE_CURRENT_BINARY_DIR: \"${CMAKE_CURRENT_BINARY_DIR}\"")
install(TARGETS EXPORT ${LIB_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${LIB_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
#install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

################################################################################

enable_testing()

find_package(GTest REQUIRED)
message("GTest_DIR: \"${GTest_DIR}\"")
#find_library(LIBNAME ${LIB_NAME} HINTS ${CMAKE_INSTALL_BINDIR})

set(TEST_NAME test_${LIB_NAME})

add_executable(${TEST_NAME}
    ${HEADERS}
    test/main.cpp
    test/helpers.hpp test/helpers.cpp
    test/test_cpufuncs.cpp
    test/test_CpuTable.cpp
    test/test_gpufuncs.cpp
)
#target_include_directories(${TEST_NAME} PUBLIC include/${LIB_NAME})
#get_property(incl_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
#message("incl_dirs: \"${incl_dirs}\"")
link_directories(${CMAKE_CURRENT_BINARY_DIR})
#get_property(lnk_dirs DIRECTORY PROPERTY LINK_DIRECTORIES)
#message("lnk_dirs: \"${lnk_dirs}\"")
target_link_libraries(${TEST_NAME} ${LIB_NAME}
#    ${CMAKE_CURRENT_BINARY_DIR}/lib${LIB_NAME}.a
#    ${CMAKE_CURRENT_BINARY_DIR}/lib${LIB_NAME}.so
)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(${TEST_NAME}
        ${GTEST_LIBRARIES}
        pthread)
else()
    target_link_libraries(${TEST_NAME} "GTest::Main")
endif ()
#get_target_property(lnk_libs ${TEST_NAME} LINK_LIBRARIES)
#message("lnk_libs: \"${lnk_libs}\"")

add_test(
    ${TEST_NAME}
    COMMAND ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/${TEST_NAME}
)

install(TARGETS EXPORT ${TEST_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
