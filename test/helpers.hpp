#pragma once

#include "CpuTable.hpp"
#include "gpufuncs.hpp"

#include <ctime>

namespace gpuacademy {

struct comparison_result_t {
  int equals;
  float runtime_1kernel_ms;
};

template <int BLOCKDIM_2DGRID, int BLOCKDIM_1DGRID, int NUM_KERNEL_RUNS>
comparison_result_t recursivefilter_and_compare_gpuvscpu(
    const CpuTable &input, float feedfwd_coeff, float feedback_coeff,
    OUTPUT_STEP output_step, float max_abs_error, bool print_tables,
    bool save_csv);

} // namespace gpuacademy
