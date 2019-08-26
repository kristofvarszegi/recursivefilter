#pragma once

#include "CpuTable.hpp"
#include "gpufuncs.hpp"

#include <ctime>

namespace gpuacademy {

struct comparison_result_t {
  int equals;
  float runtime_1kernel_ms;
};

comparison_result_t recursivefilter_and_compare_gpuvscpu(
    const CpuTable &input, float filter_coeff_0, float filter_coeff_1,
    int tableblockdim_x, int tableblockdim_y, int num_kernel_runs,
    OUTPUT_STEP output_step, float max_abs_error, bool print_tables,
    bool save_csv);

void save_to_csv(const CpuTable &table, const std::string &filename);

} // namespace gpuacademy
