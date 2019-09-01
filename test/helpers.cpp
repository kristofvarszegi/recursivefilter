#include "helpers.hpp"

#include "Logger.hpp"
#include "cpufuncs.hpp"
#include "gpufuncs.hpp"
#include "utils.hpp"

namespace gpuacademy {

template comparison_result_t recursivefilter_and_compare_gpuvscpu<config::kBlockDim2dGridSmall, config::kBlockDim1dGridSmall, config::kNumKernelRunsFew>(
    const CpuTable &input, float feedfwd_coeff, float feedback_coeff,
    OUTPUT_STEP output_step, float max_abs_error, bool print_tables,
    bool save_csv);
template comparison_result_t recursivefilter_and_compare_gpuvscpu<config::kBlockDim2dGridLarge, config::kBlockDim1dGridLarge, config::kNumKernelRunsFew>(
    const CpuTable &input, float feedfwd_coeff, float feedback_coeff,
    OUTPUT_STEP output_step, float max_abs_error, bool print_tables,
    bool save_csv);
template comparison_result_t recursivefilter_and_compare_gpuvscpu<config::kBlockDim2dGridLarge, config::kBlockDim1dGridLarge, config::kNumKernelRunsMany>(
    const CpuTable &input, float feedfwd_coeff, float feedback_coeff,
    OUTPUT_STEP output_step, float max_abs_error, bool print_tables,
    bool save_csv);
template <int BLOCKDIM_2DGRID, int BLOCKDIM_1DGRID, int NUM_KERNEL_RUNS>
comparison_result_t recursivefilter_and_compare_gpuvscpu(
    const CpuTable &input, float feedfwd_coeff, float feedback_coeff,
    OUTPUT_STEP output_step, float max_abs_error, bool print_tables,
    bool save_csv) {

  std::vector<CpuTable> summed_area_table_in_vec;
  summed_area_table_in_vec.emplace_back(
      CpuTable(input.num_rows(), input.num_cols()));
  const float runtime_1kernelrun_ms = recursivefilter_downright_gpu<BLOCKDIM_2DGRID, BLOCKDIM_1DGRID, NUM_KERNEL_RUNS>(
      input, feedfwd_coeff, feedback_coeff,
      output_step, summed_area_table_in_vec);

  CpuTable ground_truth(input.num_rows(), input.num_cols());
  Logger::new_line();
  Logger::new_line("Calculating SAT CPU for reference...");
  Logger::new_line();
  recursivefilter_downright_cpu(input, feedfwd_coeff, feedback_coeff,
                                ground_truth);

  if (print_tables) {
    Logger::new_line();
    Logger::new_line(input.toString());
    Logger::new_line(ground_truth.toString());
    Logger::new_line(summed_area_table_in_vec[0].toString());
    Logger::new_line();
  }
  if (save_csv) {
    summed_area_table_in_vec[0].saveToCsv("summed_area_table.csv");
    summed_area_table_in_vec[0].saveToCsv("ground_truth.csv");
  }

  Logger::new_line();
  comparison_result_t comparison_result;
  comparison_result.equals =
      ground_truth.equals(summed_area_table_in_vec[0], max_abs_error);
  comparison_result.runtime_1kernel_ms = runtime_1kernelrun_ms;
  return comparison_result;
}

} // namespace gpuacademy
