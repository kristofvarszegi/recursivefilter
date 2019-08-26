#include "helpers.hpp"

#include "Logger.hpp"
#include "cpufuncs.hpp"
#include "gpufuncs.hpp"
#include "utils.hpp"

#include <fstream>

namespace gpuacademy {

comparison_result_t recursivefilter_and_compare_gpuvscpu(
    const CpuTable &input, float filter_coeff_0, float filter_coeff_1,
    int tableblockdim_x, int tableblockdim_y, int num_kernel_runs,
    OUTPUT_STEP output_step, float max_abs_error, bool print_tables,
    bool save_csv) {

  std::vector<CpuTable> summed_area_table_in_vec;
  summed_area_table_in_vec.emplace_back(
      CpuTable(input.num_rows(), input.num_cols()));
  const float runtime_1kernelrun_ms = recursivefilter_downright_gpu(
      input, filter_coeff_0, filter_coeff_1, tableblockdim_x, tableblockdim_y,
      num_kernel_runs, output_step, summed_area_table_in_vec);

  CpuTable ground_truth(input.num_rows(), input.num_cols());
  Logger::new_line();
  Logger::new_line("Calculating SAT CPU for reference...");
  Logger::new_line();
  recursivefilter_downright_cpu(input, filter_coeff_0, filter_coeff_1,
                                ground_truth);

  if (print_tables) {
    Logger::new_line();
    Logger::new_line(input.toString());
    Logger::new_line(ground_truth.toString());
    Logger::new_line(summed_area_table_in_vec[0].toString());
    Logger::new_line();
  }
  if (save_csv) {
    save_to_csv(summed_area_table_in_vec[0], "summed_area_table.csv");
    save_to_csv(summed_area_table_in_vec[0], "ground_truth.csv");
  }

  Logger::new_line();
  comparison_result_t comparison_result;
  comparison_result.equals =
      ground_truth.equals(summed_area_table_in_vec[0], max_abs_error);
  comparison_result.runtime_1kernel_ms = runtime_1kernelrun_ms;
  return comparison_result;
}

void save_to_csv(const CpuTable &table, const std::string &filename) {
  std::ofstream table_file(filename);
  for (int i_row = 0; i_row < table.num_rows(); ++i_row) {
    for (int i_col = 0; i_col < table.num_cols(); ++i_col) {
      table_file << table.get(i_row, i_col) << ",";
    }
    table_file << std::endl;
  }
  table_file.close();
}

} // namespace gpuacademy
