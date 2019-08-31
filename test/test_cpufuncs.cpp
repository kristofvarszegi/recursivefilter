#include <gtest/gtest.h>

#include "helpers.hpp"

#include "Logger.hpp"
#include "cpufuncs.hpp"
#include "gpufuncs.hpp"
#include "utils.hpp"

using namespace gpuacademy;

const float kMaxAbsError = 0.001f;
const float kSatFilterCoeffs[] = {1.0f, 1.0f};
const float kArbitratyFilterCoeffs[] = {1.723f, 0.546f};

TEST(CPU_funcs, naive_sat) {
  const int n_rows = 5, n_cols = 5;
  const int input_data[] = {
      1, 0, 0, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 0, 1, 1, 0, 1, 1, 0, 1, 2, 1, 2,
  };
  CpuTable input(n_rows, n_cols, input_data);

  const int ground_truth_data[] = {
      1,  1,  1, 2, 3,  3,  4,  5, 7,  10, 4,  7,  9,
      13, 16, 5, 9, 11, 16, 20, 5, 10, 14, 20, 26,
  };
  CpuTable ground_truth(n_rows, n_cols, ground_truth_data);

  CpuTable summed_area_table(n_rows, n_cols);
  calculate_summedareatable_cpu_naive(input, summed_area_table);
  Logger::new_line("SAT-naive table:" + summed_area_table.toString());
  Logger::new_line();
  Logger::new_line();

  ASSERT_TRUE(ground_truth.equals(summed_area_table, kMaxAbsError));
}

TEST(CPU_funcs, filtery_sat) {
  const int n_rows = 5, n_cols = 5;
  const int input_data[] = {
      1, 0, 0, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 0, 1, 1, 0, 1, 1, 0, 1, 2, 1, 2,
  };
  CpuTable input(n_rows, n_cols, input_data);

  CpuTable ground_truth(n_rows, n_cols);
  calculate_summedareatable_cpu_naive(input, ground_truth);

  CpuTable summed_area_table(n_rows, n_cols);
  recursivefilter_downright_cpu(input, kSatFilterCoeffs[0], kSatFilterCoeffs[1],
                                summed_area_table);
  Logger::new_line("SAT-filtered table:" + summed_area_table.toString());
  Logger::new_line();
  Logger::new_line();

  ASSERT_TRUE(ground_truth.equals(summed_area_table, kMaxAbsError));
}

TEST(CPU_funcs, arbitrary_coeffs_filter) {
  const int n_rows = 5, n_cols = 5;
  const int input_data[] = {
      1, 0, 0, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 0, 1, 1, 0, 1, 1, 0, 1, 2, 1, 2,
  };
  CpuTable input(n_rows, n_cols, input_data);

  const float ground_truth_data[] = {
      2.968729f,         1.620926034f,      0.885025614564f,
      3.45195298555194f, 4.85349533011136f, 7.558384034f,
      7.095606682564f,   6.84293024867995f, 8.32589494977925f,
      12.1043226765795f, 7.095606682564f,   11.4325852826799f,
      10.8318465983433f, 14.3575978912594f, 11.9661261311916f,
      6.84293024867995f, 10.8318465983433f, 8.42013989125942f,
      12.1762270487436f, 11.8702241832939f, 3.73623991577925f,
      8.88291724269542f, 12.1557804146276f, 13.743826651178f,
      16.2928016527584f,
  };
  CpuTable ground_truth(n_rows, n_cols, ground_truth_data);

  CpuTable filtered_table(n_rows, n_cols);
  recursivefilter_downright_cpu(input, kArbitratyFilterCoeffs[0],
                                kArbitratyFilterCoeffs[1], filtered_table);
  Logger::new_line("Filtered table:" + filtered_table.toString());
  Logger::new_line();
  Logger::new_line();

  ASSERT_TRUE(ground_truth.equals(filtered_table, kMaxAbsError));
}