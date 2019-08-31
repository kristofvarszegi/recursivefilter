#include <gtest/gtest.h>

#include "helpers.hpp"

#include "Logger.hpp"

using namespace gpuacademy;

const float kMaxAbsError = 0.0005f;

TEST(CpuTable, setSawTooth) {
  const size_t n_rows = 3, n_cols = 4;
  CpuTable input(n_rows, n_cols);
  const float amplitude = 1.2f;
  const int period = 3;
  const float ground_truth_data[] = {
      -amplitude, 0.0f, amplitude, -amplitude, 0.0f, amplitude,
      -amplitude, 0.0f, amplitude, -amplitude, 0.0f, amplitude};
  CpuTable ground_truth(n_rows, n_cols, ground_truth_data);
  Logger::new_line("Input:" + input.toString());
  input.setSawTooth(amplitude, period);
  Logger::new_line("Result:" + input.toString());
  Logger::new_line("Ground truth:" + ground_truth.toString());
  Logger::new_line();
  ASSERT_TRUE(input.equals(ground_truth, kMaxAbsError));
}

TEST(CpuTable, transpose) {
  const size_t n_rows = 2, n_cols = 3;
  const int data[] = {1, 2, 3, 4, 5, 6};
  CpuTable input(n_rows, n_cols, data);
  const float ground_truth_data[] = {1, 4, 2, 5, 3, 6};
  CpuTable ground_truth(n_cols, n_rows, ground_truth_data);
  Logger::new_line("Input:" + input.toString());
  input.transpose();
  Logger::new_line("Result:" + input.toString());
  Logger::new_line("Ground truth:" + ground_truth.toString());
  Logger::new_line();
  ASSERT_TRUE(input.equals(ground_truth, kMaxAbsError));
}