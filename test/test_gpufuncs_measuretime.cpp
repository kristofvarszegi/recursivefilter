#include <gtest/gtest.h>

#include "helpers.hpp"

#include "CpuTable.hpp"
#include "Logger.hpp"
#include "cpufuncs.hpp"
#include "gpufuncs.hpp"
#include "utils.hpp"

#include <vector>

using namespace gpuacademy;

const float kMaxAbsError = 0.005f;
const float kMaxAuthorsBigTableRunTimeMs = 0.25f;
const float kMaxBigTableRunTimeMs = 0.4f;
const float kMaxHugeTableRunTimeMs = 1.5f;
const float kSatFilterCoeffs[] = {1.0f, 1.0f};
const float kArbitratyFilterCoeffs[] = {1.723f, 0.546f};
const int kAuthorsBigTableSizeX = 1024, kAuthorsBigTableSizeY = 1024;
const int kBigTableSizeX = 2007, kBigTableSizeY = 1003;
const int kHugeTableSizeX = 4007, kHugeTableSizeY = 2003;
const int kNumKernelRunsForMeasuringTime = 1;

TEST(GPU_funcs_measuretime, authorsbigtable_arbitraryfill_arbitrarycoeffs) {
  const size_t n_rows = kAuthorsBigTableSizeY, n_cols = kAuthorsBigTableSizeX;
  CpuTable input(n_rows, n_cols);
  input.setSawTooth(0.18f, 29);
  Logger::new_line("Authors' table size: (" + std::to_string(n_cols) + ", " +
                   std::to_string(n_rows) + ")");

  const comparison_result_t comparison_result =
      recursivefilter_and_compare_gpuvscpu(
          input, kArbitratyFilterCoeffs[0], kArbitratyFilterCoeffs[1],
          config::BLOCK_SIZE_CLASS::LARGE, kNumKernelRunsForMeasuringTime,
          OUTPUT_STEP::STEP_6, kMaxAbsError, false, false);
  EXPECT_TRUE(comparison_result.equals);
  ASSERT_LT(comparison_result.runtime_1kernel_ms, kMaxAuthorsBigTableRunTimeMs);
}

TEST(GPU_funcs_measuretime, DISABLED_bigtable_arbitraryfill_arbitrarycoeffs) {
  const size_t n_rows = kBigTableSizeY, n_cols = kBigTableSizeX;
  CpuTable input(n_rows, n_cols);
  input.setSawTooth(0.18f, 29);
  Logger::new_line("Big table size: (" + std::to_string(n_cols) + ", " +
                   std::to_string(n_rows) + ")");

  const comparison_result_t comparison_result =
      recursivefilter_and_compare_gpuvscpu(
          input, kArbitratyFilterCoeffs[0], kArbitratyFilterCoeffs[1],
          config::BLOCK_SIZE_CLASS::LARGE, kNumKernelRunsForMeasuringTime,
          OUTPUT_STEP::STEP_6, kMaxAbsError, false, false);
  EXPECT_TRUE(comparison_result.equals);
  ASSERT_LT(comparison_result.runtime_1kernel_ms, kMaxBigTableRunTimeMs);
}

TEST(GPU_funcs_measuretime, DISABLED_hugetable_arbitraryfill_arbitrarycoeffs) {
  const size_t n_rows = kHugeTableSizeY, n_cols = kHugeTableSizeX;
  CpuTable input(n_rows, n_cols);
  input.setSawTooth(0.18f, 29);
  Logger::new_line("Huge table size: (" + std::to_string(n_cols) + ", " +
                   std::to_string(n_rows) + ")");

  const comparison_result_t comparison_result =
      recursivefilter_and_compare_gpuvscpu(
          input, kArbitratyFilterCoeffs[0], kArbitratyFilterCoeffs[1],
          config::BLOCK_SIZE_CLASS::LARGE, kNumKernelRunsForMeasuringTime,
          OUTPUT_STEP::STEP_6, kMaxAbsError, false, false);
  EXPECT_TRUE(comparison_result.equals);
  ASSERT_LT(comparison_result.runtime_1kernel_ms, kMaxHugeTableRunTimeMs);
}
