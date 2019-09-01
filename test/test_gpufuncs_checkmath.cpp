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

TEST(GPU_funcs_checkmath_smalltables, step1_satcoeffs) {
  const size_t n_rows = 5, n_cols = 5;
  const int input_data[] = {
      1, 0, 0, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 0, 1, 1, 0, 1, 1, 0, 1, 2, 1, 2,
  };
  CpuTable input(n_rows, n_cols, &(*input_data));

  const size_t n_gt0_rows = (n_rows / config::get_blockdim_2dgrid_y(
                                          config::BLOCK_SIZE_CLASS::SMALL) +
                             1),
               n_gt0_cols = n_cols;
  const int ground_truth_0_data[] = {5, 4, 2, 5, 4, 0, 1, 2, 1, 2};
  CpuTable ground_truth_0(n_gt0_rows, n_gt0_cols, &(*ground_truth_0_data));

  const size_t n_gt1_rows = n_rows,
               n_gt1_cols = (n_cols / config::get_blockdim_2dgrid_x(
                                          config::BLOCK_SIZE_CLASS::SMALL) +
                             1);
  const int ground_truth_1_data[] = {
      2, 1, 7, 3, 13, 3, 16, 4, 4, 2,
  };
  CpuTable ground_truth_1(n_gt1_rows, n_gt1_cols, &(*ground_truth_1_data));

  std::vector<CpuTable> step1_outputs;
  step1_outputs.emplace_back(n_gt0_rows, n_gt0_cols);
  step1_outputs.emplace_back(n_gt1_rows, n_gt1_cols);
  try {
    const float runtime_1kernelrun_ms = recursivefilter_downright_gpu(
        input, kSatFilterCoeffs[0], kSatFilterCoeffs[1],
        config::BLOCK_SIZE_CLASS::SMALL, 1, OUTPUT_STEP::STEP_1, step1_outputs);
  } catch (const std::exception &e) {
    Logger::new_line("Error: " + std::string(e.what()));
    FAIL();
  }
  Logger::new_line("Step 1 outputs:");
  Logger::new_line(step1_outputs[0].toString());
  Logger::new_line();
  Logger::new_line(step1_outputs[1].toString());
  Logger::new_line();

  const bool output0_equals =
      step1_outputs[0].equals(ground_truth_0, kMaxAbsError);
  EXPECT_TRUE(output0_equals);
  const bool output1_equals =
      step1_outputs[1].equals(ground_truth_1, kMaxAbsError);
  EXPECT_TRUE(output1_equals);
}

TEST(GPU_funcs_checkmath_smalltables, step2_satcoeffs) {
  const size_t n_rows = 5, n_cols = 5;
  const int input_data[] = {
      1, 0, 0, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 0, 1, 1, 0, 1, 1, 0, 1, 2, 1, 2,
  };
  CpuTable input(n_rows, n_cols, &(*input_data));

  const size_t n_gt_rows = (n_rows / config::get_blockdim_2dgrid_y(
                                         config::BLOCK_SIZE_CLASS::SMALL) +
                            1),
               n_gt_cols = n_cols;
  const int ground_truth_data[] = {5, 4, 2, 5, 4, 5, 5, 4, 6, 6};
  CpuTable ground_truth(n_gt_rows, n_gt_cols, &(*ground_truth_data));

  std::vector<CpuTable> step2_outputs;
  step2_outputs.emplace_back(n_gt_rows, n_gt_cols);
  const float runtime_1kernelrun_ms = recursivefilter_downright_gpu(
      input, kSatFilterCoeffs[0], kSatFilterCoeffs[1],
      config::BLOCK_SIZE_CLASS::SMALL, 1, OUTPUT_STEP::STEP_2, step2_outputs);
  Logger::new_line("Step 2 output:");
  Logger::new_line(step2_outputs[0].toString());
  Logger::new_line();

  ASSERT_TRUE(step2_outputs[0].equals(ground_truth, kMaxAbsError));
}

TEST(GPU_funcs_checkmath_smalltables, step3_satcoeffs) {
  const size_t n_rows = 5, n_cols = 5;
  const int input_data[] = {
      1, 0, 0, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 0, 1, 1, 0, 1, 1, 0, 1, 2, 1, 2,
  };
  CpuTable input(n_rows, n_cols, &(*input_data));

  const size_t n_gt_rows = (n_rows / get_blockdim_2dgrid_y(
                                         config::BLOCK_SIZE_CLASS::SMALL) +
                            1),
               n_gt_cols = (n_cols / config::get_blockdim_2dgrid_x(
                                         config::BLOCK_SIZE_CLASS::SMALL) +
                            1);
  const int ground_truth_data[] = {16, 4, 20, 6};
  CpuTable ground_truth(n_gt_rows, n_gt_cols, &(*ground_truth_data));

  std::vector<CpuTable> step3_outputs;
  step3_outputs.emplace_back(n_gt_rows, n_gt_cols);
  const float runtime_1kernelrun_ms = recursivefilter_downright_gpu(
      input, kSatFilterCoeffs[0], kSatFilterCoeffs[1],
      config::BLOCK_SIZE_CLASS::SMALL, 1, OUTPUT_STEP::STEP_3, step3_outputs);
  Logger::new_line("Step 3 output:");
  Logger::new_line(step3_outputs[0].toString());
  Logger::new_line();

  ASSERT_TRUE(step3_outputs[0].equals(ground_truth, kMaxAbsError));
}

TEST(GPU_funcs_checkmath_smalltables, step4_satcoeffs) {
  const size_t n_rows = 5, n_cols = 5;
  const int input_data[] = {
      1, 0, 0, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 0, 1, 1, 0, 1, 1, 0, 1, 2, 1, 2,
  };
  CpuTable input(n_rows, n_cols, &(*input_data));

  const size_t n_gt_rows = 5,
               n_gt_cols = (n_cols / config::get_blockdim_2dgrid_x(
                                         config::BLOCK_SIZE_CLASS::SMALL) +
                            1);
  const int ground_truth_data[] = {
      2, 3, 7, 10, 13, 16, 16, 20, 20, 26,
  };
  CpuTable ground_truth(n_gt_rows, n_gt_cols, &(*ground_truth_data));

  std::vector<CpuTable> step4_outputs;
  step4_outputs.emplace_back(n_gt_rows, n_gt_cols);
  const float runtime_1kernelrun_ms = recursivefilter_downright_gpu(
      input, kSatFilterCoeffs[0], kSatFilterCoeffs[1],
      config::BLOCK_SIZE_CLASS::SMALL, 1, OUTPUT_STEP::STEP_4, step4_outputs);
  Logger::new_line("Step 4 output:");
  Logger::new_line(step4_outputs[0].toString());
  Logger::new_line();

  ASSERT_TRUE(step4_outputs[0].equals(ground_truth, kMaxAbsError));
}

TEST(GPU_funcs_checkmath_smalltables, step5_satcoeffs) {
  const size_t n_rows = 5, n_cols = 5;
  const int input_data[] = {
      1, 0, 0, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 0, 1, 1, 0, 1, 1, 0, 1, 2, 1, 2,
  };
  CpuTable input(n_rows, n_cols, &(*input_data));

  const size_t n_gt_rows = 5, n_gt_cols = 5;
  const int ground_truth_data[] = {
      1,  1,  1, 2, 3,  3,  4,  5, 7,  10, 4,  7,  9,
      13, 16, 5, 9, 11, 16, 20, 5, 10, 14, 20, 26,
  };
  CpuTable ground_truth(n_gt_rows, n_gt_cols, &(*ground_truth_data));

  std::vector<CpuTable> step5_outputs;
  step5_outputs.emplace_back(n_gt_rows, n_gt_cols);
  const float runtime_1kernelrun_ms = recursivefilter_downright_gpu(
      input, kSatFilterCoeffs[0], kSatFilterCoeffs[1],
      config::BLOCK_SIZE_CLASS::SMALL, 1, OUTPUT_STEP::STEP_5,
      step5_outputs);
  Logger::new_line("Step 5 output:");
  Logger::new_line(step5_outputs[0].toString());
  Logger::new_line();

  ASSERT_TRUE(step5_outputs[0].equals(ground_truth, kMaxAbsError));
}

TEST(GPU_funcs_checkmath_smalltables, step1_arbitrarycoeffs) {
  const size_t n_rows = 5, n_cols = 5;
  const int input_data[] = {
      1, 0, 0, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 0, 1, 1, 0, 1, 1, 0, 1, 2, 1, 2,
  };
  CpuTable input(n_rows, n_cols, &(*input_data));

  const size_t n_gt0_rows = (n_rows / get_blockdim_2dgrid_y(
                                          config::BLOCK_SIZE_CLASS::SMALL) +
                             1),
               n_gt0_cols = n_cols;
  const float ground_truth_0_data[] = {
      3.971520747928f, 4.118169868f, 1.454411868f, 4.398624879928f,
      3.030762747928f, 0.0f,         1.723f,       3.446f,
      1.723f,          3.446f};
  CpuTable ground_truth_0(n_gt0_rows, n_gt0_cols, &(*ground_truth_0_data));

  const size_t n_gt1_rows = n_rows,
               n_gt1_cols = (n_cols / config::get_blockdim_2dgrid_x(
                                          config::BLOCK_SIZE_CLASS::SMALL) +
                             1);
  const float ground_truth_1_data[] = {
      3.45195298555194f, 2.968729f,       8.32589494977925f, 7.558384034f,
      14.3575978912594f, 4.126877682564f, 12.1762270487436f, 5.22200421467995f,
      7.095606682564f,   5.937458f,
  };
  CpuTable ground_truth_1(n_gt1_rows, n_gt1_cols, &(*ground_truth_1_data));

  std::vector<CpuTable> outputs;
  outputs.emplace_back(n_gt0_rows, n_gt0_cols);
  outputs.emplace_back(n_gt1_rows, n_gt1_cols);
  const float runtime_1kernelrun_ms = recursivefilter_downright_gpu(
      input, kArbitratyFilterCoeffs[0], kArbitratyFilterCoeffs[1],
      config::BLOCK_SIZE_CLASS::SMALL, 1, OUTPUT_STEP::STEP_1, outputs);
  Logger::new_line("Step 1 outputs:");
  Logger::new_line(outputs[0].toString());
  Logger::new_line();
  Logger::new_line(outputs[1].toString());
  Logger::new_line();

  const bool output0_equals = outputs[0].equals(ground_truth_0, kMaxAbsError);
  EXPECT_TRUE(output0_equals);
  const bool output1_equals = outputs[1].equals(ground_truth_1, kMaxAbsError);
  EXPECT_TRUE(output1_equals);
}

TEST(GPU_funcs_checkmath_smalltables, step2_arbitrarycoeffs) {
  const size_t n_rows = 5, n_cols = 5;
  const int input_data[] = {
      1, 0, 0, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 0, 1, 1, 0, 1, 1, 0, 1, 2, 1, 2,
  };
  CpuTable input(n_rows, n_cols, &(*input_data));

  const size_t n_gt_rows = (n_rows / get_blockdim_2dgrid_y(
                                         config::BLOCK_SIZE_CLASS::SMALL) +
                            1),
               n_gt_cols = n_cols;
  const float ground_truth_data[] = {
      3.971520747928f,   4.118169868f,      1.454411868f,
      4.398624879928f,   3.030762747928f,   0.35296155699821f,
      2.08899472616396f, 3.57525816331534f, 2.11391964635472f,
      3.71535343066228f,
  };
  CpuTable ground_truth(n_gt_rows, n_gt_cols, &(*ground_truth_data));

  std::vector<CpuTable> outputs;
  outputs.emplace_back(n_gt_rows, n_gt_cols);
  const float runtime_1kernelrun_ms = recursivefilter_downright_gpu(
      input, kArbitratyFilterCoeffs[0], kArbitratyFilterCoeffs[1],
      config::BLOCK_SIZE_CLASS::SMALL, 1, OUTPUT_STEP::STEP_2, outputs);
  Logger::new_line("Step 2 output:");
  Logger::new_line(outputs[0].toString());
  Logger::new_line();

  ASSERT_TRUE(outputs[0].equals(ground_truth, kMaxAbsError));
}

TEST(GPU_funcs_checkmath_smalltables, step3_arbitrarycoeffs) {
  const size_t n_rows = 5, n_cols = 5;
  const int input_data[] = {
      1, 0, 0, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 0, 1, 1, 0, 1, 1, 0, 1, 2, 1, 2,
  };
  CpuTable input(n_rows, n_cols, &(*input_data));

  const size_t n_gt_rows = (n_rows / get_blockdim_2dgrid_y(
                                         config::BLOCK_SIZE_CLASS::SMALL) +
                            1),
               n_gt_cols = (n_cols / config::get_blockdim_2dgrid_x(
                                         config::BLOCK_SIZE_CLASS::SMALL) +
                            1);
  const float ground_truth_data[] = {
      12.1762270487436f,
      5.22200421467995f,
      8.17774632887718f,
      6.40155396103111f,
  };
  CpuTable ground_truth(n_gt_rows, n_gt_cols, &(*ground_truth_data));

  std::vector<CpuTable> outputs;
  outputs.emplace_back(n_gt_rows, n_gt_cols);
  const float runtime_1kernelrun_ms = recursivefilter_downright_gpu(
      input, kArbitratyFilterCoeffs[0], kArbitratyFilterCoeffs[1],
      config::BLOCK_SIZE_CLASS::SMALL, 1, OUTPUT_STEP::STEP_3, outputs);
  Logger::new_line("Step 3 output:");
  Logger::new_line(outputs[0].toString());
  Logger::new_line();

  ASSERT_TRUE(outputs[0].equals(ground_truth, kMaxAbsError));
}

TEST(GPU_funcs_checkmath_smalltables, step4_arbitrarycoeffs) {
  const size_t n_rows = 5, n_cols = 5;
  const int input_data[] = {
      1, 0, 0, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 0, 1, 1, 0, 1, 1, 0, 1, 2, 1, 2,
  };
  CpuTable input(n_rows, n_cols, &(*input_data));

  const size_t n_gt_rows = 5,
               n_gt_cols = (n_cols / config::get_blockdim_2dgrid_x(
                                         config::BLOCK_SIZE_CLASS::SMALL) +
                            1);
  const float ground_truth_data[] = {
      3.45195298555194f, 3.27551493360004f, 8.32589494977925f,
      8.29833254022669f, 14.3575978912594f, 5.40288262578305f,
      12.1762270487436f, 6.30414386099312f, 13.743826651178f,
      10.0101294612827f,
  };
  CpuTable ground_truth(n_gt_rows, n_gt_cols, &(*ground_truth_data));

  std::vector<CpuTable> outputs;
  outputs.emplace_back(n_gt_rows, n_gt_cols);
  const float runtime_1kernelrun_ms = recursivefilter_downright_gpu(
      input, kArbitratyFilterCoeffs[0], kArbitratyFilterCoeffs[1],
      config::BLOCK_SIZE_CLASS::SMALL, 1, OUTPUT_STEP::STEP_4, outputs);
  Logger::new_line("Step 4 output:");
  Logger::new_line(outputs[0].toString());
  Logger::new_line();

  ASSERT_TRUE(outputs[0].equals(ground_truth, kMaxAbsError));
}

TEST(GPU_funcs_checkmath_smalltables, step5_arbitrarycoeffs) {
  const size_t n_rows = 5, n_cols = 5;
  const int input_data[] = {
      1, 0, 0, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 0, 1, 1, 0, 1, 1, 0, 1, 2, 1, 2,
  };
  CpuTable input(n_rows, n_cols, &(*input_data));

  const size_t n_gt_rows = 5, n_gt_cols = 5;
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
  CpuTable ground_truth(n_gt_rows, n_gt_cols, &(*ground_truth_data));

  std::vector<CpuTable> outputs;
  outputs.emplace_back(n_gt_rows, n_gt_cols);
  const float runtime_1kernelrun_ms = recursivefilter_downright_gpu(
      input, kArbitratyFilterCoeffs[0], kArbitratyFilterCoeffs[1],
      config::BLOCK_SIZE_CLASS::SMALL, 1, OUTPUT_STEP::STEP_5, outputs);
  Logger::new_line("Step 5 output:");
  Logger::new_line(outputs[0].toString());
  Logger::new_line();

  ASSERT_TRUE(outputs[0].equals(ground_truth, kMaxAbsError));
}

TEST(GPU_funcs_checkmath_smalltables, authorstinytable_arbitrarycoeffs) {
  const size_t n_rows = 6, n_cols = 6;
  const int input_data[] = {1, 0, 0, 1, 1, 1, 2, 1, 1, 1, 2, 1,
                            1, 2, 1, 2, 0, 1, 1, 1, 0, 1, 1, 2,
                            0, 1, 2, 1, 2, 1, 1, 0, 1, 0, 1, 1};
  CpuTable input(n_rows, n_cols, &(*input_data));
  const comparison_result_t comparison_result =
      recursivefilter_and_compare_gpuvscpu(
          input, kArbitratyFilterCoeffs[0], kArbitratyFilterCoeffs[1],
          config::BLOCK_SIZE_CLASS::SMALL, 1, OUTPUT_STEP::STEP_5,
          kMaxAbsError, true, false);
  ASSERT_TRUE(comparison_result.equals);
}

TEST(GPU_funcs_checkmath_smalltables, oddnumcols_arbitrarycoeffs) {
  const size_t n_rows = 8, n_cols = 5;
  CpuTable input(n_rows, n_cols);
  input.setSawTooth(1.89f, 11);
  const comparison_result_t comparison_result =
      recursivefilter_and_compare_gpuvscpu(
          input, kArbitratyFilterCoeffs[0], kArbitratyFilterCoeffs[1],
          config::BLOCK_SIZE_CLASS::SMALL, 1, OUTPUT_STEP::STEP_5,
          kMaxAbsError, true, false);
  ASSERT_TRUE(comparison_result.equals);
}

TEST(GPU_funcs_checkmath_smalltables, oddnumrows_arbitrarycoeffs) {
  const size_t n_rows = 7, n_cols = 8;
  CpuTable input(n_rows, n_cols);
  input.setSawTooth(1.89f, 11);
  const comparison_result_t comparison_result =
      recursivefilter_and_compare_gpuvscpu(
          input, kArbitratyFilterCoeffs[0], kArbitratyFilterCoeffs[1],
          config::BLOCK_SIZE_CLASS::SMALL, 1, OUTPUT_STEP::STEP_5,
          kMaxAbsError, true, false);
  ASSERT_TRUE(comparison_result.equals);
}

TEST(GPU_funcs_checkmath_smalltables, oddnumcolsnumrows_arbitrarycoeffs) {
  const size_t n_rows = 7, n_cols = 7;
  CpuTable input(n_rows, n_cols);
  input.setSawTooth(1.89f, 11);
  const comparison_result_t comparison_result =
      recursivefilter_and_compare_gpuvscpu(
          input, kArbitratyFilterCoeffs[0], kArbitratyFilterCoeffs[1],
          config::BLOCK_SIZE_CLASS::SMALL, 1, OUTPUT_STEP::STEP_5,
          kMaxAbsError, true, false);
  ASSERT_TRUE(comparison_result.equals);
}

TEST(GPU_funcs_checkmath_largetables, authorsbigtable_arbitrarycoeffs) {
  const size_t n_rows = kAuthorsBigTableSizeY, n_cols = kAuthorsBigTableSizeX;
  CpuTable input(n_rows, n_cols);
  input.setSawTooth(0.14f, 29);
  Logger::new_line("Authors' big table size: (" + std::to_string(n_cols) +
                   ", " + std::to_string(n_rows) + ")");
  const comparison_result_t comparison_result =
      recursivefilter_and_compare_gpuvscpu(
          input, kArbitratyFilterCoeffs[0], kArbitratyFilterCoeffs[1],
          config::BLOCK_SIZE_CLASS::LARGE, 1, OUTPUT_STEP::STEP_5,
          kMaxAbsError, false, false);
  ASSERT_TRUE(comparison_result.equals);
}

TEST(GPU_funcs_checkmath_largetables, bigtable_arbitrarycoeffs) {
  const size_t n_rows = kBigTableSizeY, n_cols = kBigTableSizeX;
  CpuTable input(n_rows, n_cols);
  input.setSawTooth(0.18f, 29);
  Logger::new_line("Big table size: (" + std::to_string(n_cols) + ", " +
                   std::to_string(n_rows) + ")");
  const comparison_result_t comparison_result =
      recursivefilter_and_compare_gpuvscpu(
          input, kArbitratyFilterCoeffs[0], kArbitratyFilterCoeffs[1],
          config::BLOCK_SIZE_CLASS::LARGE, 1, OUTPUT_STEP::STEP_5,
          kMaxAbsError, false, false);
  ASSERT_TRUE(comparison_result.equals);
}

TEST(GPU_funcs_checkmath_largetables, hugetable_fullonesfill_satcoeffs) {
  const size_t n_rows = kBigTableSizeY, n_cols = kBigTableSizeX;
  CpuTable input(n_rows, n_cols, 1.0f);
  Logger::new_line("Huge table size: (" + std::to_string(n_cols) + ", " +
                   std::to_string(n_rows) + ")");
  const comparison_result_t comparison_result =
      recursivefilter_and_compare_gpuvscpu(
          input, kSatFilterCoeffs[0], kSatFilterCoeffs[1],
          config::BLOCK_SIZE_CLASS::LARGE, 1, OUTPUT_STEP::STEP_5,
          kMaxAbsError, false, false);
  ASSERT_TRUE(comparison_result.equals);
}

TEST(GPU_funcs_checkmath_largetables, hugetable_arbitraryfill_satcoeffs) {
  const size_t n_rows = kHugeTableSizeY, n_cols = kHugeTableSizeX;
  CpuTable input(n_rows, n_cols);
  input.setSawTooth(0.18f, 29);
  Logger::new_line("Huge table size: (" + std::to_string(n_cols) + ", " +
                   std::to_string(n_rows) + ")");
  const comparison_result_t comparison_result =
      recursivefilter_and_compare_gpuvscpu(
          input, kSatFilterCoeffs[0], kSatFilterCoeffs[1],
          config::BLOCK_SIZE_CLASS::LARGE, 1, OUTPUT_STEP::STEP_5,
          kMaxAbsError, false, false);
  ASSERT_TRUE(comparison_result.equals);
}

TEST(GPU_funcs_checkmath_largetables, hugetable_arbitraryfill_arbitrarycoeffs) {
  const size_t n_rows = kHugeTableSizeY, n_cols = kHugeTableSizeX;
  CpuTable input(n_rows, n_cols);
  input.setSawTooth(0.18f, 29);
  Logger::new_line("Huge table size: (" + std::to_string(n_cols) + ", " +
                   std::to_string(n_rows) + ")");
  const comparison_result_t comparison_result =
      recursivefilter_and_compare_gpuvscpu(
          input, kArbitratyFilterCoeffs[0], kArbitratyFilterCoeffs[1],
          config::BLOCK_SIZE_CLASS::LARGE, 1, OUTPUT_STEP::STEP_5,
          kMaxAbsError, false, false);
  ASSERT_TRUE(comparison_result.equals);
}