#include <gtest/gtest.h>

#include "helpers.hpp"

#include "CpuTable.hpp"
#include "Logger.hpp"
#include "cpufuncs.hpp"
#include "gpufuncs.hpp"
#include "utils.hpp"

#include <opencv2/imgcodecs.hpp>

#include <vector>

//#define TEST_RESOURCES_PATH_CPP @TEST_RESOURCES_PATH@
//#define TEST_RESOURCES_PATH_CPP "res"

using namespace gpuacademy;

const float kMaxAbsError = 0.005f;
const float kSatFeedFwdCoeff = 1.0f, kSatFeedBackCoeff = 1.0f;

void opencvimage2cputable(const cv::Mat &opencv_image, CpuTable &table) {
  if (opencv_image.rows != table.num_rows()) {
    throw std::runtime_error("Number of rows must match");
  }
  if (opencv_image.cols != table.num_cols()) {
    throw std::runtime_error("Number of cols must match");
  }
  for (int i_row = 0; i_row < opencv_image.rows; ++i_row) {
    for (int i_col = 0; i_col < opencv_image.cols; ++i_col) {
      table.set(i_row, i_col,
                static_cast<float>(opencv_image.at<uchar>(i_row, i_col)));
    }
  }
}

void cputable2opencvimage(const CpuTable &table, cv::Mat &opencv_image) {
  if (opencv_image.rows != table.num_rows()) {
    throw std::runtime_error("Number of rows must match");
  }
  if (opencv_image.cols != table.num_cols()) {
    throw std::runtime_error("Number of cols must match");
  }
  for (int i_row = 0; i_row < opencv_image.rows; ++i_row) {
    for (int i_col = 0; i_col < opencv_image.cols; ++i_col) {
      opencv_image.at<uchar>(i_row, i_col) =
          static_cast<uchar>(table.get(i_row, i_col));
    }
  }
}

TEST(GPU_funcs_filterimages, all0x010101) {
  // const std::string& input_image_path = std::string(TEST_RESOURCES_PATH_CPP)
  // + std::string("/all0x010101.png");
  const std::string &input_image_path("all0x010101.png");
  Logger::new_line("Loading image \"" + input_image_path + "\"...");
  cv::Mat input_image = cv::imread(input_image_path, cv::IMREAD_GRAYSCALE);
  if (input_image.empty()) {
    throw std::runtime_error("Failed to open image \"" + input_image_path +
                             "\"");
  }
  Logger::new_line("Image size: " + std::to_string(input_image.cols) + "x" +
                   std::to_string(input_image.rows));
  if (input_image.channels() != 1) {
    throw std::runtime_error("Image must be grayscale");
  }

  CpuTable input(input_image.rows, input_image.cols);
  opencvimage2cputable(input_image, input);

  CpuTable ground_truth(input.num_rows(), input.num_cols());
  recursivefilter_downright_cpu(input, kSatFeedFwdCoeff, kSatFeedBackCoeff,
                                ground_truth);

  std::vector<CpuTable> outputs;
  outputs.emplace_back(input.num_rows(), input.num_cols());
  recursivefilter_downright_gpu<config::kBlockDim2dGridLarge,
                                config::kBlockDim1dGridLarge,
                                config::kNumKernelRunsFew>(
      input, kSatFeedFwdCoeff, kSatFeedBackCoeff, OUTPUT_STEP::STEP_5, outputs);
  Logger::new_line();
  Logger::new_line();

  cv::Mat output_image = cv::Mat::zeros(input_image.size(), CV_8UC1);
  cputable2opencvimage(outputs[0], output_image);
  const std::string &output_image_path = "all0x010101_sat.png";
  cv::imwrite(output_image_path, output_image);

  const bool output_equals = outputs[0].equals(ground_truth, kMaxAbsError);
  ASSERT_TRUE(output_equals);
}

TEST(GPU_funcs_filterimages, arbitraryimage) {
  // const std::string& input_image_path = std::string(TEST_RESOURCES_PATH_CPP)
  // + std::string("/all0x010101.png");
  const std::string &input_image_path("arbitrary.png");
  Logger::new_line("Loading image \"" + input_image_path + "\"...");
  cv::Mat input_image = cv::imread(input_image_path, cv::IMREAD_GRAYSCALE);
  if (input_image.empty()) {
    throw std::runtime_error("Failed to open image \"" + input_image_path +
                             "\"");
  }
  Logger::new_line("Image size: " + std::to_string(input_image.cols) + "x" +
                   std::to_string(input_image.rows));
  if (input_image.channels() != 1) {
    throw std::runtime_error("Image must be grayscale");
  }

  CpuTable input(input_image.rows, input_image.cols);
  opencvimage2cputable(input_image, input);

  CpuTable ground_truth(input.num_rows(), input.num_cols());
  recursivefilter_downright_cpu(input, kSatFeedFwdCoeff, kSatFeedBackCoeff,
                                ground_truth);

  std::vector<CpuTable> outputs;
  outputs.emplace_back(input.num_rows(), input.num_cols());
  recursivefilter_downright_gpu<config::kBlockDim2dGridLarge,
                                config::kBlockDim1dGridLarge,
                                config::kNumKernelRunsFew>(
      input, kSatFeedFwdCoeff, kSatFeedBackCoeff, OUTPUT_STEP::STEP_5, outputs);
  Logger::new_line();
  Logger::new_line();

  cv::Mat output_image = cv::Mat::zeros(input_image.size(), CV_8UC1);
  cputable2opencvimage(outputs[0], output_image);
  const std::string &output_image_path = "arbitrary_sat.png";
  cv::imwrite(output_image_path, output_image);

  const bool output_equals = outputs[0].equals(ground_truth, kMaxAbsError);
  ASSERT_TRUE(output_equals);
}