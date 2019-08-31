#include "CpuTable.hpp"
#include "Logger.hpp"
#include "utils.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace gpuacademy {

CpuTable::CpuTable(size_t num_rows, size_t num_cols) {
  if (num_rows < 2) {
    throw std::runtime_error("Number of table rows must be at least 2");
  }
  if (num_cols < 2) {
    throw std::runtime_error("Number of table cols must be at least 2");
  }
  resetData(num_rows, num_cols);
}

CpuTable::CpuTable(size_t num_rows, size_t num_cols, float fill_val) {
  if (num_rows < 2) {
    throw std::runtime_error("Number of table rows must be at least 2");
  }
  if (num_cols < 2) {
    throw std::runtime_error("Number of table cols must be at least 2");
  }
  resetData(num_rows, num_cols);
  for (int i_row = 0; i_row < num_rows; ++i_row) {
    for (int i_col = 0; i_col < num_cols; ++i_col) {
      data_[i_row][i_col] = fill_val;
    }
  }
}

CpuTable::CpuTable(size_t num_rows, size_t num_cols, const float *data) {
  if (num_rows < 2) {
    throw std::runtime_error("Number of table rows must be at least 2");
  }
  if (num_cols < 2) {
    throw std::runtime_error("Number of table cols must be at least 2");
  }
  if (data == NULL) {
    throw std::runtime_error("data must not be NULL");
  }
  resetData(num_rows, num_cols);
  set(data);
}

CpuTable::CpuTable(size_t num_rows, size_t num_cols, const int *data) {
  if (num_rows < 2) {
    throw std::runtime_error("Number of table rows must be at least 2");
  }
  if (num_cols < 2) {
    throw std::runtime_error("Number of table cols must be at least 2");
  }
  if (data == NULL) {
    throw std::runtime_error("data must not be NULL");
  }
  resetData(num_rows, num_cols);
  set(data);
}

CpuTable::~CpuTable() {}

void CpuTable::set(size_t row_id, size_t col_id, float val) {
  if (row_id >= data_.size()) {
    throw std::runtime_error("Row ID must be less than the number of rows");
  }
  if (col_id >= data_[0].size()) {
    throw std::runtime_error("Col ID must be less than the number of cols");
  }
  data_[row_id][col_id] = val;
}

void CpuTable::add(size_t row_id, size_t col_id, float val) {
  if (row_id >= data_.size()) {
    throw std::runtime_error("Row ID must be less than the number of rows");
  }
  if (col_id >= data_[0].size()) {
    throw std::runtime_error("Col ID must be less than the number of cols");
  }
  data_[row_id][col_id] += val;
}

void CpuTable::set(const float *data) {
  for (int i_row = 0; i_row < num_rows(); ++i_row) {
    for (int i_col = 0; i_col < num_cols(); ++i_col) {
      data_[i_row][i_col] = data[i_col + i_row * num_cols()];
    }
  }
}

void CpuTable::set(const int *data) {
  for (int i_row = 0; i_row < num_rows(); ++i_row) {
    for (int i_col = 0; i_col < num_cols(); ++i_col) {
      set(i_row, i_col,
          static_cast<float>(data[i_col + i_row * data_[0].size()]));
    }
  }
}

void CpuTable::reset(size_t num_rows, size_t num_cols) {
  if (num_rows < 2) {
    throw std::runtime_error("Number of table rows must be at least 2");
  }
  if (num_cols < 2) {
    throw std::runtime_error("Number of table cols must be at least 2");
  }
  resetData(num_rows, num_cols);
}

void CpuTable::reset(size_t num_rows, size_t num_cols, const float *data) {
  if (num_rows < 2) {
    throw std::runtime_error("Number of table rows must be at least 2");
  }
  if (num_cols < 2) {
    throw std::runtime_error("Number of table cols must be at least 2");
  }
  resetData(num_rows, num_cols);
  set(data);
}

void CpuTable::resetData(size_t num_rows, size_t num_cols) {
  if (num_rows < 2) {
    throw std::runtime_error("Number of table rows must be at least 2");
  }
  if (num_cols < 2) {
    throw std::runtime_error("Number of table cols must be at least 2");
  }
  data_ = std::vector<std::vector<float>>();
  for (int i_row = 0; i_row < num_rows; ++i_row) {
    data_.emplace_back(num_cols);
  }
}

void CpuTable::setSawTooth(float amplitude, int period) {
  float val = -amplitude;
  const float increment = 2 * amplitude / static_cast<float>(period - 1);
  for (size_t i_row = size_t(0); i_row < data_.size(); ++i_row) {
    for (size_t i_col = size_t(0); i_col < data_[0].size(); ++i_col) {
      const size_t index_1d = i_col + i_row * data_[0].size();
      if (index_1d % period == 0) {
        val = -amplitude;
      } else {
        val += increment;
      }
      data_[i_row][i_col] = val;
    }
  }
}

void CpuTable::transpose() {
  std::vector<std::vector<float>> new_data = std::vector<std::vector<float>>();
  for (int i_col = 0; i_col < data_[0].size(); ++i_col) {
    new_data.emplace_back(data_.size());
    for (int i_row = 0; i_row < data_.size(); ++i_row) {
      new_data[i_col][i_row] = data_[i_row][i_col];
    }
  }
  data_ = new_data;
}

size_t CpuTable::num_rows() const { return data_.size(); }

size_t CpuTable::num_cols() const { return data_[0].size(); }

std::vector<std::vector<float>> CpuTable::data() const { return data_; }

float CpuTable::get(size_t row_id, size_t col_id) const {
  if (row_id >= data_.size()) {
    throw std::runtime_error("Row ID must be less than the number of rows");
  }
  if (col_id >= data_[0].size()) {
    throw std::runtime_error("Col ID must be less than the number of cols");
  }
  return data_[row_id][col_id];
}

std::string CpuTable::toString() const {
  std::stringstream ss;
  for (int i_row = 0; i_row < data_.size(); ++i_row) {
    ss << std::endl;
    for (int i_col = 0; i_col < data_[0].size(); ++i_col) {
      ss << std::setw(9) << data_[i_row][i_col];
    }
  }
  return ss.str();
}

bool CpuTable::equals(const CpuTable &other_table, float max_abserror) const {
  if (num_rows() != other_table.num_rows()) {
    throw std::runtime_error("Number of table rows must match");
  }
  if (num_cols() != other_table.num_cols()) {
    throw std::runtime_error("Number of table cols must match");
  }

  for (int i_row = 0; i_row < num_rows(); ++i_row) {
    for (int i_col = 0; i_col < num_cols(); ++i_col) {
      const float abs_error =
          abs(data_[i_row][i_col] - other_table.data_[i_row][i_col]);
      // Logger::new_line("(" + std::to_string(i_col) + ", " +
      // std::to_string(i_row) + "): "
      //	+ std::to_string(data_[index])	+ " vs. " +
      // std::to_string(other_table.data_[index])
      //	+ " (" + std::to_string(abs_error) + ")");
      if (abs_error > max_abserror) {
        Logger::new_line(
            "Tables are not equal: " + std::to_string(data_[i_row][i_col]) +
            " vs. " + std::to_string(other_table.data_[i_row][i_col]) +
            " at (" + std::to_string(i_col) + ", " + std::to_string(i_row) +
            ")" + " (abs error: " + std::to_string(abs_error) + ")");
        Logger::new_line();
        return false;
      }
    }
  }
  return true;
}

void CpuTable::saveToCsv(const std::string &filename) const {
  std::ofstream table_file(filename);
  for (int i_row = 0; i_row < num_rows(); ++i_row) {
    for (int i_col = 0; i_col < num_cols(); ++i_col) {
      table_file << get(i_row, i_col) << ",";
    }
    table_file << std::endl;
  }
  table_file.close();
}

} // namespace gpuacademy
