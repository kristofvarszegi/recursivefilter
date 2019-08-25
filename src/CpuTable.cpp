#include "CpuTable.hpp"
#include "Logger.hpp"
#include "utils.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>

namespace gpuacademy {

CpuTable::CpuTable(int num_rows, int num_cols) {
	if (num_rows < 2) {
		throw std::runtime_error("Number of table rows must be at least 2");
	}
	if (num_cols < 2) {
		throw std::runtime_error("Number of table cols must be at least 2");
	}
	resetData(num_rows, num_cols);
}

CpuTable::CpuTable(int num_rows, int num_cols, float fill_val) {
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

CpuTable::CpuTable(int num_rows, int num_cols, const float* data) {
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

CpuTable::CpuTable(int num_rows, int num_cols, const int* data) {
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

CpuTable::~CpuTable() {
}

void CpuTable::set(int row_id, int col_id, float val) {
	if (row_id < 0) {
		throw std::runtime_error("Row ID must be at least 0");
	}
	if (row_id >= data_.size()) {
		throw std::runtime_error("Row ID must be less than the number of rows");
	}
	if (col_id < 0) {
		throw std::runtime_error("Col ID must be at least 0");
	}
	if (col_id >= data_[0].size()) {
		throw std::runtime_error("Col ID must be less than the number of cols");
	}
	data_[row_id][col_id] = val;
}

void CpuTable::add(int row_id, int col_id, float val) {
	if (row_id < 0) {
		throw std::runtime_error("Row ID must be at least 0");
	}
	if (row_id >= data_.size()) {
		throw std::runtime_error("Row ID must be less than the number of rows");
	}
	if (col_id < 0) {
		throw std::runtime_error("Col ID must be at least 0");
	}
	if (col_id >= data_[0].size()) {
		throw std::runtime_error("Col ID must be less than the number of cols");
	}
	data_[row_id][col_id] += val;
}

void CpuTable::set(const float* data) {
	for (int i_row = 0; i_row < num_rows(); ++i_row) {
		for (int i_col = 0; i_col < num_cols(); ++i_col) {
			data_[i_row][i_col] = data[i_col + i_row * num_cols()];
		}
	}
}

void CpuTable::set(const int* data) {
	for (int i_row = 0; i_row < num_rows(); ++i_row) {
		for (int i_col = 0; i_col < num_cols(); ++i_col) {
			set(i_row, i_col, static_cast<float>(data[i_col + i_row * data_[0].size()]));
		}
	}
}

void CpuTable::reset(int num_rows, int num_cols) {
	if (num_rows < 2) {
		throw std::runtime_error("Number of table rows must be at least 2");
	}
	if (num_cols < 2) {
		throw std::runtime_error("Number of table cols must be at least 2");
	}
	resetData(num_rows, num_cols);
}

void CpuTable::resetData(int num_rows, int num_cols) {
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
	for (int i_row = 0; i_row < data_.size(); ++i_row) {
		for (int i_col = 0; i_col < data_[0].size(); ++i_col) {
			const int index_1d = i_col + i_row * data_[0].size();
			if (index_1d % period == 0) {
				val = -amplitude;
			}
			else {
				val += increment;
			}
			data_[i_row][i_col] = val;
		}
	}
}

/*CpuTable& CpuTable::operator=(const CpuTable& other_table) {
	if (this != &other_table) {
		if (data_ != NULL) {
			free(data_);
		}
		num_rows_ = other_table.num_rows();
		num_cols_ = other_table.num_cols();
		data_ = (float*)calloc(num_rows_ * num_cols_, kAllocUnitSize);
		memcpy(...);
	}
	return *this;
}*/

int CpuTable::num_rows() const {
	return data_.size();
}

int CpuTable::num_cols() const {
	return data_[0].size();
}

std::vector<std::vector<float>> CpuTable::data() const {
	return data_;
}

float CpuTable::get(int row_id, int col_id) const {
	if (row_id < 0) {
		throw std::runtime_error("Row ID must be at least 0");
	}
	if (row_id >= data_.size()) {
		throw std::runtime_error("Row ID must be less than the number of rows");
	}
	if (col_id < 0) {
		throw std::runtime_error("Col ID must be at least 0");
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

bool CpuTable::equals(const CpuTable& other_table,
	float max_abserror) const {
	if (num_rows() != other_table.num_rows()) {
		throw std::runtime_error("Number of table rows must match");
	}
	if (num_cols() != other_table.num_cols()) {
		throw std::runtime_error("Number of table cols must match");
	}

	for (int i_row = 0; i_row < num_rows(); ++i_row) {
		for (int i_col = 0; i_col < num_cols(); ++i_col) {
			const float abs_error = abs(data_[i_row][i_col] - other_table.data_[i_row][i_col]);
			//Logger::new_line("(" + std::to_string(i_col) + ", " + std::to_string(i_row) + "): "
			//	+ std::to_string(data_[index])	+ " vs. " + std::to_string(other_table.data_[index])
			//	+ " (" + std::to_string(abs_error) + ")");
			if (abs_error > max_abserror) {
				Logger::new_line("Tables are not equal: " + std::to_string(data_[i_row][i_col])
					+ " vs. " + std::to_string(other_table.data_[i_row][i_col])
				+ " at (" + std::to_string(i_col) + ", " + std::to_string(i_row) + ")"
				+ " (abs error: " + std::to_string(abs_error) + ")");
				Logger::new_line();
				return false;
			}
		}
	}
	return true;
}

}
