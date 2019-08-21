#include "CpuTable.hpp"
#include "Logger.hpp"

#include <sstream>
#include <iomanip>

namespace gpuacademy {

CpuTable::CpuTable(int num_rows, int num_cols) : num_rows_(num_rows), num_cols_(num_cols){
	data_ = (float*)calloc(num_rows * num_cols, sizeof(float));
}

CpuTable::CpuTable(int num_rows, int num_cols, float fill_val) : num_rows_(num_rows), num_cols_(num_cols) {
	data_ = (float*)calloc(num_rows * num_cols, sizeof(float));
	for (int i_row = 0; i_row < num_rows_; ++i_row) {
		for (int i_col = 0; i_col < num_cols_; ++i_col) {
			data_[i_col + i_row * num_cols_] = fill_val;
		}
	}
}

CpuTable::CpuTable(int num_rows, int num_cols, float* data) : num_rows_(num_rows), num_cols_(num_cols) {
	if (data == NULL) {
		throw std::runtime_error("data must not be NULL");
	}
	data_ = (float*)calloc(num_rows * num_cols, sizeof(float));
	memcpy(data_, data, num_rows_ * num_cols_ * sizeof(float));
}

CpuTable::~CpuTable() {
	free(data_);
}

void CpuTable::set(int row_id, int col_id, float val) {
	data_[col_id + row_id * num_cols_] = val;
}

void CpuTable::add(int row_id, int col_id, float val) {
	data_[col_id + row_id * num_cols_] += val;
}

void CpuTable::set(const float* data) {
	memcpy(data_, data, num_rows_ * num_cols_ * sizeof(float));
}

void CpuTable::setIncreasing(float first_value, float increment) {
	float val = first_value;
	for (int i_row = 0; i_row < num_rows_; ++i_row) {
		for (int i_col = 0; i_col < num_cols_; ++i_col) {
			set(i_row, i_col, val);
			val += increment;
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
		data_ = (float*)calloc(num_rows_ * num_cols_, sizeof(float));
		memcpy(...);
	}
	return *this;
}*/

int CpuTable::num_rows() const {
	return num_rows_;
}

int CpuTable::num_cols() const {
	return num_cols_;
}

float* CpuTable::data() const {
	return data_;
}

float CpuTable::get(int row_id, int col_id) const {
	return data_[col_id + row_id * num_cols_];
}

std::string CpuTable::toString() const {
	std::stringstream ss;
	for (int i_row = 0; i_row < num_rows_; ++i_row) {
		ss << std::endl;
		for (int i_col = 0; i_col < num_cols_; ++i_col) {
			ss << std::setw(9) << data_[i_col + i_row * num_cols_];
		}
	}
	return ss.str();
}

bool CpuTable::equals(const CpuTable& other_table, float epsilon) const {
	if (num_rows_ != other_table.num_rows()) {
		throw std::runtime_error("Number of table rows must match");
	}
	if (num_cols_ != other_table.num_cols()) {
		throw std::runtime_error("Number of table cols must match");
	}

	for (int i_row = 0; i_row < num_rows_; ++i_row) {
		for (int i_col = 0; i_col < num_cols_; ++i_col) {
			const int index = i_col + i_row * num_cols_;
			if (abs(data_[index] - other_table.data_[index]) > epsilon) {
				return false;
			}
		}
	}
	return true;
}

}