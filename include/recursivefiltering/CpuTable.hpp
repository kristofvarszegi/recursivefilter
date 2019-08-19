#pragma once

#include <string>

namespace gpuacademy {

class CpuTable {
public:
	CpuTable(int num_rows, int num_cols);
	CpuTable(int num_rows, int num_cols, float fill_val);
	CpuTable(int num_rows, int num_cols, float* data);
	~CpuTable();
	void set(int row_id, int col_id, float val);
	void add(int row_id, int col_id, float val);
	void set(const float* data);
	//CpuTable& operator=(const CpuTable& other_table);
	int num_rows() const;
	int num_cols() const;
	float* data() const;
	float get(int row_id, int col_id) const;
	std::string toString() const;
	bool equals(const CpuTable& other_table, float epsilon) const;
private:
	int num_rows_;
	int num_cols_;
	float* data_;
};

}