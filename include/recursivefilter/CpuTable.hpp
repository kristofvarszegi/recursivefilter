#ifndef CPUTABLE_HPP
#define CPUTABLE_HPP

#include <string>
#include <vector>

namespace gpuacademy {

class CpuTable {
public:
  CpuTable(int num_rows, int num_cols);
  CpuTable(int num_rows, int num_cols, float fill_val);
  CpuTable(int num_rows, int num_cols, const float *data);
  CpuTable(int num_rows, int num_cols, const int *data);
  ~CpuTable();
  void set(int row_id, int col_id, float val);
  void add(int row_id, int col_id, float val);
  void set(const float *data);
  void set(const int *data);
  void reset(int num_rows, int num_cols);
  void reset(int num_rows, int num_cols, const float *data);
  void setSawTooth(float amplitude, int period);
  void transpose();
  int num_rows() const;
  int num_cols() const;
  std::vector<std::vector<float>> data() const;
  float get(int row_id, int col_id) const;
  std::string toString() const;
  bool equals(const CpuTable &other_table, float max_abserror) const;

private:
  std::vector<std::vector<float>> data_;
  void resetData(int num_rows, int num_cols);
};

} // namespace gpuacademy
#endif
