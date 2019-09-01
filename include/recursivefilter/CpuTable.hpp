#ifndef CPUTABLE_HPP
#define CPUTABLE_HPP

#include <string>
#include <vector>

namespace gpuacademy {

class CpuTable {
public:
  CpuTable(size_t num_rows, size_t num_cols);
  CpuTable(size_t num_rows, size_t num_cols, float fill_val);
  CpuTable(size_t num_rows, size_t num_cols, const float *data);
  CpuTable(size_t num_rows, size_t num_cols, const int *data);
  CpuTable(size_t num_rows, size_t num_cols, size_t alignment_floats, size_t blockdim, const float *data);
  ~CpuTable();
  void set(size_t row_id, size_t col_id, float val);
  void add(size_t row_id, size_t col_id, float val);
  void set(const float *data);
  void set(const int *data);
  void set(size_t alignment_floats, size_t blockdim, const float *data);
  void reset(size_t num_rows, size_t num_cols);
  void reset(size_t num_rows, size_t num_cols, const float *data);
  void reset(size_t num_rows, size_t num_cols, size_t alignment_floats, size_t blockdim, const float *data);
  void setSawTooth(float amplitude, int period);
  void transpose();
  size_t num_rows() const;
  size_t num_cols() const;
  std::vector<std::vector<float>> data() const;
  float get(size_t row_id, size_t col_id) const;
  std::string toString() const;
  bool equals(const CpuTable &other_table, float max_abserror) const;
  void saveToCsv(const std::string &filename) const;

private:
  std::vector<std::vector<float>> data_;
  void resetData(size_t num_rows, size_t num_cols);
};

} // namespace gpuacademy
#endif
