#include "utils.hpp"

#include <sstream>

namespace gpuacademy {

std::string to_string(const float* data, int num_rows, int num_cols) {
	std::stringstream ss;
	ss << std::endl;
	for (int i_row = 0; i_row < num_rows; ++i_row) {
		for (int i_col = 0; i_col < num_cols; ++i_col) {
			ss << std::to_string(data[i_col + i_row * num_cols]) << ", ";
		}
		ss << std::endl;
	}
	return ss.str();
}

float to_ms(const clock_t& t) {
	return (float)t / (float)CLOCKS_PER_SEC * 1000.0f;
}

std::string to_ms_str(const clock_t& t) {
	return std::to_string(to_ms(t));
}

}
