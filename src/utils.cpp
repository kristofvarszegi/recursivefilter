#include "utils.hpp"

namespace gpuacademy {

float to_ms(const clock_t& t) {
	return (float)t / (float)CLOCKS_PER_SEC * 1000.0f;
}

std::string to_ms_str(const clock_t& t) {
	return std::to_string(to_ms(t));
}

}