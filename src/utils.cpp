#include "utils.hpp"

namespace gpuacademy {

std::string to_ms_str(const clock_t& t) {
	return std::to_string((float)t / (float)CLOCKS_PER_SEC * 1000.0f);
}

}