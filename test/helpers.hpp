#pragma once

#include "CpuTable.hpp"

namespace gpuacademy {

extern const float kEpsilon;
extern const float kSatFilterCoeffs[];

int apply_recursivefilter_gpu_and_compare_with_cpu(const CpuTable& input,
	float filter_coeff_0, float filter_coeff_1, bool print_tables);
}
