#ifndef CPUFUNCS_HPP
#define CPUFUNCS_HPP

#include "CpuTable.hpp"

namespace gpuacademy {
void calculate_summedareatable_cpu_naive(const CpuTable& input, CpuTable& output);
void recursivefilter_downright_cpu(const CpuTable& input,
	float filter_coeff_0, float filter_coeff_1, CpuTable& output);
}
#endif
