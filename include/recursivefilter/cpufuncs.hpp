#ifndef CPUFUNCS_HPP
#define CPUFUNCS_HPP

#include "CpuTable.hpp"

namespace gpuacademy {
void calculate_summedareatable_cpu_naive(const CpuTable &input,
                                         CpuTable &output);
void recursivefilter_downright_cpu(const CpuTable &input, float feedfwd_coeff,
                                   float feedback_coeff, CpuTable &output);
} // namespace gpuacademy
#endif
