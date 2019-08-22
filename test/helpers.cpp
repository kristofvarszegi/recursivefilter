#include "helpers.hpp"

#include "gpufuncs.hpp"
#include "cpufuncs.hpp"
#include "utils.hpp"
#include "Logger.hpp"

namespace gpuacademy {

const float kEpsilon = 0.000001f;
const float kSatFilterCoeffs[] = { 1.0f, 1.0f };

template <int tableblockdim_x, int tableblockdim_y>
comparison_result_t apply_recursivefilter_gpu_and_compare_with_cpu(const CpuTable& input,
	float filter_coeff_0, float filter_coeff_1, int num_kernel_runs,
	bool print_tables) {
	CpuTable summed_area_table(input.num_rows(), input.num_cols());
	const float runtime_1kernelrun_ms = recursivefilter_downright_gpu
		<tableblockdim_x, tableblockdim_y>
			(input, filter_coeff_0, filter_coeff_1, num_kernel_runs, summed_area_table);

	CpuTable ground_truth(input.num_rows(), input.num_cols());
	Logger::new_line("\nCalculating SAT CPU for reference...");
	recursivefilter_downright_cpu(input, filter_coeff_0, filter_coeff_1,
		ground_truth);
	if (print_tables) {
		Logger::new_line();
		Logger::new_line(input.toString());
		Logger::new_line(ground_truth.toString());
		Logger::new_line(summed_area_table.toString());
		Logger::new_line();
		Logger::new_line();
	}

	comparison_result_t comparison_result;
	comparison_result.equals = ground_truth.equals(summed_area_table, kEpsilon);
	comparison_result.runtime_1kernel_ms = runtime_1kernelrun_ms;
	return comparison_result;
}

}