#include "helpers.hpp"

#include "gpufuncs.hpp"
#include "cpufuncs.hpp"
#include "utils.hpp"
#include "Logger.hpp"

namespace gpuacademy {

const float kEpsilon = 0.000001f;
const float kSatFilterCoeffs[] = { 1.0f, 1.0f };

int apply_recursivefilter_gpu_and_compare_with_cpu(const CpuTable& input,
	float filter_coeff_0, float filter_coeff_1, bool print_tables) {
	CpuTable summed_area_table(input.num_rows(), input.num_cols());
	apply_right_down_recursive_filter_gpu(input, filter_coeff_0, filter_coeff_1,
		summed_area_table);

	CpuTable ground_truth(input.num_rows(), input.num_cols());
	Logger::new_line("\nCalculating SAT CPU for reference...");
	apply_right_down_recursive_filter_cpu(input, filter_coeff_0, filter_coeff_1,
		ground_truth);
	if (print_tables) {
		Logger::new_line();
		Logger::new_line(input.toString());
		Logger::new_line(ground_truth.toString());
		Logger::new_line(summed_area_table.toString());
		Logger::new_line();
	}

	int ret = -1;
	if (ground_truth.equals(summed_area_table, kEpsilon)) {
		ret = 0;
	}
	return ret;
}

}