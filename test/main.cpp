#include <gtest/gtest.h>
#include "Logger.hpp"

using namespace gpuacademy;

int main(int argc, char* argv[]) {
	Logger::new_line("sizeof(float): " + std::to_string(sizeof(float)));
	Logger::new_line("sizeof(int): " + std::to_string(sizeof(int)));
	Logger::new_line();

	//testing::GTEST_FLAG(filter) = "-GPU_funcs.*";
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
