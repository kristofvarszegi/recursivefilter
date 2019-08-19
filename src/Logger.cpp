#include "Logger.hpp"

#include <iostream>

namespace gpuacademy {

void Logger::new_line(const std::string& message) {
	std::cout << std::endl << message;
}

void Logger::new_line() {
	std::cout << std::endl;
}

void Logger::in_line(const std::string& message) {
	std::cout << message;
}

}