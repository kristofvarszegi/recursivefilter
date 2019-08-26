#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <string>

namespace gpuacademy {

class Logger {
public:
  static void new_line(const std::string &message);
  static void new_line();
  static void in_line(const std::string &message);

private:
};

} // namespace gpuacademy
#endif
