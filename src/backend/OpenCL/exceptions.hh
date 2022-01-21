#ifndef OPENCL_EXCEPTIONS_H
#define OPENCL_EXCEPTIONS_H

#include <stdexcept>

class InvalidDeviceType : public std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};

class InvalidPlatformOrDeviceNumber : public std::out_of_range {
  using std::out_of_range::out_of_range;
};
#endif