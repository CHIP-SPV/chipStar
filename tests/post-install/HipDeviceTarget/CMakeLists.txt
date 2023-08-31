# Check C++ application depending on hip::device.
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(HipHostTrial
  DESCRIPTION "Checking hip::device dependence."
  LANGUAGES CXX)
find_package(HIP CONFIG REQUIRED)
add_executable(foo foo.cpp)
target_link_libraries(foo PRIVATE hip::device)
