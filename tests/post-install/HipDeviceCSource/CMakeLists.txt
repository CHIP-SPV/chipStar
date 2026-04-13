# Regression test for #1001: C source files in a CMake target that links
# against hip::device must not receive -x hip / -xhip compilation flags.
# With the bug, helper.c (which uses 'class' as a C identifier) fails to
# compile because it is treated as C++ / HIP source.
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(HipDeviceCSourceTrial
  DESCRIPTION "Checking C sources are not compiled as HIP when using hip::device."
  LANGUAGES C CXX)
find_package(HIP CONFIG REQUIRED)

# A mixed C+HIP executable that links against hip::device.
add_executable(hip_c_source main.cpp helper.c)
target_link_libraries(hip_c_source PRIVATE hip::device)
