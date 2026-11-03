# Check hip::deviceRDC does not make clang warn about differing data layouts (#525).
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(HipDeviceRDCDataLayout
  DESCRIPTION "Checking hip::deviceRDC link output."
  LANGUAGES CXX)
find_package(HIP CONFIG REQUIRED)
add_executable(foo main.cpp)
target_link_libraries(foo PRIVATE hip::deviceRDC)
