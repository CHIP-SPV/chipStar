# Check hip::host target dependence. Sources depending on it should compile
# with any C++ compiler.
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(HipHostTrial
  DESCRIPTION "Check hip::host target dependence"
  LANGUAGES CXX)
find_package(HIP CONFIG REQUIRED)
add_executable(foo foo.cpp)
target_link_libraries(foo PRIVATE hip::host)
