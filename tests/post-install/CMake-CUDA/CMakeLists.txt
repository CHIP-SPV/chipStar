cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

# This must be defined (for now). Otherwise, CMake aborts with:
# "Failed to detect a default CUDA architecture" error.
set(CMAKE_CUDA_ARCHITECTURES "72")

project(MyCudaProject LANGUAGES CUDA CXX C)

add_library(mylib SHARED main.cu some.cu some.cpp some.c)
target_link_options(mylib PRIVATE -Wl,--no-undefined)

add_executable(myexe main.cu some.cu some.cpp some.c)
target_link_options(myexe PRIVATE -Wl,--no-undefined)

message(STATUS "CMAKE_CUDA_COMPILER_TOOLKIT_ROOT: ${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}")
message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CUDA_COMPILER_ID: ${CMAKE_CUDA_COMPILER_ID}")
