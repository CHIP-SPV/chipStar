cmake_minimum_required(VERSION 3.10)
project(opencl-spirv-compiler)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCL package
find_package(OpenCL REQUIRED)

# Add executable
add_executable(opencl-spirv-compiler opencl-spirv-compiler.cpp)

# Link OpenCL library
target_link_libraries(opencl-spirv-compiler PRIVATE OpenCL::OpenCL)

# Add filesystem library for GCC versions < 9
if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(opencl-spirv-compiler PRIVATE stdc++fs)
endif()
