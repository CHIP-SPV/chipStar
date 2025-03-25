cmake_minimum_required(VERSION 3.15)
project(shimCPURepro LANGUAGES CXX)

# Add the path to the chipStar HIP installation
set(HIP_PATH /space/pvelesko/chipStar/main)
set(CMAKE_PREFIX_PATH ${HIP_PATH} ${CMAKE_PREFIX_PATH})

# Find HIP package
find_package(HIP REQUIRED)

# Create the executable
add_executable(shimCPURepro backend_mismatch_reproducer.cpp)

# Include the HIP directories
target_include_directories(shimCPURepro PRIVATE
    ${HIP_PATH}/include
)

# Link against HIP
target_link_libraries(shimCPURepro PRIVATE 
    hip::host
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) 