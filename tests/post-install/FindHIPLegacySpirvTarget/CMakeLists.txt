cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(FindHIPLegacySpirvTarget LANGUAGES CXX)
find_package(HIP REQUIRED)
# Force compilation through run_hipcc.cmake (same as SeisSol's Device submodule)
set_source_files_properties(foo.hip PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
# SHARED triggers --shared -Xcompiler '-fPIC' in HIP_NVCC_FLAGS, which clang
# cannot understand when run_hipcc.cmake falls through to the nvcc branch
hip_add_library(foo SHARED foo.hip)
set_property(TARGET foo PROPERTY HIP_ARCHITECTURES OFF)
