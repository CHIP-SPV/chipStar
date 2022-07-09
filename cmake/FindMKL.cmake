# Build LLVM-PASSES 
set(MKL_DIR "" CACHE PATH "Path to MKL install part of oneapi")
if(MKL_DIR STREQUAL "") 
  message(FATAL_ERROR "icpx was found, but -DMKL_DIR option was not passed to CMake.")
endif()

# TODO: find MKL based on paths relative to where icpx lives