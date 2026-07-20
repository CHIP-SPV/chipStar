#=============================================================================
#  CMake build system files
#
#  Copyright (c) 2022 chipStar developers
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.
#
#=============================================================================

add_hip_test(TestHiprtcCPPKernels.cc)
add_hip_test(TestHiprtcOptions.cc)
add_hip_test(TestShellMetacharacters.cc)
add_hip_test(TestConstantMemory.cc)
add_hip_test(TestProjectionBasisKernel.cc)
add_hip_test(TestHiprtcCacheNameExpressions.cc)
# This test exercises the HIPRTC SPIR-V cache, so it needs CHIP_MODULE_CACHE_DIR
# set in the environment at libCHIP static-init time. Use a private directory
# under the build tree so the test does not collide with the default cache
# location. The test itself clears any stale `hiprtc/` subdir at startup so
# repeat ctest invocations always start from a known-empty cache.
set_tests_properties(TestHiprtcCacheNameExpressions PROPERTIES
  ENVIRONMENT "CHIP_MODULE_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/test_cache_TestHiprtcCacheNameExpressions")

# TestHiprtcCacheIncludes needs its cache env set before the process starts (see
# tests/run_test_with_fresh_cache.cmake), so instead of add_hip_test's plain
# command we build the executable the same way and register it through the run
# wrapper that provisions a fresh, unique cache directory per run.
add_executable(TestHiprtcCacheIncludes EXCLUDE_FROM_ALL TestHiprtcCacheIncludes.cc)
set_target_properties(TestHiprtcCacheIncludes PROPERTIES CXX_STANDARD_REQUIRED ON)
target_link_libraries(TestHiprtcCacheIncludes CHIP deviceInternal)
target_include_directories(TestHiprtcCacheIncludes
  PUBLIC ${CMAKE_SOURCE_DIR}/HIP/include ${CMAKE_SOURCE_DIR}/include)
add_dependencies(build_tests TestHiprtcCacheIncludes)
add_test(NAME TestHiprtcCacheIncludes
  COMMAND ${CMAKE_COMMAND}
    -DTEST_EXECUTABLE=$<TARGET_FILE:TestHiprtcCacheIncludes>
    -P ${CMAKE_SOURCE_DIR}/tests/run_test_with_fresh_cache.cmake)
set_tests_properties(TestHiprtcCacheIncludes PROPERTIES
  SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST")
