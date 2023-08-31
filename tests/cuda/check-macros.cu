#include <cuda_runtime.h>
#ifndef __CUDACC__
#  error "__CUDACC__ is not defined."
#endif
#ifndef __NVCC__
#  error "__NVCC__ is not defined."
#endif

#ifndef FOO
#  error "FOO was not defined!"
#endif

#ifndef BAR
#  error "BAR was not defined!"
#endif

#if FOO != 123
#  error "Expected FOO == 123!"
#endif

#if BAR != 321
#  error "Expected BAR == 321!"
#endif
