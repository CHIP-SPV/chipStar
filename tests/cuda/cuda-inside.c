// A source that is intended to be compiled in CUDA mode (-x cu).
#include <cuda_runtime.h>
#ifndef __CUDACC__
#  error "__CUDACC__ is not defined."
#endif
#ifndef __NVCC__
#  error "__NVCC__ is not defined."
#endif
