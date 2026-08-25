// __builtin_flt_rounds, which is what the C FLT_ROUNDS macro expands to,
// emits llvm.get.rounding. Neither SPIR-V producer accepts it. An OpenCL
// device only supports round to nearest even and clang rejects the matching
// setter __builtin_set_flt_rounds on spirv64, so HipLowerHintIntrinsicsPass
// folds the getter to the FLT_ROUNDS encoding of that mode, 1.
#include <hip/hip_runtime.h>

__global__ void k(int *Out) { Out[0] = __builtin_flt_rounds(); }
