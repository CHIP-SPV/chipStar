// __builtin_return_address emits llvm.returnaddress, which no SPIR-V producer
// chipStar uses accepts. A SPIR-V kernel has no addressable call frame and
// LangRef allows the intrinsic to yield zero when the address cannot be
// identified, so HipLowerHintIntrinsicsPass folds it to a null pointer.
#include <hip/hip_runtime.h>

__global__ void k(void **Out) { Out[0] = __builtin_return_address(0); }
