// __builtin_return_address emits llvm.returnaddress, which llvm-spirv rejects.
#include <hip/hip_runtime.h>

__global__ void k(void **Out) { Out[0] = __builtin_return_address(0); }
