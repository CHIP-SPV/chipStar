// __builtin_frame_address emits llvm.frameaddress, which llvm-spirv rejects.
#include <hip/hip_runtime.h>

__global__ void k(void **Out) { Out[0] = __builtin_frame_address(0); }
