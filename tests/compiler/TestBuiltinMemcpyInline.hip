// __builtin_memcpy_inline emits llvm.memcpy.inline, which the llvm-spirv
// translator rejects and which the in-tree SPIR-V backend fails to legalize
// for the generic address space clang emits here. chipStar already expands
// llvm.memset.inline, because MemSetInst::classof covers it, but nothing in
// the pipeline touched the memcpy form. HipLowerMemIntrinsicsPass expands it
// to a loop, which keeps the LangRef guarantee that no external function is
// called.
#include <hip/hip_runtime.h>

__global__ void k(char *D, const char *S) {
  __builtin_memcpy_inline(D, S, 16);
}
