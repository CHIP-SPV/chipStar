#include <hip/hip_runtime.h>

// Regression test for llvm-patches/spirv-translator/0005
// (coalesce duplicate-predecessor OpPhi).
// This results in LLVM IR produced to have a phi node with
// duplicate entries (for at least LLVM22) which then
// produces invalid SPIRV since SPIRV does not allow dupilcate entries. 

__device__ int grad(int sel) {
  int v = 7;
  switch (sel) {
    case 0:
    case 4:
    case 10:
      return v;      // 3 cases share one return -> switch block has 3 edges to the merge
    default:
      v = 9;
  }
  return v;
}

__global__ __attribute__((optnone))
void k(int *out, int sel) { out[0] = grad(sel); }
