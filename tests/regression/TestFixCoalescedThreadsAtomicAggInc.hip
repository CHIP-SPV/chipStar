// Reproduces: cooperative_groups::coalesced_threads() relies on the AMDGCN
// exec register (__builtin_amdgcn_read_exec) to identify active lanes in
// the current warp. chipStar's SPIR-V backend has no way to query the
// active-lane mask, and the existing stub returns 0 -> empty group with
// size()==0. Any warp-aggregated pattern that uses g.size() then degenerates
// silently:
//
//   __device__ int atomicAggInc(int *ctr) {
//     auto g = cooperative_groups::coalesced_threads();
//     // g.size() returns 0 before the fix
//     return atomicAdd(ctr, (int)g.size());
//   }
//
// Without the fix every thread's atomicAdd contributes 0, so the global
// counter never advances. The fix is to model each thread as its own
// coalesced group of size 1, which keeps semantics correct (every thread
// performs an atomicAdd(ctr, 1)) at the cost of warp-level aggregation.
//
// Surfaced by HeCBench filter-hip.

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;

__global__ void agg_count(int *ctr, int n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= n) return;
  cg::coalesced_group g = cg::coalesced_threads();
  // Use only .size() and .thread_rank() — both must be >= 1 for an
  // active lane. The buggy stub makes both 0 because the underlying
  // member_mask is 0.
  if (g.thread_rank() == 0)
    atomicAdd(ctr, (int)g.size());
}

int main() {
  const int N = 1024;
  int *d_ctr;
  hipMalloc(&d_ctr, sizeof(int));
  hipMemset(d_ctr, 0, sizeof(int));

  agg_count<<<(N + 63) / 64, 64>>>(d_ctr, N);
  hipDeviceSynchronize();

  int counter = -1;
  hipMemcpy(&counter, d_ctr, sizeof(int), hipMemcpyDeviceToHost);

  // After the fix, every thread forms a singleton group with size()==1
  // and thread_rank()==0, so every thread adds 1 -> counter == N.
  // Before the fix, size()==0 so atomicAdd adds 0 -> counter == 0.
  int ok = (counter == N);
  printf("agg counter: %d (expected %d) => %s\n", counter, N,
         ok ? "PASS" : "FAIL");

  hipFree(d_ctr);
  return ok ? 0 : 1;
}
