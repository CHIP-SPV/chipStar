// Regression test for CHIP-SPV/chipStar#797.
// __shfl(__half, ...) was ambiguous because only integer/float/double
// overloads were declared; __half implicitly converts to several of them.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__global__ void callShflHalf(__half *InOut, int LaneSel) {
  *InOut = __shfl(*InOut, LaneSel);
}
__global__ void callShflUpHalf(__half *InOut, int LaneSel) {
  *InOut = __shfl_up(*InOut, LaneSel);
}
__global__ void callShflDownHalf(__half *InOut, int LaneSel) {
  *InOut = __shfl_down(*InOut, LaneSel);
}
__global__ void callShflXorHalf(__half *InOut, int LaneSel) {
  *InOut = __shfl_xor(*InOut, LaneSel);
}

__global__ void callShflHalf2(__half2 *InOut, int LaneSel) {
  *InOut = __shfl(*InOut, LaneSel);
}

__global__ void callShflSyncHalf(__half *InOut, int LaneSel) {
  *InOut = __shfl_sync(0xffffffff, *InOut, LaneSel);
}
__global__ void callShflUpSyncHalf(__half *InOut, int LaneSel) {
  *InOut = __shfl_up_sync(0xffffffff, *InOut, LaneSel);
}
__global__ void callShflDownSyncHalf(__half *InOut, int LaneSel) {
  *InOut = __shfl_down_sync(0xffffffff, *InOut, LaneSel);
}
__global__ void callShflXorSyncHalf(__half *InOut, int LaneSel) {
  *InOut = __shfl_xor_sync(0xffffffff, *InOut, LaneSel);
}
