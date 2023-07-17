// Regression test for CHIP-SPV/chipStar#364.
#include <hip/hip_runtime.h>

__global__ void callShfl(double *InOut, int LaneSel) {
  *InOut = __shfl(*InOut, LaneSel);
}
__global__ void callShflUp(double *InOut, int LaneSel) {
  *InOut = __shfl_up(*InOut, LaneSel);
}
__global__ void callShflDown(double *InOut, int LaneSel) {
  *InOut = __shfl_down(*InOut, LaneSel);
}
__global__ void callShflXor(double *InOut, int LaneSel) {
  *InOut = __shfl_xor(*InOut, LaneSel);
}
