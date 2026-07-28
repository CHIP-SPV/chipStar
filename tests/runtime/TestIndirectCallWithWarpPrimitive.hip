// A device indirect call must still receive its arguments when the module also
// contains a warp primitive.
//
// HipWarpsPass stamps intel_reqd_sub_group_size on kernels so that __shfl and
// friends see the warp width chipStar was built for. It used to stamp *every*
// kernel as soon as a single shuffle declaration appeared anywhere in the
// module. A stamped kernel then loses the arguments of an indirect call: the
// callee gets the first argument and zero for every argument after it,
// silently - no error, no diagnostic, just wrong results.
//
// One #include of Kokkos_Core.hpp was enough to trigger this, which broke
// device-side dispatch for every Kokkos application (kynema-ugf / Nalu-Wind
// builds its whole kernel architecture on device-side virtual calls, which
// lower to the same indirect call).
//
// touchShfl() below never runs. It only has to exist so the shuffle
// declaration is in the module.

#include <hip/hip_runtime.h>
#include <cstdio>

static __device__ double implA(double X, int I) {
  return X * 1000.0 + I * 10.0 + 6.0;
}
static __device__ double implB(double X, int I) { return -1.0; }

typedef double (*Callee)(double, int);

__device__ int Sink;
__global__ void touchShfl(int V) { Sink = __shfl(V, 0); }

// Select through a kernel argument so the indirect call survives the optimizer.
__global__ void dispatch(double *Out, int Sel) {
  Callee Fns[] = {implA, implB};
  Out[0] = Fns[Sel](7.0, 3);
}

#define CHECK(Expr)                                                            \
  do {                                                                         \
    hipError_t Err_ = (Expr);                                                  \
    if (Err_ != hipSuccess) {                                                  \
      printf("FAILED: %s -> %s\n", #Expr, hipGetErrorString(Err_));            \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  double *Out;
  CHECK(hipMalloc(&Out, sizeof(double)));

  dispatch<<<1, 1>>>(Out, 0);
  CHECK(hipDeviceSynchronize());

  double H = -1.0;
  CHECK(hipMemcpy(&H, Out, sizeof(double), hipMemcpyDeviceToHost));

  // 7.0 * 1000 + 3 * 10 + 6.0. A dropped second argument gives 7006.0.
  if (H != 7036.0) {
    printf("FAILED: indirect f(7.0, 3) = %f, want 7036.000000\n", H);
    return 1;
  }
  printf("PASSED\n");
  return 0;
}
