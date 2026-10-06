// A pointer selected at run time between __shared__ and global memory must not
// be launched on an OpenCL device without generic address space support
// (CHIP-SPV/chipStar#1807); other kernels of the module still run there.

#include <hip/hip_interop.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

// Weak so this also links on a build without the OpenCL backend.
extern "C" __attribute__((weak)) int clGetDeviceInfo(void *Device,
                                                     unsigned ParamName,
                                                     size_t ParamValueSize,
                                                     void *ParamValue,
                                                     size_t *ParamValueSizeRet);
// CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT
static constexpr unsigned ClDeviceGenericAS = 0x1069;

static bool deviceLacksGenericAS() {
  int NumHandles = 0;
  if (hipGetBackendNativeHandles(0, nullptr, &NumHandles) != hipSuccess ||
      NumHandles < 3)
    return false;
  std::vector<uintptr_t> Handles(NumHandles);
  if (hipGetBackendNativeHandles(0, Handles.data(), nullptr) != hipSuccess)
    return false;
  const char *Backend = reinterpret_cast<const char *>(Handles[0]);
  if (!Backend || strcmp(Backend, "opencl") || !clGetDeviceInfo)
    return false;
  unsigned Supported = 1; // cl_bool
  // Handles[2] is the cl_device_id, see CHIPQueueOpenCL::getBackendHandles.
  return !clGetDeviceInfo(reinterpret_cast<void *>(Handles[2]),
                          ClDeviceGenericAS, sizeof(Supported), &Supported,
                          nullptr) &&
         !Supported;
}

__global__ void mixed(int *G, int C) {
  __shared__ int S;
  if (threadIdx.x == 0)
    S = 0;
  __syncthreads();
  int *P = C ? &S : G;
  atomicAdd(P, 1);
  __syncthreads();
  if (threadIdx.x == 0 && C)
    G[1] = S;
}

__global__ void plain(int *G) {
  __shared__ int S;
  if (threadIdx.x == 0)
    S = 0;
  __syncthreads();
  atomicAdd(&S, 1);
  __syncthreads();
  if (threadIdx.x == 0)
    G[1] = S;
}

// Launch Kernel and expect G = {Want0, Want1}.
template <typename F> static bool run(F Kernel, int *G, int Want0, int Want1) {
  int H[2] = {0, 0};
  if (hipMemcpy(G, H, sizeof(H), hipMemcpyHostToDevice) != hipSuccess)
    return false;
  Kernel();
  if (hipGetLastError() != hipSuccess ||
      hipMemcpy(H, G, sizeof(H), hipMemcpyDeviceToHost) != hipSuccess)
    return false;
  return H[0] == Want0 && H[1] == Want1;
}

int main() {
  int *G;
  if (hipMalloc(&G, 2 * sizeof(int)) != hipSuccess)
    return 1;
  if (!run([=] { plain<<<1, 64>>>(G); }, G, 0, 64)) {
    printf("FAIL: plain\n");
    return 1;
  }

  if (deviceLacksGenericAS()) {
    // The global branch would run correctly; the launch must be refused anyway.
    mixed<<<1, 64>>>(G, 0);
    hipError_t Err = hipGetLastError();
    if (Err != hipErrorInvalidDeviceFunction) {
      printf("FAIL: launch without generic address space support returned "
             "%s\n",
             hipGetErrorName(Err));
      return 1;
    }
    printf("PASS\n");
    return 0;
  }

  if (!run([=] { mixed<<<1, 64>>>(G, 0); }, G, 64, 0) ||
      !run([=] { mixed<<<1, 64>>>(G, 1); }, G, 0, 64)) {
    printf("FAIL: mixed\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}
