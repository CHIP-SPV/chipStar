// Runtime reproducer for device side C++ virtual dispatch
// (CHIP-SPV/chipStar#1373). The compile side is covered on every target by
// tests/compiler/TestDeviceVirtualFunctions.hip; this test executes the
// dispatch and checks the results, which needs a device that can run an
// indirect call.
//
// Indirect calls are modelled with SPV_INTEL_function_pointers and support
// cannot be probed at runtime: Intel accepts such modules without advertising
// cl_intel_function_pointers, rusticl rejects them only in the device JIT, and
// PoCL hangs. So the stacks known to implement it are allow-listed before the
// first launch, mirroring the isIntelGPU idiom of the OpenCL backend. A GPU is
// required as well as the Intel vendor: an Intel OpenCL CPU device reports the
// same vendor string.

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
static constexpr unsigned ClDeviceType = 0x1000;   // CL_DEVICE_TYPE
static constexpr unsigned ClDeviceVendor = 0x102C; // CL_DEVICE_VENDOR
static constexpr unsigned long long ClDeviceTypeGpu = 1 << 2; // CL_DEVICE_TYPE_GPU

static bool deviceSupportsIndirectCalls() {
  int NumHandles = 0;
  if (hipGetBackendNativeHandles(0, nullptr, &NumHandles) != hipSuccess ||
      NumHandles < 1)
    return false;
  std::vector<uintptr_t> Handles(NumHandles);
  if (hipGetBackendNativeHandles(0, Handles.data(), nullptr) != hipSuccess)
    return false;

  const char *Backend = reinterpret_cast<const char *>(Handles[0]);
  if (!Backend)
    return false;
  if (!strcmp(Backend, "level0"))
    return true;
  // Handles[2] is the cl_device_id, see CHIPQueueOpenCL::getBackendHandles.
  if (strcmp(Backend, "opencl") || NumHandles < 3 || !clGetDeviceInfo)
    return false;
  void *Device = reinterpret_cast<void *>(Handles[2]);
  char Vendor[256] = {0};
  unsigned long long Type = 0;
  if (clGetDeviceInfo(Device, ClDeviceVendor, sizeof(Vendor) - 1, Vendor,
                      nullptr) ||
      clGetDeviceInfo(Device, ClDeviceType, sizeof(Type), &Type, nullptr))
    return false;
  return strstr(Vendor, "Intel") && (Type & ClDeviceTypeGpu);
}

struct Shape {
  float Scale;
  __device__ Shape(float S) : Scale(S) {}
  __device__ virtual ~Shape() {}
  __device__ virtual float area() const { return Scale; }
};

struct Square : Shape {
  __device__ Square(float S) : Shape(S) {}
  __device__ float area() const override { return Scale * Scale; }
};

struct Circle : Shape {
  __device__ Circle(float S) : Shape(S) {}
  __device__ float area() const override { return 3.0f * Scale * Scale; }
};

// The dynamic type comes from a runtime value so the call cannot be
// devirtualized and quietly turn the test into a no-op.
__global__ void dispatchKernel(const int *Sel, float *Area) {
  Square Sq(2.0f);
  Circle Ci(2.0f);
  const Shape *S = Sel[threadIdx.x] ? static_cast<const Shape *>(&Ci) : &Sq;
  Area[threadIdx.x] = S->area();
}

// The base pointer is round tripped through memory, which is the shape real
// code such as Trilinos STK produces.
__device__ __attribute__((noinline)) float areaOf(const Shape *S) {
  return S->area();
}

__global__ void indirectDispatchKernel(const int *Sel, float *Area) {
  Square Sq(3.0f);
  Circle Ci(3.0f);
  const Shape *Table[2] = {&Sq, &Ci};
  Area[threadIdx.x] = areaOf(Table[Sel[threadIdx.x] & 1]);
}

// Dispatch through an abstract base. Clang keeps a __cxa_pure_virtual slot in
// the vtable of any class with an unoverridden pure virtual member, and the
// device side has to define that symbol (CHIP-SPV/chipStar#1381). Without it
// the consumer rejects the module and every kernel in the program fails to
// build.
struct Abstract {
  // noinline keeps the store of Abstract's own vptr, and with it the reference
  // to __cxa_pure_virtual, in the module at every optimisation level. Inlined
  // into a derived constructor that store is dead and the vtable is dropped.
  __device__ __attribute__((noinline)) Abstract() {}
  __device__ virtual ~Abstract() {}
  __device__ virtual float area() const = 0;
};

struct Rect final : Abstract {
  float W, H;
  __device__ Rect(float W_, float H_) : W(W_), H(H_) {}
  __device__ float area() const final { return W * H; }
};

struct AbstractSquare final : Abstract {
  float S;
  __device__ explicit AbstractSquare(float S_) : S(S_) {}
  __device__ float area() const final { return S * S; }
};

__global__ void pureVirtualKernel(const int *Sel, float *Area) {
  Rect R(3.0f, 4.0f);
  AbstractSquare Sq(5.0f);
  const Abstract *Table[2] = {&R, &Sq};
  Area[threadIdx.x] = Table[Sel[threadIdx.x] & 1]->area();
}

static constexpr int N = 4;

template <typename KernelT>
static bool runAndCheck(const char *Name, KernelT Kernel, const int *DSel,
                        float *DArea, const float *Want) {
  float HArea[N] = {0};
  (void)hipMemset(DArea, 0, N * sizeof(float));
  Kernel<<<1, N>>>(DSel, DArea);
  (void)hipDeviceSynchronize();
  hipError_t Err = hipGetLastError();
  if (Err != hipSuccess) {
    printf("FAIL: %s: %s\n", Name, hipGetErrorString(Err));
    return false;
  }
  (void)hipMemcpy(HArea, DArea, N * sizeof(float), hipMemcpyDeviceToHost);
  bool Ok = true;
  for (int I = 0; I < N; I++)
    if (HArea[I] != Want[I]) {
      printf("FAIL: %s[%d]: %f (want %f)\n", Name, I, HArea[I], Want[I]);
      Ok = false;
    }
  return Ok;
}

int main() {
  int DeviceCount = 0;
  hipError_t Err = hipGetDeviceCount(&DeviceCount);
  if (Err == hipErrorInitializationError || Err == hipErrorNoDevice) {
    printf("HIP_SKIP_THIS_TEST: no backend available (error %d)\n", Err);
    return CHIP_SKIP_TEST;
  }
  if (!deviceSupportsIndirectCalls()) {
    printf("HIP_SKIP_THIS_TEST: device does not support device side indirect "
           "calls (SPV_INTEL_function_pointers)\n");
    return CHIP_SKIP_TEST;
  }

  const int HSel[N] = {0, 1, 1, 0};
  const float WantArea[N] = {4.0f, 12.0f, 12.0f, 4.0f};
  const float WantIndirectArea[N] = {9.0f, 27.0f, 27.0f, 9.0f};
  const float WantPureVirtualArea[N] = {12.0f, 25.0f, 25.0f, 12.0f};

  int *DSel = nullptr;
  float *DArea = nullptr;
  (void)hipMalloc(&DSel, N * sizeof(int));
  (void)hipMalloc(&DArea, N * sizeof(float));
  (void)hipMemcpy(DSel, HSel, N * sizeof(int), hipMemcpyHostToDevice);

  bool Ok = runAndCheck("dispatch", dispatchKernel, DSel, DArea, WantArea);
  Ok &= runAndCheck("indirectDispatch", indirectDispatchKernel, DSel, DArea,
                    WantIndirectArea);
  Ok &= runAndCheck("pureVirtual", pureVirtualKernel, DSel, DArea,
                    WantPureVirtualArea);

  (void)hipFree(DSel);
  (void)hipFree(DArea);
  if (Ok)
    printf("PASSED\n");
  return Ok ? 0 : 1;
}
