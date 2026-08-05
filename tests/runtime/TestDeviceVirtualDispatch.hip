// Runtime check for device side C++ virtual dispatch (CHIP-SPV/chipStar#1373).
//
// The compile side of this is covered everywhere by
// tests/compiler/TestDeviceVirtualFunctions.hip. This test additionally
// executes the dispatch and checks the results numerically, which requires a
// device that can actually run an indirect call.
//
// Capability gate
// ---------------
// Device side indirect calls are modelled with SPV_INTEL_function_pointers,
// and support for it is not discoverable at runtime:
//
//   * Intel accepts such modules but does not advertise
//     cl_intel_function_pointers in CL_DEVICE_EXTENSIONS, so the extension
//     string is not a usable probe.
//   * rusticl rejects the module during the device JIT ("spirv_to_nir
//     failed"), which surfaces only once a kernel is launched.
//   * PoCL neither runs nor rejects it, it hangs, so "try it and catch the
//     build error" cannot work either.
//
// Because of the PoCL behaviour the decision has to be made before the first
// launch, so we allow-list the stacks known to implement the extension rather
// than probing. This mirrors the isIntelGPU idiom the OpenCL backend already
// uses in src/backend/OpenCL/CHIPBackendOpenCL.cc. Everything else is skipped.

#include <hip/hip_interop.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

// Declared weakly so this test also links on a build without the OpenCL
// backend, where libOpenCL is absent and the symbol is simply null.
extern "C" __attribute__((weak)) int clGetDeviceInfo(void *Device,
                                                     unsigned ParamName,
                                                     size_t ParamValueSize,
                                                     void *ParamValue,
                                                     size_t *ParamValueSizeRet);

// From cl.h, spelled out so that this test needs no OpenCL headers.
static constexpr unsigned ClDeviceType = 0x1000;
static constexpr unsigned ClDeviceVendor = 0x102C;
static constexpr unsigned long long ClDeviceTypeGpu = 1 << 2;

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

  // Level Zero is Intel's own stack, which implements the extension.
  if (!strcmp(Backend, "level0"))
    return true;

  if (strcmp(Backend, "opencl"))
    return false;

  // Handles[2] is the cl_device_id, see CHIPQueueOpenCL::getBackendHandles.
  if (NumHandles < 3 || !clGetDeviceInfo)
    return false;
  void *Device = reinterpret_cast<void *>(Handles[2]);

  char Vendor[256] = {0};
  if (clGetDeviceInfo(Device, ClDeviceVendor, sizeof(Vendor) - 1, Vendor,
                      nullptr))
    return false;

  unsigned long long Type = 0;
  if (clGetDeviceInfo(Device, ClDeviceType, sizeof(Type), &Type, nullptr))
    return false;

  return strstr(Vendor, "Intel") != nullptr && (Type & ClDeviceTypeGpu);
}

struct Shape {
  double Scale;
  __device__ Shape(double S) : Scale(S) {}
  __device__ virtual ~Shape() {}
  __device__ virtual double area() const { return Scale; }
  __device__ virtual int tag() const { return 0; }
};

struct Square : Shape {
  __device__ Square(double S) : Shape(S) {}
  __device__ ~Square() override {}
  __device__ double area() const override { return Scale * Scale; }
  __device__ int tag() const override { return 1; }
};

struct Circle : Shape {
  __device__ Circle(double S) : Shape(S) {}
  __device__ ~Circle() override {}
  __device__ double area() const override { return 3.0 * Scale * Scale; }
  __device__ int tag() const override { return 2; }
};

// The dynamic type is picked from a runtime value so the compiler cannot
// devirtualize the call and quietly turn the test into a no-op.
__global__ void dispatchKernel(const int *Sel, double *Area, int *Tag) {
  int I = threadIdx.x;
  Square Sq(2.0);
  Circle Ci(2.0);
  Shape *S = Sel[I] ? static_cast<Shape *>(&Ci) : static_cast<Shape *>(&Sq);
  Area[I] = S->area();
  Tag[I] = S->tag();
}

// Here the base pointer is round tripped through memory, which is the shape
// real code such as Trilinos STK produces.
__device__ __attribute__((noinline)) double areaOf(const Shape *S) {
  return S->area();
}

__global__ void indirectDispatchKernel(const int *Sel, double *Area) {
  int I = threadIdx.x;
  Square Sq(3.0);
  Circle Ci(3.0);
  const Shape *Table[2] = {&Sq, &Ci};
  Area[I] = areaOf(Table[Sel[I] & 1]);
}

// Dispatch through an *abstract* base. Clang keeps a __cxa_pure_virtual slot in
// the vtable of any class with an unoverridden pure virtual member, and the
// device side has to define that symbol (CHIP-SPV/chipStar#1381). Without it
// the consumer rejects the module and every kernel in the program silently
// fails to build, while launches keep reporting hipSuccess.
struct Abstract {
  __device__ virtual ~Abstract() {}
  __device__ virtual double area() const = 0;
};

struct AbstractRect final : Abstract {
  double W, H;
  __device__ AbstractRect(double W_, double H_) : W(W_), H(H_) {}
  __device__ double area() const final { return W * H; }
};

struct AbstractSquare final : Abstract {
  double S;
  __device__ explicit AbstractSquare(double S_) : S(S_) {}
  __device__ double area() const final { return S * S; }
};

__global__ void pureVirtualKernel(const int *Sel, double *Area) {
  int I = threadIdx.x;
  AbstractRect R(3.0, 4.0);
  AbstractSquare Sq(5.0);
  const Abstract *Table[2] = {&R, &Sq};
  Area[I] = Table[Sel[I] & 1]->area();
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

  const int N = 4;
  int HSel[N] = {0, 1, 1, 0};
  double WantArea[N] = {4.0, 12.0, 12.0, 4.0};
  int WantTag[N] = {1, 2, 2, 1};
  double WantIndirectArea[N] = {9.0, 27.0, 27.0, 9.0};
  double WantPureVirtualArea[N] = {12.0, 25.0, 25.0, 12.0};

  int *DSel = nullptr;
  double *DArea = nullptr;
  int *DTag = nullptr;
  (void)hipMalloc(&DSel, N * sizeof(int));
  (void)hipMalloc(&DArea, N * sizeof(double));
  (void)hipMalloc(&DTag, N * sizeof(int));
  (void)hipMemcpy(DSel, HSel, N * sizeof(int), hipMemcpyHostToDevice);

  bool Ok = true;
  double HArea[N] = {0};
  int HTag[N] = {0};

  dispatchKernel<<<1, N>>>(DSel, DArea, DTag);
  (void)hipDeviceSynchronize();
  Err = hipGetLastError();
  if (Err != hipSuccess) {
    printf("FAIL: dispatchKernel: %s\n", hipGetErrorString(Err));
    Ok = false;
  }
  (void)hipMemcpy(HArea, DArea, N * sizeof(double), hipMemcpyDeviceToHost);
  (void)hipMemcpy(HTag, DTag, N * sizeof(int), hipMemcpyDeviceToHost);
  for (int I = 0; I < N; I++) {
    if (HArea[I] != WantArea[I] || HTag[I] != WantTag[I]) {
      printf("FAIL: dispatch[%d]: area=%f (want %f) tag=%d (want %d)\n", I,
             HArea[I], WantArea[I], HTag[I], WantTag[I]);
      Ok = false;
    }
  }

  for (int I = 0; I < N; I++)
    HArea[I] = 0;
  indirectDispatchKernel<<<1, N>>>(DSel, DArea);
  (void)hipDeviceSynchronize();
  Err = hipGetLastError();
  if (Err != hipSuccess) {
    printf("FAIL: indirectDispatchKernel: %s\n", hipGetErrorString(Err));
    Ok = false;
  }
  (void)hipMemcpy(HArea, DArea, N * sizeof(double), hipMemcpyDeviceToHost);
  for (int I = 0; I < N; I++) {
    if (HArea[I] != WantIndirectArea[I]) {
      printf("FAIL: indirectDispatch[%d]: area=%f (want %f)\n", I, HArea[I],
             WantIndirectArea[I]);
      Ok = false;
    }
  }

  for (int I = 0; I < N; I++)
    HArea[I] = 0;
  pureVirtualKernel<<<1, N>>>(DSel, DArea);
  (void)hipDeviceSynchronize();
  Err = hipGetLastError();
  if (Err != hipSuccess) {
    printf("FAIL: pureVirtualKernel: %s\n", hipGetErrorString(Err));
    Ok = false;
  }
  (void)hipMemcpy(HArea, DArea, N * sizeof(double), hipMemcpyDeviceToHost);
  for (int I = 0; I < N; I++) {
    if (HArea[I] != WantPureVirtualArea[I]) {
      printf("FAIL: pureVirtual[%d]: area=%f (want %f)\n", I, HArea[I],
             WantPureVirtualArea[I]);
      Ok = false;
    }
  }

  (void)hipFree(DSel);
  (void)hipFree(DArea);
  (void)hipFree(DTag);

  if (Ok)
    printf("PASSED\n");
  return Ok ? 0 : 1;
}
