// Compile-only reproducer for CHIP-SPV/chipStar#1373: device side virtual
// dispatch. Clang emits every vtable slot as a function pointer cast into the
// global address space. SPIR-V has casts into and out of Generic but none from
// the Function storage class (a plain pointer) into CrossWorkgroup, so the
// device link that -c already performs fails before anything reaches a device:
//
//   InvalidModule: Invalid SPIR-V module: Casts from private/local/global
//   address space are allowed only to generic
//
// The dynamic type is picked from a kernel argument so the call cannot be
// devirtualized. Executing the dispatch needs SPV_INTEL_function_pointers and
// is covered by tests/runtime/TestDeviceVirtualDispatch.hip.

#include <hip/hip_runtime.h>

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

__global__ void dispatch(const int *Sel, float *Area) {
  Square Sq(2.0f);
  Circle Ci(2.0f);
  const Shape *S = Sel[threadIdx.x] ? static_cast<const Shape *>(&Ci) : &Sq;
  Area[threadIdx.x] = S->area();
}

// A virtual base gives the class a VTT and construction vtables. The VTT holds
// constant GEPs into those tables, which must be rebuilt when the tables are
// retyped.
struct VBase {
  __device__ virtual ~VBase() {}
  __device__ virtual int id() const { return 1; }
};

struct Mid : virtual VBase {
  __device__ int id() const override { return 2; }
};

struct Leaf : Mid {
  __device__ int id() const override { return 3; }
};

__global__ void dispatchVirtualBase(const int *Sel, int *Out) {
  Mid M;
  Leaf L;
  const VBase *B = Sel[threadIdx.x] ? static_cast<const VBase *>(&L) : &M;
  Out[threadIdx.x] = B->id();
}
