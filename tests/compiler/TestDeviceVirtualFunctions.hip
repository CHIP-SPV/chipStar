// Compile regression test for CHIP-SPV/chipStar#1373.
//
// Clang emits C++ vtables for SPIR-V targets with every slot in the global
// address space, which makes each virtual function slot an addrspacecast of a
// function into CrossWorkGroup. SPIR-V only allows casts into Generic, so the
// device link step used to abort with
//
//   InvalidModule: Invalid SPIR-V module: Casts from private/local/global
//   address space are allowed only to generic
//
// before the code ever reached a device. HipFunctionPointerASPass moves those
// slots into the generic address space.
//
// This test only compiles. -c still runs the device link that translates to
// SPIR-V, which is exactly where the unfixed compiler failed, so a regression
// of the pass is caught here on every target. It deliberately does not launch
// anything, because executing an indirect call needs
// SPV_INTEL_function_pointers, which only Intel's stack implements today. The
// runtime check that validates the dispatched values numerically is
// tests/runtime/TestDeviceVirtualDispatch.hip, which gates itself on device
// capability.

#include <hip/hip_runtime.h>

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
// devirtualize the call and quietly stop exercising the vtable.
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
