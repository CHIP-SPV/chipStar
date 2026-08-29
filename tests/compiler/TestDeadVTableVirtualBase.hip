// Input of TestDeadVTableVirtualBase.bash, the reproducer for
// CHIP-SPV/chipStar#1382.
//
// The explicit instantiation makes clang emit the vtable, VTT and construction
// vtable of Impl<int> into the device module with weak_odr linkage although
// nothing on the device dispatches through them. Their slots are null because
// the virtual members are host only, and the virtual base offset is an
// inttoptr constant, which is what aborted llvm-spirv on Tpetra::RowMatrix's
// HIP instantiations:
//
//   SPIRVWriter.cpp: transConstantUse: Assertion `(C->getType()->isPointerTy()
//   || ExpectedType->isTypeUntypedPointerKHR()) && "Only pointer type
//   mismatches should be possible"' failed.

#include <hip/hip_runtime.h>

template <typename T> struct Base {
  T Pad;
  virtual ~Base() {}
  virtual T get() const { return Pad; }
};

template <typename T> struct Impl : virtual Base<T> {
  T V;
  T get() const override { return V; }
};

template struct Impl<int>;

__global__ void unrelated(int *Out) { Out[0] = 1; }
