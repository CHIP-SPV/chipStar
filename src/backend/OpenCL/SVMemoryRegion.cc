#include "CHIPBackendOpenCL.hh"

#define SVM_ALIGNMENT 128

void *SVMemoryRegion::allocate(cl::Context Ctx, size_t Size) {
  void *Ptr = ::clSVMAlloc(Ctx.get(), CL_MEM_READ_WRITE, Size, SVM_ALIGNMENT);
  if (Ptr) {
    logTrace("clSVMAlloc allocated: {} / {}\n", Ptr, Size);
    SvmAllocations.emplace(Ptr, Size);
  } else
    CHIPERR_LOG_AND_THROW("clSVMAlloc failed", hipErrorMemoryAllocation);
  return Ptr;
}

bool SVMemoryRegion::free(void *Ptr, size_t *Size) {
  auto I = SvmAllocations.find(Ptr);
  if (I != SvmAllocations.end()) {
    void *Ptr = I->first;
    *Size = I->second;
    logTrace("clSVMFree on: {}\n", Ptr);
    SvmAllocations.erase(I);
    ::clSVMFree(Context(), Ptr);
    return true;
  } else
    CHIPERR_LOG_AND_THROW("clSVMFree failure", hipErrorRuntimeMemory);
}

bool SVMemoryRegion::hasPointer(const void *Ptr) {
  logTrace("hasPointer on: {}\n", Ptr);
  return (SvmAllocations.find((void *)Ptr) != SvmAllocations.end());
}

bool SVMemoryRegion::pointerSize(void *Ptr, size_t *Size) {
  logTrace("pointerSize on: {}\n", Ptr);
  auto I = SvmAllocations.find(Ptr);
  if (I != SvmAllocations.end()) {
    *Size = I->second;
    return true;
  } else {
    return false;
  }
}

bool SVMemoryRegion::pointerInfo(void *Ptr, void **Base, size_t *Size) {
  logTrace("pointerInfo on: {}\n", Ptr);
  for (auto I : SvmAllocations) {
    if ((I.first <= Ptr) && (Ptr < ((const char *)I.first + I.second))) {
      if (Base)
        *Base = I.first;
      if (Size)
        *Size = I.second;
      return true;
    }
  }
  return false;
}

void SVMemoryRegion::clear() {
  for (auto I : SvmAllocations) {
    ::clSVMFree(Context(), I.first);
  }
  SvmAllocations.clear();
}
