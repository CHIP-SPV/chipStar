#include "CHIPBackendOpenCL.hh"

#define SVM_ALIGNMENT 128

SVMemoryRegion &SVMemoryRegion::operator=(SVMemoryRegion &&Rhs) {
  SvmAllocations_ = std::move(Rhs.SvmAllocations_);
  Context_ = std::move(Rhs.Context_);
  return *this;
}

void *SVMemoryRegion::allocate(size_t Size) {
  void *Ptr = ::clSVMAlloc(Context_(), CL_MEM_READ_WRITE, Size, SVM_ALIGNMENT);
  if (Ptr) {
    logTrace("clSVMAlloc allocated: {} / {}\n", Ptr, Size);
    SvmAllocations_.emplace(Ptr, Size);
  } else
    CHIPERR_LOG_AND_THROW("clSVMAlloc failed", hipErrorMemoryAllocation);
  return Ptr;
}

bool SVMemoryRegion::free(void *Ptr) {
  auto I = SvmAllocations_.find(Ptr);
  if (I != SvmAllocations_.end()) {
    void *Ptr = I->first;
    logTrace("clSVMFree on: {}\n", Ptr);
    SvmAllocations_.erase(I);
    ::clSVMFree(Context_(), Ptr);
    return true;
  } else
    CHIPERR_LOG_AND_THROW("clSVMFree failure", hipErrorRuntimeMemory);
}

bool SVMemoryRegion::hasPointer(const void *Ptr) {
  logTrace("hasPointer on: {}\n", Ptr);
  return (SvmAllocations_.find((void *)Ptr) != SvmAllocations_.end());
}

bool SVMemoryRegion::pointerSize(void *Ptr, size_t *Size) {
  logTrace("pointerSize on: {}\n", Ptr);
  auto I = SvmAllocations_.find(Ptr);
  if (I != SvmAllocations_.end()) {
    *Size = I->second;
    return true;
  } else {
    return false;
  }
}

bool SVMemoryRegion::pointerInfo(void *Ptr, void **Base, size_t *Size) {
  logTrace("pointerInfo on: {}\n", Ptr);
  for (auto I : SvmAllocations_) {
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
  for (auto I : SvmAllocations_) {
    ::clSVMFree(Context_(), I.first);
  }
  SvmAllocations_.clear();
}
