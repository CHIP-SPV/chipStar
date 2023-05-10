/*
 * Copyright (c) 2021-22 CHIP-SPV developers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "CHIPBackendOpenCL.hh"

#define SVM_ALIGNMENT 128

void SVMemoryRegion::init(cl::Context C, cl::Device D, CHIPContextUSMExts U,
                          bool FineGrain, bool IntelUSM) {
  Device_ = D;
  Context_ = C;
  USM = U;
  SupportsFineGrain = FineGrain;
  SupportsIntelUSM = IntelUSM;
}

SVMemoryRegion &SVMemoryRegion::operator=(SVMemoryRegion &&Rhs) {
  SvmAllocations_ = std::move(Rhs.SvmAllocations_);
  Context_ = std::move(Rhs.Context_);
  Device_ = std::move(Rhs.Device_);
  USM = std::move(Rhs.USM);
  SupportsFineGrain = Rhs.SupportsFineGrain;
  SupportsIntelUSM = Rhs.SupportsIntelUSM;
  return *this;
}

void *SVMemoryRegion::allocate(size_t Size, size_t Alignment,
                               hipMemoryType MemType) {
  // 0 passed for the alignment will use the default alignment which is equal to
  // the largest data type supported.
  void *Ptr;
  int Err;
  if (SupportsIntelUSM) {
    switch (MemType) {
    case hipMemoryTypeHost:
      Ptr = USM.clHostMemAllocINTEL(Context_(), NULL, Size, Alignment, &Err);
      break;
    case hipMemoryTypeDevice:
      Ptr = USM.clDeviceMemAllocINTEL(Context_(), Device_(), NULL, Size,
                                      Alignment, &Err);
      break;
    case hipMemoryTypeManaged:
    case hipMemoryTypeUnified:
    default:
      Ptr = USM.clSharedMemAllocINTEL(Context_(), Device_(), NULL, Size,
                                      Alignment, &Err);
      break;
    }
  } else if (SupportsFineGrain) {
    Ptr = ::clSVMAlloc(
        Context_(), CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, Size, 0);
  } else {
    Ptr = ::clSVMAlloc(Context_(), CL_MEM_READ_WRITE, Size, 0);
  }

  if (Ptr) {
    auto Deleter = [Ctx = this->Context_, SupportsUSM = this->SupportsIntelUSM,
                    clMemFreeINTEL =
                        this->USM.clMemFreeINTEL](void *PtrToFree) -> void {
      logTrace("clSVMFree on: {}\n", PtrToFree);
      if (SupportsUSM)
        clMemFreeINTEL(Ctx(), PtrToFree);
      else
        clSVMFree(Ctx(), PtrToFree);
    };
    auto SPtr = std::shared_ptr<void>(Ptr, Deleter);
    logTrace("Memory allocated: {} / {}\n", Ptr, Size);
    SvmAllocations_.emplace(SPtr, Size);
  } else
    CHIPERR_LOG_AND_THROW("clSVMAlloc failed", hipErrorMemoryAllocation);

  return Ptr;
}

bool SVMemoryRegion::free(void *Ptr) {
  auto I = SvmAllocations_.find(Ptr);
  if (I != SvmAllocations_.end())
    SvmAllocations_.erase(I);
  return true;
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
    if ((I.first.get() <= Ptr) &&
        (Ptr < ((const char *)I.first.get() + I.second))) {
      if (Base)
        *Base = I.first.get();
      if (Size)
        *Size = I.second;
      return true;
    }
  }
  return false;
}

void SVMemoryRegion::clear() { SvmAllocations_.clear(); }
