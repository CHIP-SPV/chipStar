/*
 * Copyright (c) 2021-24 chipStar developers
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

void MemoryManager::init(cl::Context C, cl::Device D, CHIPContextUSMExts &U,
                         bool FineGrain, bool IntelUSM) {
  Device_ = D;
  Context_ = C;
  USM = U;
  UseSVMFineGrain = FineGrain;
  UseIntelUSM = IntelUSM;
}

MemoryManager &MemoryManager::operator=(MemoryManager &&Rhs) {
  Allocations_ = std::move(Rhs.Allocations_);
  Context_ = std::move(Rhs.Context_);
  Device_ = std::move(Rhs.Device_);
  USM = std::move(Rhs.USM);
  UseSVMFineGrain = Rhs.UseSVMFineGrain;
  UseIntelUSM = Rhs.UseIntelUSM;
  return *this;
}

void *MemoryManager::allocate(size_t Size, size_t Alignment,
                              hipMemoryType MemType) {
  // 0 passed for the alignment will use the default alignment which is equal to
  // the largest data type supported.
  void *Ptr;
  int Err;
  if (UseIntelUSM) {
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
  } else if (UseSVMFineGrain) {
    Ptr = ::clSVMAlloc(
        Context_(), CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, Size, 0);
  } else {
    Ptr = ::clSVMAlloc(Context_(), CL_MEM_READ_WRITE, Size, 0);
  }

  if (Ptr) {
    auto Deleter = [Ctx = this->Context_, UseUSM = this->UseIntelUSM,
                    clMemFreeINTEL =
                        this->USM.clMemFreeINTEL](void *PtrToFree) -> void {
      logTrace("clSVMFree on: {}\n", PtrToFree);
      if (UseUSM)
        clMemFreeINTEL(Ctx(), PtrToFree);
      else
        clSVMFree(Ctx(), PtrToFree);
    };
    auto SPtr = std::shared_ptr<void>(Ptr, Deleter);
    logTrace("Memory allocated: {} / {}\n", Ptr, Size);
    assert(Allocations_.find(SPtr) == Allocations_.end());
    Allocations_.emplace(SPtr, Size);
  } else
    CHIPERR_LOG_AND_THROW("clSVMAlloc failed", hipErrorMemoryAllocation);

  return Ptr;
}

bool MemoryManager::free(void *Ptr) {
  auto I = Allocations_.find(Ptr);
  if (I != Allocations_.end())
    Allocations_.erase(I);
  return true;
}

bool MemoryManager::hasPointer(const void *Ptr) {
  logTrace("hasPointer on: {}\n", Ptr);
  return (Allocations_.find((void *)Ptr) != Allocations_.end());
}

bool MemoryManager::pointerSize(void *Ptr, size_t *Size) {
  logTrace("pointerSize on: {}\n", Ptr);
  auto I = Allocations_.find(Ptr);
  if (I != Allocations_.end()) {
    *Size = I->second;
    return true;
  } else {
    return false;
  }
}

bool MemoryManager::pointerInfo(void *Ptr, void **Base, size_t *Size) {
  logTrace("pointerInfo on: {}\n", Ptr);
  for (auto I : Allocations_) {
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

void MemoryManager::clear() { Allocations_.clear(); }
