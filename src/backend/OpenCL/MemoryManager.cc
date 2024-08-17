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

void MemoryManager::init(CHIPContextOpenCL *ChipCtxCl_) {
  ChipCtxCl = ChipCtxCl_;
  CHIPDeviceOpenCL *ChipDevCl = (CHIPDeviceOpenCL *)ChipCtxCl->getDevice();
  Device_ = *ChipDevCl->get();
  Context_ = *ChipCtxCl_->get();

  // Initialize set of allowed allocation strategies.
  using AS = AllocationStrategy;
  std::set<AS> AllowedAllocStrats;
  if (auto ChoiceStrOpt = ChipEnvVars.getOclUseAllocStrategy()) {
    // Note: the environment variable is lower-cased by ChipEnvVars instance.
    const auto &ChoiceStr = *ChoiceStrOpt;

#ifdef CHIP_USE_INTEL_USM
    if (ChoiceStr == "intelusm" || ChoiceStr == "usm")
      AllowedAllocStrats = {AS::IntelUSM};
    else
#endif
        if (ChoiceStr == "svm")
      AllowedAllocStrats = {AS::FineGrainSVM, AS::CoarseGrainSVM};
    else if (ChoiceStr == "bufferdevaddr")
      AllowedAllocStrats = {AS::BufferDevAddr};
    else
      CHIPERR_LOG_AND_THROW("Unrecognized allocation strategy.",
                            hipErrorInitializationError);
  } else {
    // Default set.
    AllowedAllocStrats = {
#ifdef CHIP_USE_INTEL_USM
        AS::IntelUSM,
#endif
        AS::FineGrainSVM, AS::CoarseGrainSVM, AS::BufferDevAddr};
  }

  std::string DevExts = Device_.getInfo<CL_DEVICE_EXTENSIONS>();
  cl_device_svm_capabilities SVMCapabilities =
      Device_.getInfo<CL_DEVICE_SVM_CAPABILITIES>();
  auto ChosenStrategy = AS::Unset;

  // Select an available strategy in this order.
  if (AllowedAllocStrats.count(AS::IntelUSM) &&
      DevExts.find("cl_intel_unified_shared_memory") != std::string::npos) {
    logDebug("Chosen allocation strategy: Intel USM.");
    ChosenStrategy = AS::IntelUSM;
  } else if (AllowedAllocStrats.count(AS::FineGrainSVM) &&
             (SVMCapabilities & (CL_DEVICE_SVM_FINE_GRAIN_BUFFER |
                                 CL_DEVICE_SVM_FINE_GRAIN_SYSTEM))) {
    logDebug("Chosen allocation strategy: fine-grain SVM.");
    ChosenStrategy = AS::FineGrainSVM;
  } else if (AllowedAllocStrats.count(AS::CoarseGrainSVM) &&
             (SVMCapabilities & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)) {
    logDebug("Chosen allocation strategy: coarse-grain SVM.");
    ChosenStrategy = AS::CoarseGrainSVM;
  } else if (AllowedAllocStrats.count(AS::BufferDevAddr) &&
             DevExts.find("cl_ext_buffer_device_address") !=
                 std::string::npos) {
    logDebug("Chosen allocation strategy: buffer-device-address\n");
    ChosenStrategy = AS::BufferDevAddr;
  }

  if (ChosenStrategy == AS::Unset)
    CHIPERR_LOG_AND_THROW("Insufficient memory capabilities.",
                          hipErrorInitializationError);
  AllocStrategy_ = ChosenStrategy;

  const cl::Platform &Plat = ChipCtxCl_->getPlatform();
  if (AllocStrategy_ == AllocationStrategy::IntelUSM) {
    USM_.clSharedMemAllocINTEL =
        (clSharedMemAllocINTEL_fn)::clGetExtensionFunctionAddressForPlatform(
            Plat(), "clSharedMemAllocINTEL");
    USM_.clDeviceMemAllocINTEL =
        (clDeviceMemAllocINTEL_fn)::clGetExtensionFunctionAddressForPlatform(
            Plat(), "clDeviceMemAllocINTEL");
    USM_.clHostMemAllocINTEL =
        (clHostMemAllocINTEL_fn)::clGetExtensionFunctionAddressForPlatform(
            Plat(), "clHostMemAllocINTEL");
    USM_.clMemFreeINTEL =
        (clMemFreeINTEL_fn)::clGetExtensionFunctionAddressForPlatform(
            Plat(), "clMemFreeINTEL");
  } else {
    std::memset(&USM_, 0, sizeof(USM_));
  }

  // Initialize the allocation usage bools
  hostAllocUsed = false;
  deviceAllocUsed = false;
  sharedAllocUsed = false;
}

MemoryManager &MemoryManager::operator=(MemoryManager &&Rhs) {
  Allocations_ = std::move(Rhs.Allocations_);
  Context_ = std::move(Rhs.Context_);
  Device_ = std::move(Rhs.Device_);
  USM_ = std::move(Rhs.USM_);
  AllocStrategy_ = std::move(Rhs.AllocStrategy_);
  return *this;
}

std::shared_ptr<void> MemoryManager::allocateSVM(size_t Size, size_t Alignment,
                                                 hipMemoryType MemType,
                                                 bool FineGrain) {
  std::shared_ptr<void> Result;
  auto MemFlags = CL_MEM_READ_WRITE;
  MemFlags |= FineGrain ? CL_MEM_SVM_FINE_GRAIN_BUFFER : 0;
  void *RawPtr = ::clSVMAlloc(Context_(), MemFlags, Size, 0);
  if (!RawPtr)
    return Result;

  auto Deleter = [Ctx = this->Context_()](void *PtrToFree) -> void {
    clSVMFree(Ctx, PtrToFree);
  };

  Result.reset(RawPtr, Deleter);
  return Result;
}

std::shared_ptr<void> MemoryManager::allocateUSM(size_t Size, size_t Alignment,
                                                 hipMemoryType MemType) {
  std::shared_ptr<void> Result;
  cl_int Err;
  void *RawPtr = nullptr;
  switch (MemType) {
  default:
    assert(!"Unexpected HIP memory type!");
    return Result;
  case hipMemoryTypeHost:
    RawPtr =
        USM_.clHostMemAllocINTEL(Context_(), nullptr, Size, Alignment, &Err);
    break;
  case hipMemoryTypeDevice:
    RawPtr = USM_.clDeviceMemAllocINTEL(Context_(), Device_(), nullptr, Size,
                                        Alignment, &Err);
    break;
  case hipMemoryTypeManaged:
  case hipMemoryTypeUnified:
    RawPtr = USM_.clSharedMemAllocINTEL(Context_(), Device_(), nullptr, Size,
                                        Alignment, &Err);
    break;
  }

  if (!RawPtr || Err != CL_SUCCESS)
    return Result;

  auto Deleter =
      [Ctx = this->Context_(), clMemFreeINTEL = this->USM_.clMemFreeINTEL](
          void *PtrToFree) -> void { clMemFreeINTEL(Ctx, PtrToFree); };

  Result.reset(RawPtr, Deleter);
  return Result;
}

std::shared_ptr<void>
MemoryManager::allocateBufferDevAddr(size_t Size, size_t Alignment,
                                     hipMemoryType MemType) {
  std::shared_ptr<void> Result;
  cl_int Err = CL_SUCCESS;
  void *RawPtr = nullptr;
  cl::Buffer Buf;

  switch (MemType) {
  default:
    assert(!"Unexpected HIP memory type!");
    return Result;
    break;

  case hipMemoryTypeDevice: {
    const cl_mem_flags MemFlags = CL_MEM_READ_WRITE | CL_MEM_DEVICE_ADDRESS_EXT;
    Buf = cl::Buffer(Context_, MemFlags, Size, nullptr, &Err);
    if (Err != CL_SUCCESS)
      break;
    Err = Buf.getInfo(CL_MEM_DEVICE_PTR_EXT, &RawPtr);
    DevPtrToBuffer_.insert(std::make_pair(RawPtr, Buf.get()));
    break;
  }
  case hipMemoryTypeHost:
  case hipMemoryTypeManaged:
  case hipMemoryTypeUnified:
    logWarn("hipMemoryTypeHost/Managed/Unified memory types are not currently "
            "supported.");
    return Result;
  }

  if (!RawPtr || Err != CL_SUCCESS)
    return Result;

  auto Deleter = [ToBeDestructed = Buf](void *Ignored) -> void {
    // This lambda keeps the buffer (Buf) alive until the outer object gets
    // destroyed.
  };
  Result.reset(RawPtr, Deleter);
  return Result;
}

void *MemoryManager::allocate(size_t Size, size_t Alignment,
                              hipMemoryType MemType) {
  std::shared_ptr<void> Ptr;

  switch (AllocStrategy_) {
  default:
    assert(!"Unexpected allocation strategy!");
    return nullptr;
  case AllocationStrategy::FineGrainSVM:
  case AllocationStrategy::CoarseGrainSVM: {
    Ptr = allocateSVM(Size, Alignment, MemType,
                      AllocStrategy_ == AllocationStrategy::FineGrainSVM);
    break;
  }
  case AllocationStrategy::IntelUSM: {
    Ptr = allocateUSM(Size, Alignment, MemType);
    break;
  }
  case AllocationStrategy::BufferDevAddr:
    Ptr = allocateBufferDevAddr(Size, Alignment, MemType);
    break;
  }

  if (!Ptr)
    CHIPERR_LOG_AND_THROW("Memory allocation failed", hipErrorMemoryAllocation);

  logTrace("Memory allocated: {} / {}\n", Ptr.get(), Size);
  assert(Allocations_.find(Ptr) == Allocations_.end());
  Allocations_.emplace(Ptr, Size);

  // Set the appropriate bool based on the MemType
  switch (MemType) {
  case hipMemoryTypeHost:
    hostAllocUsed = true;
    break;
  case hipMemoryTypeDevice:
    deviceAllocUsed = true;
    break;
  case hipMemoryTypeManaged:
  case hipMemoryTypeUnified:
    sharedAllocUsed = true;
    break;
  default:
    // Handle unexpected memory type
    assert(!"Unexpected memory type!");
    break;
  }

  return Ptr.get();
}

bool MemoryManager::free(void *Ptr) {
  auto I = Allocations_.find(Ptr);
  if (I != Allocations_.end())
    Allocations_.erase(I);

  if (AllocStrategy_ == AllocationStrategy::BufferDevAddr)
    DevPtrToBuffer_.erase(Ptr);

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

void MemoryManager::clear() {
  Backend->getActiveDevice()->getLegacyDefaultQueue()->finish();
  Allocations_.clear();
}

/// Returns a cl_mem object the 'DevPtr' corresponds to and distance
/// to its base.
///
/// returns {nullptr, 0 } if 'DevPtr' is not pointing to any known
/// device pinned allocation.
///
/// Precondition: getAllocStrategy() == AllocationStrategy::BufferDevAddr;
std::pair<cl_mem, size_t>
MemoryManager::translateDevPtrToBuffer(const void *DevPtr) const {
  assert(getAllocStrategy() == AllocationStrategy::BufferDevAddr);

  // Find entry with largest key so that 'key <= DevPtr'.
  auto UpperIt = Allocations_.upper_bound(DevPtr);
  auto CandIt = UpperIt == Allocations_.begin() ? UpperIt : std::prev(UpperIt);
  if (CandIt == Allocations_.end() || DevPtr < CandIt->first.get())
    return {nullptr, 0};

  const char *BasePtr = static_cast<const char *>(CandIt->first.get());
  size_t Offset = 0;

  // Handle offseted pointer, check it is within the bounds of the allocation.
  if (BasePtr != DevPtr) {
    const char *CandPtr = static_cast<const char *>(DevPtr);
    if (CandPtr >= (BasePtr + CandIt->second))
      return {nullptr, 0};
    Offset = CandPtr - BasePtr;
  }

  return {DevPtrToBuffer_.at(BasePtr), Offset};
}