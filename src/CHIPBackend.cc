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

#include "CHIPBackend.hh"

#include <utility>

/// Queue a kernel for retrieving information about the device variable.
static void queueKernel(CHIPQueue *Q, CHIPKernel *K, void *Args[] = nullptr,
                        dim3 GridDim = dim3(1), dim3 BlockDim = dim3(1),
                        size_t SharedMemSize = 0) {
  assert(Q);
  assert(K);
  // FIXME: Should construct backend specific exec item or make the exec
  //        item a backend agnostic class.
  CHIPExecItem EI(GridDim, BlockDim, SharedMemSize, Q);
  EI.setArgPointer(Args);
  EI.setKernel(K);

  auto ChipQueue = EI.getQueue();
  if (!ChipQueue)
    CHIPERR_LOG_AND_THROW(
        "Tried to launch kernel for an ExecItem which has a null queue",
        hipErrorTbd);

  ChipQueue->launch(&EI);
}

/// Queue a shadow kernel for binding a device variable (a pointer) to
/// the given allocation.
static void queueVariableInfoShadowKernel(CHIPQueue *Q, CHIPModule *M,
                                          const CHIPDeviceVar *Var,
                                          void *InfoBuffer) {
  assert(M && Var && InfoBuffer);
  auto *K = M->getKernel(std::string(ChipVarInfoPrefix) + Var->getName());
  assert(K && "Module is missing a shadow kernel?");
  void *Args[] = {&InfoBuffer};
  queueKernel(Q, K, Args);
}

static void queueVariableBindShadowKernel(CHIPQueue *Q, CHIPModule *M,
                                          const CHIPDeviceVar *Var) {
  assert(M && Var);
  auto *DevPtr = Var->getDevAddr();
  assert(DevPtr && "Space has not be allocated for a variable.");
  auto *K = M->getKernel(std::string(ChipVarBindPrefix) + Var->getName());
  assert(K && "Module is missing a shadow kernel?");
  void *Args[] = {&DevPtr};
  queueKernel(Q, K, Args);
}

static void queueVariableInitShadowKernel(CHIPQueue *Q, CHIPModule *M,
                                          const CHIPDeviceVar *Var) {
  assert(M && Var);
  auto *K = M->getKernel(std::string(ChipVarInitPrefix) + Var->getName());
  assert(K && "Module is missing a shadow kernel?");
  queueKernel(Q, K);
}

CHIPCallbackData::CHIPCallbackData(hipStreamCallback_t TheCallbackF,
                                   void *TheCallbackArgs,
                                   CHIPQueue *TheChipQueue)
    : ChipQueue(TheChipQueue), CallbackArgs(TheCallbackArgs),
      CallbackF(TheCallbackF) {}

// CHIPDeviceVar
// ************************************************************************
CHIPDeviceVar::CHIPDeviceVar(std::string TheName, size_t TheSize)
    : Name_(TheName), Size_(TheSize) {}

CHIPDeviceVar::~CHIPDeviceVar() { assert(!DevAddr_ && "Memory leak?"); }

// CHIPAllocationTracker
// ************************************************************************
CHIPAllocationTracker::CHIPAllocationTracker(size_t GlobalMemSize,
                                             std::string Name)
    : GlobalMemSize(GlobalMemSize), TotalMemSize(0), MaxMemUsed(0) {
  Name_ = Name;
}
CHIPAllocationTracker::~CHIPAllocationTracker() {
  for (auto &Member : PtrToAllocInfo_) {
    AllocationInfo *AllocInfo = Member.second;
    delete AllocInfo;
  }
}

AllocationInfo *CHIPAllocationTracker::getAllocInfo(const void *Ptr) {
  {
    std::lock_guard<std::mutex> Lock(AllocationTrackerMtx);
    // In case that Ptr is the base of the allocation, check hash map directly
    auto Found = PtrToAllocInfo_.count(const_cast<void *>(Ptr));
    if (Found)
      return PtrToAllocInfo_[const_cast<void *>(Ptr)];
  }

  // Ptr can be offset from the base pointer. In this case, iterate through all
  // allocations, and check if Ptr falls within any of these allocation ranges
  auto AllocInfo = getAllocInfoCheckPtrRanges(const_cast<void *>(Ptr));

  return AllocInfo;
}

bool CHIPAllocationTracker::reserveMem(size_t Bytes) {
  std::lock_guard<std::mutex> Lock(AllocationTrackerMtx);
  if (Bytes <= (GlobalMemSize - TotalMemSize)) {
    TotalMemSize += Bytes;
    if (TotalMemSize > MaxMemUsed)
      MaxMemUsed = TotalMemSize;
    logDebug("Currently used memory on dev {}: {} M\n", Name_,
             (TotalMemSize >> 20));
    return true;
  } else {
    CHIPERR_LOG_AND_THROW("Failed to allocate memory",
                          hipErrorMemoryAllocation);
  }
}

bool CHIPAllocationTracker::releaseMemReservation(unsigned long Bytes) {
  std::lock_guard<std::mutex> Lock(AllocationTrackerMtx);
  if (TotalMemSize >= Bytes) {
    TotalMemSize -= Bytes;
    return true;
  }

  return false;
}

void CHIPAllocationTracker::recordAllocation(void *DevPtr, void *HostPtr,
                                             hipDevice_t Device, size_t Size,
                                             CHIPHostAllocFlags Flags,
                                             hipMemoryType MemoryType) {
  AllocationInfo *AllocInfo = new AllocationInfo{
      DevPtr, HostPtr, Size, Flags, Device, false, MemoryType};
  std::lock_guard<std::mutex> Lock(AllocationTrackerMtx);
  // TODO AllocInfo turned into class and constructor take care of this
  if (MemoryType == hipMemoryTypeHost)
    AllocInfo->HostPtr = AllocInfo->DevPtr;

  if (MemoryType == hipMemoryTypeUnified)
    AllocInfo->HostPtr = AllocInfo->DevPtr;

  if (DevPtr)
    PtrToAllocInfo_[DevPtr] = AllocInfo;
  if (HostPtr)
    PtrToAllocInfo_[HostPtr] = AllocInfo;

  logDebug(
      "CHIPAllocationTracker::recordAllocation size: {} HOST {} DEV {} TYPE {}",
      Size, HostPtr, DevPtr, (unsigned)MemoryType);
  return;
}

AllocationInfo *
CHIPAllocationTracker::getAllocInfoCheckPtrRanges(void *DevPtr) {
  std::lock_guard<std::mutex> Lock(AllocationTrackerMtx);
  for (auto &Info : PtrToAllocInfo_) {
    AllocationInfo *AllocInfo = Info.second;
    void *Start = AllocInfo->DevPtr;
    void *End = (char *)Start + AllocInfo->Size;

    if (Start <= DevPtr && DevPtr < End)
      return AllocInfo;
  }

  return nullptr;
}

// CHIPEvent
// ************************************************************************

CHIPEvent::CHIPEvent(CHIPContext *Ctx, CHIPEventFlags Flags)
    : EventStatus_(EVENT_STATUS_INIT), Flags_(Flags), Refc_(new size_t()),
      ChipContext_(Ctx), Msg("") {}

// CHIPModuleflags_
//*************************************************************************************
void CHIPModule::consumeSPIRV() {
  FuncIL_ = (uint8_t *)Src_.data();
  IlSize_ = Src_.length();

  // Parse the SPIR-V fat binary to retrieve kernel function
  size_t NumWords = IlSize_ / 4;
  BinaryData_ = new int32_t[NumWords + 1];
  std::memcpy(BinaryData_, FuncIL_, IlSize_);
  // Extract kernel function information
  bool Res = parseSPIR(BinaryData_, NumWords, FuncInfos_);
  delete[] BinaryData_;
  if (!Res) {
    CHIPERR_LOG_AND_THROW("SPIR-V parsing failed", hipErrorUnknown);
  }
}

CHIPModule::CHIPModule(std::string *ModuleStr) {
  Src_ = *ModuleStr;
  consumeSPIRV();
}
CHIPModule::CHIPModule(std::string &&ModuleStr) {
  Src_ = ModuleStr;
  consumeSPIRV();
}
CHIPModule::~CHIPModule() {}

void CHIPModule::addKernel(CHIPKernel *Kernel) {
  ChipKernels_.push_back(Kernel);
}

void CHIPModule::compileOnce(CHIPDevice *ChipDevice) {
  std::call_once(Compiled_, &CHIPModule::compile, this, ChipDevice);
}

CHIPKernel *CHIPModule::findKernel(const std::string &Name) {
  auto KernelFound = std::find_if(ChipKernels_.begin(), ChipKernels_.end(),
                                  [&Name](CHIPKernel *Kernel) {
                                    return Kernel->getName().compare(Name) == 0;
                                  });
  return KernelFound == ChipKernels_.end() ? nullptr : *KernelFound;
}

CHIPKernel *CHIPModule::getKernel(std::string Name) {
  auto *Kernel = findKernel(Name);
  if (!Kernel) {
    std::string Msg = "Failed to find kernel via kernel name: " + Name;
    CHIPERR_LOG_AND_THROW(Msg, hipErrorLaunchFailure);
  }
  return Kernel;
}

bool CHIPModule::hasKernel(std::string Name) { return findKernel(Name); }

CHIPKernel *CHIPModule::getKernel(const void *HostFPtr) {
  for (auto &Kernel : ChipKernels_)
    logDebug("chip kernel: {} {}", Kernel->getHostPtr(), Kernel->getName());
  auto KernelFound = std::find_if(ChipKernels_.begin(), ChipKernels_.end(),
                                  [HostFPtr](CHIPKernel *Kernel) {
                                    return Kernel->getHostPtr() == HostFPtr;
                                  });
  if (KernelFound == ChipKernels_.end()) {
    std::string Msg = "Failed to find kernel via host pointer";
    CHIPERR_LOG_AND_THROW(Msg, hipErrorLaunchFailure);
  }

  return *KernelFound;
}

std::vector<CHIPKernel *> &CHIPModule::getKernels() { return ChipKernels_; }

CHIPDeviceVar *CHIPModule::getGlobalVar(const char *VarName) {
  auto VarFound = std::find_if(ChipVars_.begin(), ChipVars_.end(),
                               [VarName](CHIPDeviceVar *Var) {
                                 return Var->getName().compare(VarName) == 0;
                               });
  if (VarFound == ChipVars_.end()) {
    logDebug("Failed to find global variable by name: {}",
             std::string(VarName));
    return nullptr;
  }

  return *VarFound;
}

hipError_t CHIPModule::allocateDeviceVariablesNoLock(CHIPDevice *Device,
                                                     CHIPQueue *Queue) {
  // Mark as allocated if the module does not have any variables.
  DeviceVariablesAllocated_ |= ChipVars_.empty();

  if (DeviceVariablesAllocated_)
    return hipSuccess;

  logTrace("Allocate storage for device variables in module: {}", (void *)this);

  // TODO: catch any exception and abort as it's probably an unrecoverable
  //       condition?

  size_t VarInfoBufSize = sizeof(CHIPVarInfo) * ChipVars_.size();
  auto *Ctx = Device->getContext();
  CHIPVarInfo *VarInfoBufD = (CHIPVarInfo *)Ctx->allocate(
      VarInfoBufSize, hipMemoryType::hipMemoryTypeUnified);
  assert(VarInfoBufD && "Could not allocate space for a shadow kernel.");
  auto VarInfoBufH = std::make_unique<CHIPVarInfo[]>(ChipVars_.size());

  // Gather information for storage allocation.
  std::vector<std::pair<CHIPDeviceVar *, CHIPVarInfo *>> VarInfos;
  for (auto *Var : ChipVars_) {
    auto I = VarInfos.size();
    queueVariableInfoShadowKernel(Queue, this, Var, &VarInfoBufD[I]);
    VarInfos.push_back(std::make_pair(Var, &VarInfoBufH[I]));
  }
  Queue->memCopyAsync(VarInfoBufH.get(), VarInfoBufD, VarInfoBufSize);
  Queue->finish();

  // Allocate storage for the device variables.
  for (auto &VarInfo : VarInfos) {
    size_t Size = (*VarInfo.second)[0];
    size_t Alignment = (*VarInfo.second)[1];
    size_t HasInitializer = (*VarInfo.second)[2];
    assert(Size && "Unexpected zero sized device variable.");
    assert(Alignment && "Unexpected alignment requirement.");

    auto *Var = VarInfo.first;
    Var->setDevAddr(
        Ctx->allocate(Size, Alignment, hipMemoryType::hipMemoryTypeUnified));
    Var->markHasInitializer(HasInitializer);
    // Sanity check for object sizes reported by the shadow kernels vs
    // __hipRegisterVar.
    assert(Var->getSize() == Size && "Object size discrepancy!");
    queueVariableBindShadowKernel(Queue, this, Var);
  }
  Queue->finish();
  DeviceVariablesAllocated_ = true;

  return hipSuccess;
}

void CHIPModule::initializeDeviceVariablesNoLock(CHIPDevice *Device,
                                                 CHIPQueue *Queue) {
  auto Err = allocateDeviceVariablesNoLock(Device, Queue);
  (void)Err;

  // Mark initialized if the module does not have any device variables.
  auto *NonSymbolResetKernel = findKernel(ChipNonSymbolResetKernelName);
  DeviceVariablesInitialized_ |= ChipVars_.empty() && !NonSymbolResetKernel;

  if (DeviceVariablesInitialized_) {
    // Can't be initialized if no storage is not allocated.
    assert(DeviceVariablesAllocated_ && "Should have storage.");
    return;
  }

  logTrace("Initialize device variables in module: {}", (void *)this);

  bool QueuedKernels = false;
  for (auto *Var : ChipVars_) {
    if (!Var->hasInitializer())
      continue;
    queueVariableInitShadowKernel(Queue, this, Var);
    QueuedKernels = true;
  }

  // Launch kernel for resetting host-inaccessible global device variables.
  if (NonSymbolResetKernel) {
    queueKernel(Queue, NonSymbolResetKernel);
    QueuedKernels = true;
  }

  if (QueuedKernels)
    Queue->finish();

  DeviceVariablesInitialized_ = true;
}

void CHIPModule::invalidateDeviceVariablesNoLock() {
  DeviceVariablesInitialized_ = false;
}

void CHIPModule::deallocateDeviceVariablesNoLock(CHIPDevice *Device) {
  invalidateDeviceVariablesNoLock();
  for (auto *Var : ChipVars_) {
    auto Err = Device->getContext()->free(Var->getDevAddr());
    (void)Err;
    Var->setDevAddr(nullptr);
  }
  DeviceVariablesAllocated_ = false;
}

OCLFuncInfo *CHIPModule::findFunctionInfo(const std::string &FName) {
  return FuncInfos_.count(FName) ? FuncInfos_.at(FName).get() : nullptr;
}

// CHIPKernel
//*************************************************************************************
CHIPKernel::CHIPKernel(std::string HostFName, OCLFuncInfo *FuncInfo)
    : HostFName_(HostFName), FuncInfo_(FuncInfo) {}
CHIPKernel::~CHIPKernel(){};
std::string CHIPKernel::getName() { return HostFName_; }
const void *CHIPKernel::getHostPtr() { return HostFPtr_; }
const void *CHIPKernel::getDevPtr() { return DevFPtr_; }

OCLFuncInfo *CHIPKernel::getFuncInfo() { return FuncInfo_; }

void CHIPKernel::setName(std::string HostFName) { HostFName_ = HostFName; }
void CHIPKernel::setHostPtr(const void *HostFPtr) { HostFPtr_ = HostFPtr; }
void CHIPKernel::setDevPtr(const void *DevFPtr) { DevFPtr_ = DevFPtr; }

// CHIPExecItem
//*************************************************************************************
CHIPExecItem::CHIPExecItem(dim3 GridDim, dim3 BlockDim, size_t SharedMem,
                           hipStream_t ChipQueue)
    : SharedMem_(SharedMem), GridDim_(GridDim), BlockDim_(BlockDim),
      ChipQueue_(ChipQueue){};
CHIPExecItem::~CHIPExecItem(){};

std::vector<uint8_t> CHIPExecItem::getArgData() { return ArgData_; }

void CHIPExecItem::setArg(const void *Arg, size_t Size, size_t Offset) {
  if ((Offset + Size) > ArgData_.size())
    ArgData_.resize(Offset + Size + 1024);

  std::memcpy(ArgData_.data() + Offset, Arg, Size);
  logDebug("CHIPExecItem.setArg() on {} size {} offset {}\n", (void *)this,
           Size, Offset);
  OffsetSizes_.push_back(std::make_tuple(Offset, Size));
}

dim3 CHIPExecItem::getBlock() { return BlockDim_; }
dim3 CHIPExecItem::getGrid() { return GridDim_; }
CHIPKernel *CHIPExecItem::getKernel() { return ChipKernel_; }
size_t CHIPExecItem::getSharedMem() { return SharedMem_; }
CHIPQueue *CHIPExecItem::getQueue() { return ChipQueue_; }
// CHIPDevice
//*************************************************************************************
CHIPDevice::CHIPDevice(CHIPContext *Ctx, int DeviceIdx)
    : Ctx_(Ctx), Idx_(DeviceIdx) {
  LegacyDefaultQueue = nullptr;
  PerThreadDefaultQueue = nullptr;
}

CHIPDevice::~CHIPDevice() {}
CHIPQueue *CHIPDevice::getLegacyDefaultQueue() {
  assert(LegacyDefaultQueue);
  return LegacyDefaultQueue.get();
}

CHIPQueue *CHIPDevice::getDefaultQueue() {
#ifdef HIP_API_PER_THREAD_DEFAULT_STREAM
  return getPerThreadDefaultQueue();
#else
  return getLegacyDefaultQueue();
#endif
}

CHIPQueue *CHIPDevice::getPerThreadDefaultQueue() {
  if (!PerThreadDefaultQueue.get()) {
    logDebug("PerThreadDefaultQueue is null.. Creating a new queue.");
    PerThreadDefaultQueue =
        std::unique_ptr<CHIPQueue>(Backend->createCHIPQueue(this));
    PerThreadStreamUsed = true;
  }

  return PerThreadDefaultQueue.get();
}

std::vector<CHIPKernel *> CHIPDevice::getKernels() {
  std::vector<CHIPKernel *> ChipKernels;
  for (auto ModuleIt : ChipModules) {
    auto *Module = ModuleIt.second;
    for (CHIPKernel *Kernel : Module->getKernels())
      ChipKernels.push_back(Kernel);
  }
  return ChipKernels;
}

std::unordered_map<const std::string *, CHIPModule *> &
CHIPDevice::getModules() {
  return ChipModules;
}

std::string CHIPDevice::getName() { return std::string(HipDeviceProps_.name); }

void CHIPDevice::init() {
  std::call_once(PropsPopulated_, &CHIPDevice::populateDevicePropertiesImpl,
                 this);
  if (!AllocationTracker)
    AllocationTracker = new CHIPAllocationTracker(
        HipDeviceProps_.totalGlobalMem, HipDeviceProps_.name);

  CHIPQueueFlags Flags;
  int Priority = 1; // TODO : set a default
  auto ChipQueue = createQueue(Flags, Priority);
  LegacyDefaultQueue = std::unique_ptr<CHIPQueue>(ChipQueue);
}

void CHIPDevice::copyDeviceProperties(hipDeviceProp_t *Prop) {
  logDebug("CHIPDevice->copy_device_properties()");
  if (Prop)
    std::memcpy(Prop, &this->HipDeviceProps_, sizeof(hipDeviceProp_t));
}

CHIPKernel *CHIPDevice::findKernelByHostPtr(const void *HostPtr) {
  logDebug("CHIPDevice::findKernelByHostPtr({})", HostPtr);
  std::vector<CHIPKernel *> ChipKernels = getKernels();
  if (ChipKernels.size() == 0) {
    CHIPERR_LOG_AND_THROW("chip_kernels is empty for this device",
                          hipErrorLaunchFailure);
  }
  logDebug("Listing Kernels for device {}", getName());
  for (auto &Kernel : ChipKernels) {
    logDebug("Kernel name: {} host_f_ptr: {}", Kernel->getName(),
             Kernel->getHostPtr());
  }

  auto KernelFound = std::find_if(ChipKernels.begin(), ChipKernels.end(),
                                  [&HostPtr](CHIPKernel *Kernel) {
                                    return Kernel->getHostPtr() == HostPtr;
                                  });

  if (KernelFound == ChipKernels.end()) {
    std::string Msg =
        "Tried to find kernel by host pointer but kernel was not found";
    CHIPERR_LOG_AND_THROW(Msg, hipErrorLaunchFailure);
  } else {
    logDebug("Found kernel {} with host pointer {}", (*KernelFound)->getName(),
             (*KernelFound)->getHostPtr());
  }

  return *KernelFound;
}

CHIPContext *CHIPDevice::getContext() { return Ctx_; }
int CHIPDevice::getDeviceId() { return Idx_; }

CHIPDeviceVar *CHIPDevice::getStatGlobalVar(const void *HostPtr) {
  if (DeviceVarLookup_.count(HostPtr)) {
    auto *Var = DeviceVarLookup_[HostPtr];
    assert(Var->getDevAddr() && "Missing device pointer.");
    return Var;
  }
  return nullptr;
}

CHIPDeviceVar *CHIPDevice::getGlobalVar(const void *HostPtr) {
  if (auto *Found = getDynGlobalVar(HostPtr))
    return Found;

  if (auto *Found = getStatGlobalVar(HostPtr))
    return Found;

  return nullptr;
}

int CHIPDevice::getAttr(hipDeviceAttribute_t Attr) {
  hipDeviceProp_t Prop = {};
  copyDeviceProperties(&Prop);

  switch (Attr) {
  case hipDeviceAttributeMaxThreadsPerBlock:
    return Prop.maxThreadsPerBlock;
    break;
  case hipDeviceAttributeMaxBlockDimX:
    return Prop.maxThreadsDim[0];
    break;
  case hipDeviceAttributeMaxBlockDimY:
    return Prop.maxThreadsDim[1];
    break;
  case hipDeviceAttributeMaxBlockDimZ:
    return Prop.maxThreadsDim[2];
    break;
  case hipDeviceAttributeMaxGridDimX:
    return Prop.maxGridSize[0];
    break;
  case hipDeviceAttributeMaxGridDimY:
    return Prop.maxGridSize[1];
    break;
  case hipDeviceAttributeMaxGridDimZ:
    return Prop.maxGridSize[2];
    break;
  case hipDeviceAttributeMaxSharedMemoryPerBlock:
    return Prop.sharedMemPerBlock;
    break;
  case hipDeviceAttributeTotalConstantMemory:
    return Prop.totalConstMem;
    break;
  case hipDeviceAttributeWarpSize:
    return Prop.warpSize;
    break;
  case hipDeviceAttributeMaxRegistersPerBlock:
    return Prop.regsPerBlock;
    break;
  case hipDeviceAttributeClockRate:
    return Prop.clockRate;
    break;
  case hipDeviceAttributeMemoryClockRate:
    return Prop.memoryClockRate;
    break;
  case hipDeviceAttributeMemoryBusWidth:
    return Prop.memoryBusWidth;
    break;
  case hipDeviceAttributeMultiprocessorCount:
    return Prop.multiProcessorCount;
    break;
  case hipDeviceAttributeComputeMode:
    return Prop.computeMode;
    break;
  case hipDeviceAttributeL2CacheSize:
    return Prop.l2CacheSize;
    break;
  case hipDeviceAttributeMaxThreadsPerMultiProcessor:
    return Prop.maxThreadsPerMultiProcessor;
    break;
  case hipDeviceAttributeComputeCapabilityMajor:
    return Prop.major;
    break;
  case hipDeviceAttributeComputeCapabilityMinor:
    return Prop.minor;
    break;
  case hipDeviceAttributePciBusId:
    return Prop.pciBusID;
    break;
  case hipDeviceAttributeConcurrentKernels:
    return Prop.concurrentKernels;
    break;
  case hipDeviceAttributePciDeviceId:
    return Prop.pciDeviceID;
    break;
  case hipDeviceAttributeMaxSharedMemoryPerMultiprocessor:
    return Prop.maxSharedMemoryPerMultiProcessor;
    break;
  case hipDeviceAttributeIsMultiGpuBoard:
    return Prop.isMultiGpuBoard;
    break;
  case hipDeviceAttributeCooperativeLaunch:
    return Prop.cooperativeLaunch;
    break;
  case hipDeviceAttributeCooperativeMultiDeviceLaunch:
    return Prop.cooperativeMultiDeviceLaunch;
    break;
  case hipDeviceAttributeIntegrated:
    return Prop.integrated;
    break;
  case hipDeviceAttributeMaxTexture1DWidth:
    return Prop.maxTexture1D;
    break;
  case hipDeviceAttributeMaxTexture1DLinear:
    return Prop.maxTexture1DLinear;
    break;
  case hipDeviceAttributeMaxTexture2DWidth:
    return Prop.maxTexture2D[0];
    break;
  case hipDeviceAttributeMaxTexture2DHeight:
    return Prop.maxTexture2D[1];
    break;
    // For some reason HIP has single attribute for pitched memory
    // instead of having separate attribute for width, height and
    // pitch dimensions, like on Cuda.
    //
    // hipDeviceProp_t does not have maxTexture2DLinear like in
    // Cuda. Use maxTexture2D.
  case hipDeviceAttributeMaxTexture2DLinear:
    return std::min<int>(Prop.maxTexture2D[0], Prop.maxTexture2D[1]);
    break;
  case hipDeviceAttributeMaxTexture3DWidth:
    return Prop.maxTexture3D[0];
    break;
  case hipDeviceAttributeMaxTexture3DHeight:
    return Prop.maxTexture3D[1];
    break;
  case hipDeviceAttributeMaxTexture3DDepth:
    return Prop.maxTexture3D[2];
    break;
  case hipDeviceAttributeHdpMemFlushCntl:
  case hipDeviceAttributeHdpRegFlushCntl:
    UNIMPLEMENTED(-1);
  case hipDeviceAttributeMaxPitch:
    return Prop.memPitch;
    break;
  case hipDeviceAttributeTextureAlignment:
    return Prop.textureAlignment;
    break;
  case hipDeviceAttributeTexturePitchAlignment:
    return Prop.texturePitchAlignment;
    break;
  case hipDeviceAttributeKernelExecTimeout:
    return Prop.kernelExecTimeoutEnabled;
    break;
  case hipDeviceAttributeCanMapHostMemory:
    return Prop.canMapHostMemory;
    break;
  case hipDeviceAttributeEccEnabled:
    return Prop.ECCEnabled;
    break;
  case hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc:
    return Prop.cooperativeMultiDeviceUnmatchedFunc;
    break;
  case hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim:
    return Prop.cooperativeMultiDeviceUnmatchedGridDim;
    break;
  case hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim:
    return Prop.cooperativeMultiDeviceUnmatchedBlockDim;
    break;
  case hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem:
    return Prop.cooperativeMultiDeviceUnmatchedSharedMem;
    break;
  case hipDeviceAttributeAsicRevision:
    return Prop.asicRevision;
    break;
  case hipDeviceAttributeManagedMemory:
    return Prop.managedMemory;
    break;
  case hipDeviceAttributeDirectManagedMemAccessFromHost:
    return Prop.directManagedMemAccessFromHost;
    break;
  case hipDeviceAttributeConcurrentManagedAccess:
    return Prop.concurrentManagedAccess;
    break;
  case hipDeviceAttributePageableMemoryAccess:
    return Prop.pageableMemoryAccess;
    break;
  case hipDeviceAttributePageableMemoryAccessUsesHostPageTables:
    return Prop.pageableMemoryAccessUsesHostPageTables;
    break;
  case hipDeviceAttributeCanUseStreamWaitValue:
    // hipStreamWaitValue64() and hipStreamWaitValue32() support
    // return g_devices[device]->devices()[0]->info().aqlBarrierValue_;
    CHIPERR_LOG_AND_THROW(
        "CHIPDevice::getAttr(hipDeviceAttributeCanUseStreamWaitValue path "
        "unimplemented",
        hipErrorTbd);
    break;
  default:
    CHIPERR_LOG_AND_THROW("CHIPDevice::getAttr asked for an unkown attribute",
                          hipErrorInvalidValue);
  }
  return -1;
}

size_t CHIPDevice::getGlobalMemSize() { return HipDeviceProps_.totalGlobalMem; }

void CHIPDevice::addModule(const std::string *ModuleStr, CHIPModule *Module) {
  this->ChipModules[ModuleStr] = Module;
}

void CHIPDevice::registerFunctionAsKernel(std::string *ModuleStr,
                                          const void *HostFPtr,
                                          const char *HostFName) {
  CHIPModule *Module = nullptr;
  if (ChipModules.count(ModuleStr)) {
    Module = ChipModules[ModuleStr];
  } else {
    Module = addModule(ModuleStr);
    Module->compileOnce(this);
  }

  CHIPKernel *Kernel = Module->getKernel(std::string(HostFName));
  if (!Kernel) {
    std::string Msg = "Device " + getName() +
                      " tried to register host function " + HostFName +
                      " but failed to find kernel with a matching name";
    CHIPERR_LOG_AND_THROW(Msg, hipErrorLaunchFailure);
  }

  Kernel->setHostPtr(HostFPtr);

  logDebug("Device {}: successfully registered function {} as kernel {}",
           getName(), HostFName, Kernel->getName().c_str());
  return;
}

void CHIPDevice::registerDeviceVariable(std::string *ModuleStr,
                                        const void *HostPtr, const char *Name,
                                        size_t Size) {
  if (!ChipModules.count(ModuleStr)) {
    auto *NewModule = addModule(ModuleStr);
    NewModule->compileOnce(this);
  }
  if (!ChipModules.count(ModuleStr))
    CHIPERR_LOG_AND_THROW(
        "Broken expectation: could not find a module by name,", hipErrorTbd);
  CHIPModule *Module = ChipModules[ModuleStr];

  std::string VarInfoKernelName = std::string(ChipVarInfoPrefix) + Name;
  if (!Module->hasKernel(VarInfoKernelName)) {
    // The kernel compilation pipe is allowed to remove device-side unused
    // global variables from the device modules. This is utilized in the abort
    // implementation to signal that abort is not called in the module. The
    // lack of the variable in the device module is used as a quick (and dirty)
    // way to not query for the global flag value after each kernel execution
    // (reading of which requires kernel launches).
    logTrace("Device variable {} not found in the module -- removed as unused?",
             Name);
    return;
  }

  auto *Var = new CHIPDeviceVar(Name, Size);
  Module->addDeviceVariable(Var);
  DeviceVarLookup_.insert(std::make_pair(HostPtr, Var));
}

void CHIPDevice::addQueue(CHIPQueue *ChipQueue) {
  std::lock_guard<std::mutex> LockDevice(Backend->BackendMtx);

  auto QueueFound =
      std::find(ChipQueues_.begin(), ChipQueues_.end(), ChipQueue);
  if (QueueFound == ChipQueues_.end()) {
    ChipQueues_.push_back(ChipQueue);
  } else {
    CHIPERR_LOG_AND_THROW("Tried to add a queue to the backend which was "
                          "already present in the backend queue list",
                          hipErrorTbd);
  }
  logDebug("CHIPQueue {} added to the queue vector for device {} ",
           (void *)ChipQueue, (void *)this);

  return;
}

void CHIPEvent::track() {
  std::lock_guard<std::mutex> LockBackend(Backend->EventsMtx);
  std::lock_guard<std::mutex> Lock(EventMtx);
  if (!TrackCalled_) {
    trackImpl();
    TrackCalled_ = true;
  }
}

void CHIPEvent::trackImpl() {
  Backend->Events.push_back(this);
}

CHIPQueue *CHIPDevice::createQueueAndRegister(CHIPQueueFlags Flags,
                                              int Priority) {

  auto ChipQueue = createQueue(Flags, Priority);
  // Add the queue handle to the device and the Backend
  addQueue(ChipQueue);
  Backend->addQueue(ChipQueue);
  return ChipQueue;
}

CHIPQueue *CHIPDevice::createQueueAndRegister(const uintptr_t *NativeHandles,
                                              const size_t NumHandles) {
  auto ChipQueue = createQueue(NativeHandles, NumHandles);
  // Add the queue handle to the device and the Backend
  addQueue(ChipQueue);
  Backend->addQueue(ChipQueue);
  return ChipQueue;
}

std::vector<CHIPQueue *> &CHIPDevice::getQueues() {
  std::lock_guard<std::mutex> LockDevice(DeviceMtx);
  return ChipQueues_;
}

hipError_t CHIPDevice::setPeerAccess(CHIPDevice *Peer, int Flags,
                                     bool CanAccessPeer) {
  UNIMPLEMENTED(hipSuccess);
}

int CHIPDevice::getPeerAccess(CHIPDevice *PeerDevice) { UNIMPLEMENTED(0); }

void CHIPDevice::setCacheConfig(hipFuncCache_t Cfg) { UNIMPLEMENTED(); }

void CHIPDevice::setFuncCacheConfig(const void *Func, hipFuncCache_t Cfg) {
  UNIMPLEMENTED();
}

hipFuncCache_t CHIPDevice::getCacheConfig() {
  UNIMPLEMENTED(hipFuncCachePreferNone);
}

hipSharedMemConfig CHIPDevice::getSharedMemConfig() {
  UNIMPLEMENTED(hipSharedMemBankSizeDefault);
}

bool CHIPDevice::removeQueue(CHIPQueue *ChipQueue) {
  std::lock_guard<std::mutex> LockBackend(Backend->BackendMtx);
  // If this stream has a LastEvent, it will release it, decrement its refcount
  // and let the StaleEventMonitor to collect it
  ChipQueue->updateLastEvent(nullptr);

  // Remove from device queue list
  auto FoundQueue =
      std::find(ChipQueues_.begin(), ChipQueues_.end(), ChipQueue);
  if (FoundQueue == ChipQueues_.end()) {
    std::string Msg =
        "Tried to remove a queue for a device but the queue was not found in "
        "device queue list";
    CHIPERR_LOG_AND_THROW(Msg, hipErrorUnknown);
  }
  ChipQueues_.erase(FoundQueue);

  // Remove from the Backend Queue List
  FoundQueue = std::find(Backend->getQueues().begin(),
                         Backend->getQueues().end(), ChipQueue);
  if (FoundQueue == Backend->getQueues().end()) {
    std::string Msg = "Tried to remove a queue for a the backend but the queue "
                      "was not found in "
                      "backend queue list";
    CHIPERR_LOG_AND_THROW(Msg, hipErrorUnknown);
  }

  Backend->getQueues().erase(FoundQueue);
  return true;
}

void CHIPDevice::setSharedMemConfig(hipSharedMemConfig Cfg) { UNIMPLEMENTED(); }

size_t CHIPDevice::getUsedGlobalMem() {
  return AllocationTracker->TotalMemSize;
}

bool CHIPDevice::hasPCIBusId(int PciDomainID, int PciBusID, int PciDeviceID) {
  auto T1 = this->HipDeviceProps_.pciBusID == PciBusID;
  auto T2 = this->HipDeviceProps_.pciDomainID == PciDomainID;
  auto T3 = this->HipDeviceProps_.pciDeviceID == PciDeviceID;

  return (T1 && T2 && T3);
}

hipError_t CHIPDevice::allocateDeviceVariables() {
  std::lock_guard<std::mutex> Lock(DeviceMtx);
  logTrace("Allocate storage for device variables.");
  for (auto I : ChipModules) {
    auto Status =
        I.second->allocateDeviceVariablesNoLock(this, getDefaultQueue());
    if (Status != hipSuccess)
      return Status;
  }
  return hipSuccess;
}

void CHIPDevice::initializeDeviceVariables() {
  std::lock_guard<std::mutex> Lock(DeviceMtx);
  logTrace("Initialize device variables.");
  for (auto Module : ChipModules)
    Module.second->initializeDeviceVariablesNoLock(this, getDefaultQueue());
}

void CHIPDevice::invalidateDeviceVariables() {
  std::lock_guard<std::mutex> Lock(DeviceMtx);
  logTrace("invalidate device variables.");
  for (auto Module : ChipModules)
    Module.second->invalidateDeviceVariablesNoLock();
}

void CHIPDevice::deallocateDeviceVariables() {
  std::lock_guard<std::mutex> Lock(DeviceMtx);
  logTrace("Deallocate storage for device variables.");
  for (auto Module : ChipModules)
    Module.second->deallocateDeviceVariablesNoLock(this);
}

// CHIPContext
//*************************************************************************************
CHIPContext::CHIPContext() {}
CHIPContext::~CHIPContext() {}

void CHIPContext::syncQueues(CHIPQueue *TargetQueue) {
  auto DefaultQueue = Backend->getActiveDevice()->getDefaultQueue();
#ifdef HIP_API_PER_THREAD_DEFAULT_STREAM
  // The per-thread default stream is an implicit stream local to both the
  // thread and the CUcontext, and which does not synchronize with other streams
  // (just like explcitly created streams). The per-thread default stream is not
  // a non-blocking stream and will synchronize with the legacy default stream
  // if both are used in a program.

  // since HIP_API_PER_THREAD_DEFAULT_STREAM is enabled, there is no legacy
  // default stream thus no syncronization necessary
  if (TargetQueue == DefaultQueue)
    return;
#endif
  std::lock_guard<std::mutex> LockContext(ContextMtx);
  std::vector<CHIPQueue *> QueuesToSyncWith;

  // The per-thread default stream is not a non-blocking stream and will
  // synchronize with the legacy default stream if both are used in a program
  if (Backend->getActiveDevice()->PerThreadStreamUsed) {
    if (TargetQueue == Backend->getActiveDevice()->getPerThreadDefaultQueue())
      QueuesToSyncWith.push_back(DefaultQueue);
    else if (TargetQueue == Backend->getActiveDevice()->getLegacyDefaultQueue())
      QueuesToSyncWith.push_back(
          Backend->getActiveDevice()->getPerThreadDefaultQueue());
  }

  {
    std::lock_guard<std::mutex> LockBackend(Backend->BackendMtx);
    // Always sycn with all blocking queues
    for (auto &Queue : Backend->getQueues())
      if (Queue->getQueueFlags().isBlocking())
        QueuesToSyncWith.push_back(Queue);
  }

  // default stream waits on all blocking streams to complete
  std::vector<CHIPEvent *> EventsToWaitOn;

  CHIPEvent *SyncQueuesEvent;
  if (TargetQueue == DefaultQueue) {
    for (auto &q : QueuesToSyncWith) {
      auto Ev = q->getLastEvent();
      if (Ev)
        EventsToWaitOn.push_back(Ev);
    }
    std::lock_guard<std::mutex> LockQueue(TargetQueue->QueueMtx);
    SyncQueuesEvent = TargetQueue->enqueueBarrierImpl(&EventsToWaitOn);
    SyncQueuesEvent->Msg = "barrierSyncQueue";
    TargetQueue->updateLastEvent(SyncQueuesEvent);
  } else { // blocking stream must wait until default stream is done
    auto Ev = DefaultQueue->getLastEvent();
    if (Ev)
      EventsToWaitOn.push_back(Ev);
    std::lock_guard<std::mutex> LockQueue(TargetQueue->QueueMtx);
    SyncQueuesEvent = TargetQueue->enqueueBarrierImpl(&EventsToWaitOn);
    SyncQueuesEvent->Msg = "barrierSyncQueue";
    TargetQueue->updateLastEvent(SyncQueuesEvent);
  }
  SyncQueuesEvent->track();
}

void CHIPContext::addDevice(CHIPDevice *ChipDevice) {
  logDebug("CHIPContext.add_device() {}", ChipDevice->getName());
  std::lock_guard<std::mutex> LockContext(ContextMtx);
  ChipDevices_.push_back(ChipDevice);
}

std::vector<CHIPDevice *> &CHIPContext::getDevices() {
  if (ChipDevices_.size() == 0)
    logWarn("CHIPContext.get_devices() was called but chip_devices is empty");
  return ChipDevices_;
}

void *CHIPContext::allocate(size_t Size, hipMemoryType MemType) {
  return allocate(Size, 0, MemType, CHIPHostAllocFlags());
}

void *CHIPContext::allocate(size_t Size, size_t Alignment,
                            hipMemoryType MemType) {
  return allocate(Size, Alignment, MemType, CHIPHostAllocFlags());
}

void *CHIPContext::allocate(size_t Size, size_t Alignment,
                            hipMemoryType MemType, CHIPHostAllocFlags Flags) {
  std::lock_guard<std::mutex> Lock(ContextMtx);
  void *AllocatedPtr, *HostPtr = nullptr;

  if (!Flags.isDefault()) {
    if (Flags.isMapped())
      MemType = hipMemoryType::hipMemoryTypeHost;
    if (Flags.isCoherent())
      UNIMPLEMENTED(nullptr);
    if (Flags.isNonCoherent())
      UNIMPLEMENTED(nullptr);
    if (Flags.isNumaUser())
      UNIMPLEMENTED(nullptr);
    if (Flags.isPortable())
      UNIMPLEMENTED(nullptr);
  }

  CHIPDevice *ChipDev = Backend->getActiveDevice();
  assert(ChipDev->getContext() == this);

  assert(ChipDev->AllocationTracker && "AllocationTracker was not created!");
  if (!ChipDev->AllocationTracker->reserveMem(Size))
    return nullptr;
  AllocatedPtr = allocateImpl(Size, Alignment, MemType);
  if (AllocatedPtr == nullptr)
    ChipDev->AllocationTracker->releaseMemReservation(Size);

  if (MemType == hipMemoryTypeUnified || isAllocatedPtrUSM(AllocatedPtr)) {
    HostPtr = AllocatedPtr;
    MemType = hipMemoryTypeUnified;
  }
  ChipDev->AllocationTracker->recordAllocation(
      AllocatedPtr, HostPtr, ChipDev->getDeviceId(), Size, Flags, MemType);

  return AllocatedPtr;
}

unsigned int CHIPContext::getFlags() { return Flags_; }

void CHIPContext::setFlags(unsigned int Flags) { Flags_ = Flags; }

void CHIPContext::reset() {
  logDebug("Resetting CHIPContext: deleting allocations");
  // Free all allocations in this context
  for (auto &Ptr : AllocatedPtrs_)
    freeImpl(Ptr);
  // Free all the memory reservations on each device
  for (auto &Dev : ChipDevices_)
    Dev->AllocationTracker->releaseMemReservation(
        Dev->AllocationTracker->TotalMemSize);
  AllocatedPtrs_.clear();

  for (auto *Dev : ChipDevices_)
    Dev->reset();

  // TODO Is all the state reset?
}

CHIPContext *CHIPContext::retain() { UNIMPLEMENTED(nullptr); }

hipError_t CHIPContext::free(void *Ptr) {
  CHIPDevice *ChipDev = Backend->getActiveDevice();
  std::lock_guard<std::mutex> LockDevice(ChipDev->DeviceMtx);
  AllocationInfo *AllocInfo = ChipDev->AllocationTracker->getAllocInfo(Ptr);
  if (!AllocInfo)
    return hipErrorInvalidDevicePointer;

  ChipDev->AllocationTracker->releaseMemReservation(AllocInfo->Size);
  ChipDev->AllocationTracker->eraseRecord(AllocInfo);
  freeImpl(Ptr);
  return hipSuccess;
}

// CHIPBackend
//*************************************************************************************
int CHIPBackend::getQueuePriorityRange() {
  assert(MinQueuePriority_);
  return MinQueuePriority_;
}

void CHIPBackend::uninitialize() { logDebug("CHIPBackend::uninitialize()"); }

std::string CHIPBackend::getJitFlags() {
  std::string Flags;
  if (CustomJitFlags != "") {
    Flags = CustomJitFlags;
  } else {
    Flags = getDefaultJitFlags();
  }
  logDebug("JIT compiler flags: {}", Flags);
  return Flags;
}

CHIPBackend::CHIPBackend() { logDebug("CHIPBackend Base Constructor"); };
CHIPBackend::~CHIPBackend() {
  logDebug("CHIPBackend Destructor. Deleting all pointers.");
  while (!ChipExecStack.empty())
    ChipExecStack.pop();
  while (!CallbackQueue.empty())
    CallbackQueue.pop();

  Events.clear();
  for (auto &Ctx : ChipContexts)
    delete Ctx;
  for (auto &Q : ChipQueues)
    delete Q;
  for (auto &Mod : ModulesStr_)
    delete Mod;
}

void CHIPBackend::initialize(std::string PlatformStr, std::string DeviceTypeStr,
                             std::string DeviceIdStr) {
  initializeImpl(PlatformStr, DeviceTypeStr, DeviceIdStr);
  CustomJitFlags = read_env_var("CHIP_JIT_FLAGS", false);
  if (ChipDevices.size() == 0) {
    std::string Msg = "No CHIPDevices were initialized";
    CHIPERR_LOG_AND_THROW(Msg, hipErrorInitializationError);
  }

  // check if all the devices had their default queues initialized
  for (auto Dev : ChipDevices) {
    if (Dev->LegacyDefaultQueue == nullptr)
      CHIPERR_LOG_AND_THROW("LegacyDefaultQueue not initialized",
                            hipErrorInitializationError);
    // if (Dev->PerThreadDefaultQueue == nullptr)
    //   CHIPERR_LOG_AND_THROW("PerThreadDefaultQueue not initialized",
    //                         hipErrorInitializationError);
  }
  setActiveDevice(ChipDevices[0]);
}

void CHIPBackend::setActiveDevice(CHIPDevice *ChipDevice) {
  std::lock_guard<std::mutex> LockSetActive(Backend->SetActiveMtx);

  auto DeviceFound =
      std::find(ChipDevices.begin(), ChipDevices.end(), ChipDevice);
  if (DeviceFound == ChipDevices.end()) {
    std::string Msg =
        "Tried to set active device with CHIPDevice pointer that is not in "
        "CHIPBackend::chip_devices";
    CHIPERR_LOG_AND_THROW(Msg, hipErrorLaunchFailure);
  };
  ActiveDev_ = ChipDevice;
  ActiveCtx_ = ChipDevice->getContext();
}
std::vector<CHIPQueue *> &CHIPBackend::getQueues() { return ChipQueues; }

CHIPContext *CHIPBackend::getActiveContext() {
  std::lock_guard<std::mutex> LockSetActive(Backend->SetActiveMtx);
  if (ActiveCtx_ == nullptr) {
    std::string Msg = "Active context is null";
    CHIPERR_LOG_AND_THROW(Msg, hipErrorUnknown);
  }
  return ActiveCtx_;
};

CHIPDevice *CHIPBackend::getActiveDevice() {
  std::lock_guard<std::mutex> LockSetActive(Backend->SetActiveMtx);
  if (ActiveDev_ == nullptr) {
    CHIPERR_LOG_AND_THROW(
        "CHIPBackend.getActiveDevice() was called but active_ctx is null",
        hipErrorUnknown);
  }
  return ActiveDev_;
};

std::vector<CHIPDevice *> &CHIPBackend::getDevices() {
  std::lock_guard<std::mutex> LockSetActive(Backend->SetActiveMtx);

  return ChipDevices;
}

size_t CHIPBackend::getNumDevices() { return ChipDevices.size(); }
std::vector<std::string *> &CHIPBackend::getModulesStr() { return ModulesStr_; }

void CHIPBackend::addContext(CHIPContext *ChipContext) {
  ChipContexts.push_back(ChipContext);
}
void CHIPBackend::addQueue(CHIPQueue *ChipQueue) {
  std::lock_guard<std::mutex> LockBackend(BackendMtx);
  auto QueueFound = std::find(ChipQueues.begin(), ChipQueues.end(), ChipQueue);
  if (QueueFound == ChipQueues.end()) {
    ChipQueues.push_back(ChipQueue);
  } else {
    CHIPERR_LOG_AND_THROW("Tried to add a queue to the backend which was "
                          "already present in the backend queue list",
                          hipErrorTbd);
  }

  logDebug("CHIPQueue {} added to the queue vector for backend {} ",
           (void *)ChipQueue, (void *)this);

  return;
}
void CHIPBackend::addDevice(CHIPDevice *ChipDevice) {
  logDebug("CHIPDevice.add_device() {}", ChipDevice->getName());
  std::lock_guard<std::mutex> LockBackend(BackendMtx);
  ChipDevices.push_back(ChipDevice);

  logDebug("CHIPDevice {} added to the queue vector for backend {} ",
           (void *)ChipDevice, (void *)this);
}

void CHIPBackend::registerModuleStr(std::string *ModuleStr) {
  logDebug("CHIPBackend->register_module()");
  std::lock_guard<std::mutex> LockBackend(BackendMtx);
  getModulesStr().push_back(ModuleStr);
}

void CHIPBackend::unregisterModuleStr(std::string *ModuleStr) {
  logDebug("CHIPBackend->unregister_module()");
  auto ModuleFound =
      std::find(ModulesStr_.begin(), ModulesStr_.end(), ModuleStr);
  if (ModuleFound != ModulesStr_.end()) {
    getModulesStr().erase(ModuleFound);
  } else {
    logWarn("Module {} not found in CHIPBackend.modules_str while trying to "
            "unregister",
            (void *)ModuleStr);
  }
}

hipError_t CHIPBackend::configureCall(dim3 Grid, dim3 Block, size_t SharedMem,
                                      hipStream_t ChipQueue) {
  std::lock_guard<std::mutex> LockBackend(BackendMtx);
  logDebug("CHIPBackend->configureCall(grid=({},{},{}), block=({},{},{}), "
           "shared={}, q={}",
           Grid.x, Grid.y, Grid.z, Block.x, Block.y, Block.z, SharedMem,
           (void *)ChipQueue);
  CHIPExecItem *ExecItem = new CHIPExecItem(Grid, Block, SharedMem, ChipQueue);
  ChipExecStack.push(ExecItem);

  return hipSuccess;
}

hipError_t CHIPBackend::setArg(const void *Arg, size_t Size, size_t Offset) {
  logDebug("CHIPBackend->set_arg()");
  std::lock_guard<std::mutex> LockBackend(BackendMtx);
  CHIPExecItem *ExecItem = ChipExecStack.top();
  ExecItem->setArg(Arg, Size, Offset);

  return hipSuccess;
}

/**
 * @brief Register this function as a kernel for all devices initialized in
 * this backend
 *
 * @param module_str
 * @param HostFunctionPtr
 * @param FunctionName
 * @return true
 * @return false
 */

bool CHIPBackend::registerFunctionAsKernel(std::string *ModuleStr,
                                           const void *HostFPtr,
                                           const char *HostFName) {
  logDebug("CHIPBackend.registerFunctionAsKernel()");
  for (auto &Ctx : ChipContexts)
    for (auto &Dev : Ctx->getDevices())
      Dev->registerFunctionAsKernel(ModuleStr, HostFPtr, HostFName);
  return true;
}

void CHIPBackend::registerDeviceVariable(std::string *ModuleStr,
                                         const void *HostPtr, const char *Name,
                                         size_t Size) {
  for (auto *Ctx : ChipContexts)
    for (auto *Dev : Ctx->getDevices())
      Dev->registerDeviceVariable(ModuleStr, HostPtr, Name, Size);
}

CHIPDevice *CHIPBackend::findDeviceMatchingProps(const hipDeviceProp_t *Props) {
  CHIPDevice *MatchedDevice = nullptr;
  int MaxMatchedCount = 0;
  for (auto &Dev : ChipDevices) {
    hipDeviceProp_t CurrentProp = {};
    Dev->copyDeviceProperties(&CurrentProp);
    int ValidPropCount = 0;
    int MatchedCount = 0;
    if (Props->major != 0) {
      ValidPropCount++;
      if (CurrentProp.major >= Props->major) {
        MatchedCount++;
      }
    }
    if (Props->minor != 0) {
      ValidPropCount++;
      if (CurrentProp.minor >= Props->minor) {
        MatchedCount++;
      }
    }
    if (Props->totalGlobalMem != 0) {
      ValidPropCount++;
      if (CurrentProp.totalGlobalMem >= Props->totalGlobalMem) {
        MatchedCount++;
      }
    }
    if (Props->sharedMemPerBlock != 0) {
      ValidPropCount++;
      if (CurrentProp.sharedMemPerBlock >= Props->sharedMemPerBlock) {
        MatchedCount++;
      }
    }
    if (Props->maxThreadsPerBlock != 0) {
      ValidPropCount++;
      if (CurrentProp.maxThreadsPerBlock >= Props->maxThreadsPerBlock) {
        MatchedCount++;
      }
    }
    if (Props->totalConstMem != 0) {
      ValidPropCount++;
      if (CurrentProp.totalConstMem >= Props->totalConstMem) {
        MatchedCount++;
      }
    }
    if (Props->multiProcessorCount != 0) {
      ValidPropCount++;
      if (CurrentProp.multiProcessorCount >= Props->multiProcessorCount) {
        MatchedCount++;
      }
    }
    if (Props->maxThreadsPerMultiProcessor != 0) {
      ValidPropCount++;
      if (CurrentProp.maxThreadsPerMultiProcessor >=
          Props->maxThreadsPerMultiProcessor) {
        MatchedCount++;
      }
    }
    if (Props->memoryClockRate != 0) {
      ValidPropCount++;
      if (CurrentProp.memoryClockRate >= Props->memoryClockRate) {
        MatchedCount++;
      }
    }
    if (Props->memoryBusWidth != 0) {
      ValidPropCount++;
      if (CurrentProp.memoryBusWidth >= Props->memoryBusWidth) {
        MatchedCount++;
      }
    }
    if (Props->l2CacheSize != 0) {
      ValidPropCount++;
      if (CurrentProp.l2CacheSize >= Props->l2CacheSize) {
        MatchedCount++;
      }
    }
    if (Props->regsPerBlock != 0) {
      ValidPropCount++;
      if (CurrentProp.regsPerBlock >= Props->regsPerBlock) {
        MatchedCount++;
      }
    }
    if (Props->maxSharedMemoryPerMultiProcessor != 0) {
      ValidPropCount++;
      if (CurrentProp.maxSharedMemoryPerMultiProcessor >=
          Props->maxSharedMemoryPerMultiProcessor) {
        MatchedCount++;
      }
    }
    if (Props->warpSize != 0) {
      ValidPropCount++;
      if (CurrentProp.warpSize >= Props->warpSize) {
        MatchedCount++;
      }
    }
    if (ValidPropCount == MatchedCount) {
      MatchedDevice = MatchedCount > MaxMatchedCount ? Dev : MatchedDevice;
      MaxMatchedCount = std::max(MatchedCount, MaxMatchedCount);
    }
  }
  return MatchedDevice;
}

CHIPQueue *CHIPBackend::findQueue(CHIPQueue *ChipQueue) {
  std::lock_guard<std::mutex> LockBackend(BackendMtx);

  if (ChipQueue == hipStreamPerThread) {
    return Backend->getActiveDevice()->getPerThreadDefaultQueue();
  } else if (ChipQueue == hipStreamLegacy) {
    return Backend->getActiveDevice()->getLegacyDefaultQueue();
  } else if (ChipQueue == nullptr) {
    return Backend->getActiveDevice()->getDefaultQueue();
  }

  // Safety Check to make sure that the requested queue is registereted
  std::vector<CHIPQueue *> AllQueues = Backend->getActiveDevice()->getQueues();
  AllQueues.push_back(Backend->getActiveDevice()->getLegacyDefaultQueue());

  if (Backend->getActiveDevice()->PerThreadStreamUsed)
    AllQueues.push_back(Backend->getActiveDevice()->getPerThreadDefaultQueue());

  auto QueueFound = std::find(AllQueues.begin(), AllQueues.end(), ChipQueue);
  if (QueueFound == AllQueues.end())
    CHIPERR_LOG_AND_THROW("CHIPBackend::findQueue() was given a non-nullptr "
                          "queue but this queue "
                          "was not found among the backend queues.",
                          hipErrorTbd);
  return *QueueFound;
}

// CHIPQueue
//*************************************************************************************
CHIPQueue::CHIPQueue(CHIPDevice *ChipDevice, CHIPQueueFlags Flags, int Priority)
    : Priority_(Priority), QueueFlags_(Flags), ChipDevice_(ChipDevice) {
  ChipContext_ = ChipDevice->getContext();
};
CHIPQueue::CHIPQueue(CHIPDevice *ChipDevice, CHIPQueueFlags Flags)
    : CHIPQueue(ChipDevice, Flags, 0){};
CHIPQueue::~CHIPQueue() {
  logDebug("~CHIPQueue()");
  updateLastEvent(nullptr);
};

///////// Enqueue Operations //////////
hipError_t CHIPQueue::memCopy(void *Dst, const void *Src, size_t Size) {
  // Scope this so that we release mutex for finish()
  CHIPEvent *ChipEvent;
  {
#ifdef ENFORCE_QUEUE_SYNC
    ChipContext_->syncQueues(this);
#endif
    std::lock_guard<std::mutex> Lock(QueueMtx);
    ChipEvent = memCopyAsyncImpl(Dst, Src, Size);
    ChipEvent->Msg = "memCopy";
    updateLastEvent(ChipEvent);
  }
  this->finish();
  ChipEvent->track();

  return hipSuccess;
}
void CHIPQueue::memCopyAsync(void *Dst, const void *Src, size_t Size) {
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif
  std::lock_guard<std::mutex> Lock(QueueMtx);
  auto ChipEvent = memCopyAsyncImpl(Dst, Src, Size);
  ChipEvent->Msg = "memCopyAsync";
  updateLastEvent(ChipEvent);
  ChipEvent->track();
  return;
}
void CHIPQueue::memFill(void *Dst, size_t Size, const void *Pattern,
                        size_t PatternSize) {
  {
#ifdef ENFORCE_QUEUE_SYNC
    ChipContext_->syncQueues(this);
#endif
    std::lock_guard<std::mutex> Lock(QueueMtx);

    auto ChipEvent = memFillAsyncImpl(Dst, Size, Pattern, PatternSize);
    ChipEvent->Msg = "memFill";
    updateLastEvent(ChipEvent);
    ChipEvent->track();
  }
  this->finish();
  return;
}

void CHIPQueue::memFillAsync(void *Dst, size_t Size, const void *Pattern,
                             size_t PatternSize) {
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif
  std::lock_guard<std::mutex> Lock(QueueMtx);

  auto ChipEvent = memFillAsyncImpl(Dst, Size, Pattern, PatternSize);
  ChipEvent->Msg = "memFillAsync";
  updateLastEvent(ChipEvent);
  ChipEvent->track();
}
void CHIPQueue::memCopy2D(void *Dst, size_t DPitch, const void *Src,
                          size_t SPitch, size_t Width, size_t Height) {
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif
  std::lock_guard<std::mutex> Lock(QueueMtx);

  auto ChipEvent = memCopy2DAsyncImpl(Dst, DPitch, Src, SPitch, Width, Height);
  ChipEvent->Msg = "memCopy2D";
  finish();
  updateLastEvent(ChipEvent);
  ChipEvent->track();
}

void CHIPQueue::memCopy2DAsync(void *Dst, size_t DPitch, const void *Src,
                               size_t SPitch, size_t Width, size_t Height) {
  {
    std::lock_guard<std::mutex> Lock(QueueMtx);
#ifdef ENFORCE_QUEUE_SYNC
    ChipContext_->syncQueues(this);
#endif
    auto ChipEvent =
        memCopy2DAsyncImpl(Dst, DPitch, Src, SPitch, Width, Height);
    ChipEvent->Msg = "memCopy2DAsync";
    updateLastEvent(ChipEvent);
    ChipEvent->track();
  }
  this->finish();
  return;
}

void CHIPQueue::memCopy3D(void *Dst, size_t DPitch, size_t DSPitch,
                          const void *Src, size_t SPitch, size_t SSPitch,
                          size_t Width, size_t Height, size_t Depth) {
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif
  std::lock_guard<std::mutex> Lock(QueueMtx);

  auto ChipEvent = memCopy3DAsyncImpl(Dst, DPitch, DSPitch, Src, SPitch,
                                      SSPitch, Width, Height, Depth);
  ChipEvent->Msg = "memCopy3D";
  finish();
  updateLastEvent(ChipEvent);
  ChipEvent->track();
}

void CHIPQueue::memCopy3DAsync(void *Dst, size_t DPitch, size_t DSPitch,
                               const void *Src, size_t SPitch, size_t SSPitch,
                               size_t Width, size_t Height, size_t Depth) {
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif
  std::lock_guard<std::mutex> Lock(QueueMtx);

  auto ChipEvent = memCopy3DAsyncImpl(Dst, DPitch, DSPitch, Src, SPitch,
                                      SSPitch, Width, Height, Depth);
  ChipEvent->Msg = "memCopy3DAsync";
  updateLastEvent(ChipEvent);
  ChipEvent->track();
}

CHIPEvent *CHIPQueue::RegisteredVarCopy(CHIPExecItem *ExecItem,
                                        bool KernelSubmitted) {

  CHIPEvent *RegisterVarEvent = nullptr;
  auto &ArgTyInfos = ExecItem->getKernel()->getFuncInfo()->ArgTypeInfo;
  auto AllocTracker = Backend->getActiveDevice()->AllocationTracker;
  auto Args = ExecItem->getArgsPointer();
  unsigned InArgI = 0;
  for (unsigned OutArgI = 0; OutArgI < ExecItem->getNumArgs(); OutArgI++) {
    if (ArgTyInfos[OutArgI].Space == OCLSpace::Local)
      // An argument inserted by HipDynMemExternReplaceNewPass hence
      // there is no corresponding value in argument list.
      continue;
    if (ArgTyInfos[OutArgI].Type == OCLType::Sampler) {
      // Texture lowering pass splits hipTextureObject_t arguments to
      // image and sampler arguments so there are additional
      // arguments. Don't bump the InArgI when we see an additional
      // argument.
      continue;
    }
    void **k = reinterpret_cast<void **>(Args[InArgI++]);
    if (!k)
      // HIP program provided (Clang generated) argument list should
      // not have NULLs in it.
      CHIPERR_LOG_AND_THROW(
          "Unexcepted internal error: Argument list has NULLs.", hipErrorTbd);
    void *DevPtr = reinterpret_cast<void *>(*k);
    auto AllocInfo = AllocTracker->getAllocInfo(DevPtr);
    if (!AllocInfo)
      continue;
    // CHIPERR_LOG_AND_THROW("A pointer argument was passed to the kernel but
    // "
    //                       "it was not registered",
    //                       hipErrorTbd);
    void *HostPtr = AllocInfo->HostPtr;

    // If this is a shared pointer then we don't need to transfer data back
    if (AllocInfo->MemoryType == hipMemoryTypeUnified) {
      logDebug("MemoryType: unified -> skipping");
      continue;
    }

    if (HostPtr) {
      auto AllocInfo = AllocTracker->getAllocInfo(DevPtr);

      if (!KernelSubmitted) {
        logDebug("A hipHostRegister argument was found. Appending a mem copy "
                 "Host -> Device {} -> {}",
                 DevPtr, HostPtr);
        RegisterVarEvent =
            this->memCopyAsyncImpl(DevPtr, HostPtr, AllocInfo->Size);
        RegisterVarEvent->Msg = "hipHostRegisterMemCpyHostToDev";
      } else {
        logDebug("A hipHostRegister argument was found. Appending a mem copy "
                 "back to the host {} -> {}",
                 DevPtr, HostPtr);
        RegisterVarEvent =
            this->memCopyAsyncImpl(HostPtr, DevPtr, AllocInfo->Size);
        RegisterVarEvent->Msg = "hipHostRegisterMemCpyDevToHost";
      }
      updateLastEvent(RegisterVarEvent);
    }
  }
  return RegisterVarEvent;
}

void CHIPQueue::launch(CHIPExecItem *ExecItem) {
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif
  std::lock_guard<std::mutex> LockQueue(Backend->BackendMtx);

  auto TotalThreadsPerBlock =
      ExecItem->getBlock().x * ExecItem->getBlock().y * ExecItem->getBlock().z;
  auto DeviceProps = ExecItem->getQueue()->getDevice()->getDeviceProps();
  auto MaxTotalThreadsPerBlock = DeviceProps.maxThreadsPerBlock;

  if (TotalThreadsPerBlock > MaxTotalThreadsPerBlock) {
    logCritical("Requested total local size {} exceeds HW limit {}",
                TotalThreadsPerBlock, MaxTotalThreadsPerBlock);
    CHIPERR_LOG_AND_THROW("Requested local size exceeds HW max",
                          hipErrorLaunchFailure);
  }

  if (ExecItem->getBlock().x > DeviceProps.maxThreadsDim[0] ||
      ExecItem->getBlock().y > DeviceProps.maxThreadsDim[1] ||
      ExecItem->getBlock().z > DeviceProps.maxThreadsDim[2]) {
    logCritical(
        "Requested local size dimension ({}, {}, {}) exceeds max ({}, {}, {})",
        ExecItem->getBlock().x, ExecItem->getBlock().y, ExecItem->getBlock().z,
        DeviceProps.maxThreadsDim[0], DeviceProps.maxThreadsDim[1],
        DeviceProps.maxThreadsDim[2]);
    CHIPERR_LOG_AND_THROW("Requested local size exceeds HW max",
                          hipErrorLaunchFailure);
  }

  auto RegisteredVarInEvent = RegisteredVarCopy(ExecItem, false);
  auto LaunchEvent = launchImpl(ExecItem);
  auto RegisteredVarOutEvent = RegisteredVarCopy(ExecItem, true);

  RegisteredVarOutEvent ? updateLastEvent(RegisteredVarOutEvent)
                        : updateLastEvent(LaunchEvent);

  if (RegisteredVarInEvent)
    RegisteredVarInEvent->track();
  LaunchEvent->track();
  if (RegisteredVarOutEvent)
    RegisteredVarOutEvent->track();
}

CHIPEvent *
CHIPQueue::enqueueBarrier(std::vector<CHIPEvent *> *EventsToWaitFor) {
  std::lock_guard<std::mutex> Lock(QueueMtx);
  auto ChipEvent = enqueueBarrierImpl(EventsToWaitFor);
  ChipEvent->Msg = "enqueueBarrier";
  updateLastEvent(ChipEvent);
  ChipEvent->track();
  return ChipEvent;
}
CHIPEvent *CHIPQueue::enqueueMarker() {
  std::lock_guard<std::mutex> Lock(QueueMtx);
  auto ChipEvent = enqueueMarkerImpl();
  ChipEvent->Msg = "enqueueMarker";
  updateLastEvent(ChipEvent);
  ChipEvent->track();
  return ChipEvent;
}

void CHIPQueue::memPrefetch(const void *Ptr, size_t Count) {
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif
  std::lock_guard<std::mutex> Lock(QueueMtx);

  auto ChipEvent = memPrefetchImpl(Ptr, Count);
  ChipEvent->Msg = "memPrefetch";
  updateLastEvent(ChipEvent);
  ChipEvent->track();
}

void CHIPQueue::launchHostFunc(const void *HostFunction, dim3 NumBlocks,
                               dim3 DimBlocks, void **Args,
                               size_t SharedMemBytes) {
  CHIPExecItem ExecItem(NumBlocks, DimBlocks, SharedMemBytes, this);

  CHIPDevice *ChipDev = getDevice();
  CHIPKernel *ChipKernel = ChipDev->findKernelByHostPtr(HostFunction);

  ExecItem.setArgPointer(Args);
  ExecItem.setKernel(ChipKernel);
  launch(&ExecItem);
}

void CHIPQueue::launchWithKernelParams(dim3 Grid, dim3 Block,
                                       unsigned int SharedMemBytes, void **Args,
                                       CHIPKernel *Kernel) {
  UNIMPLEMENTED();
}

void CHIPQueue::launchWithExtraParams(dim3 Grid, dim3 Block,
                                      unsigned int SharedMemBytes, void **Extra,
                                      CHIPKernel *Kernel) {
  UNIMPLEMENTED();
}

///////// End Enqueue Operations //////////

CHIPDevice *CHIPQueue::getDevice() {
  if (ChipDevice_ == nullptr) {
    std::string Msg = "chip_device is null";
    CHIPERR_LOG_AND_THROW(Msg, hipErrorLaunchFailure);
  }

  return ChipDevice_;
}

CHIPQueueFlags CHIPQueue::getFlags() { return QueueFlags_; }
int CHIPQueue::getPriority() { return Priority_; }
void CHIPQueue::addCallback(hipStreamCallback_t Callback, void *UserData) {
  CHIPCallbackData *Callbackdata =
      Backend->createCallbackData(Callback, UserData, this);

  {
    std::lock_guard<std::mutex> Lock(Backend->CallbackQueueMtx);
    Backend->CallbackQueue.push(Callbackdata);
  }

  return;
}
