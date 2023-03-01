/*
 * Copyright (c) 2021-23 CHIP-SPV developers
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

/// Queue a kernel for retrieving information about the device variable.
static void queueKernel(CHIPQueue *Q, CHIPKernel *K, void *Args[] = nullptr,
                        dim3 GridDim = dim3(1), dim3 BlockDim = dim3(1),
                        size_t SharedMemSize = 0) {
  assert(Q);
  assert(K);
  // FIXME: Should construct backend specific exec item or make the exec
  //        item a backend agnostic class.
  CHIPExecItem *EI =
      Backend->createCHIPExecItem(GridDim, BlockDim, SharedMemSize, Q);
  EI->setKernel(K);

  EI->copyArgs(Args);
  EI->setupAllArgs();

  auto ChipQueue = EI->getQueue();
  if (!ChipQueue)
    CHIPERR_LOG_AND_THROW(
        "Tried to launch kernel for an ExecItem which has a null queue",
        hipErrorTbd);

  ChipQueue->launch(EI);
  delete EI;
}

/// Queue a shadow kernel for binding a device variable (a pointer) to
/// the given allocation.
static void queueVariableInfoShadowKernel(CHIPQueue *Q, CHIPModule *M,
                                          const CHIPDeviceVar *Var,
                                          void *InfoBuffer) {
  assert(M && Var && InfoBuffer);
  auto *K = M->getKernelByName(std::string(ChipVarInfoPrefix) +
                               std::string(Var->getName()));
  assert(K && "Module is missing a shadow kernel?");
  void *Args[] = {&InfoBuffer};
  queueKernel(Q, K, Args);
}

static void queueVariableBindShadowKernel(CHIPQueue *Q, CHIPModule *M,
                                          const CHIPDeviceVar *Var) {
  assert(M && Var);
  auto *DevPtr = Var->getDevAddr();
  assert(DevPtr && "Space has not be allocated for a variable.");
  auto *K = M->getKernelByName(std::string(ChipVarBindPrefix) +
                               std::string(Var->getName()));
  assert(K && "Module is missing a shadow kernel?");
  void *Args[] = {&DevPtr};
  queueKernel(Q, K, Args);
}

static void queueVariableInitShadowKernel(CHIPQueue *Q, CHIPModule *M,
                                          const CHIPDeviceVar *Var) {
  assert(M && Var);
  auto *K = M->getKernelByName(std::string(ChipVarInitPrefix) +
                               std::string(Var->getName()));
  assert(K && "Module is missing a shadow kernel?");
  queueKernel(Q, K);
}

CHIPCallbackData::CHIPCallbackData(hipStreamCallback_t TheCallbackF,
                                   void *TheCallbackArgs,
                                   CHIPQueue *TheChipQueue)
    : ChipQueue(TheChipQueue), CallbackArgs(TheCallbackArgs),
      CallbackF(TheCallbackF) {}

void CHIPCallbackData::execute(hipError_t ResultFromDependency) {
  CallbackF(ChipQueue, ResultFromDependency, CallbackArgs);
}

// CHIPDeviceVar
// ************************************************************************
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
    LOCK( // CHIPAllocationTracker::PtrToAllocInfo_
        AllocationTrackerMtx);
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
  LOCK(AllocationTrackerMtx); // reading CHIPAllocationTracker::GlobalMemSize
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
  LOCK(AllocationTrackerMtx); // reading CHIPAllocationTracker::GlobalMemSize
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
  LOCK(AllocationTrackerMtx); // writing CHIPAllocationTracker::PtrToAllocInfo_
  // TODO AllocInfo turned into class and constructor take care of this
  if (MemoryType == hipMemoryTypeHost) {
    AllocInfo->HostPtr = AllocInfo->DevPtr;
    // Map onto host so that the data can be potentially initialized on host
    Backend->getActiveDevice()->getDefaultQueue()->MemMap(
        AllocInfo, CHIPQueue::MEM_MAP_TYPE::HOST_WRITE);
  }

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
  LOCK(AllocationTrackerMtx); // CHIPAllocationTracker::PtrToAllocInfo_
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

void CHIPEvent::releaseDependencies() {
  for (auto Event : DependsOnList) {
    Event->decreaseRefCount("An event that depended on this one has finished");
  }
  LOCK(EventMtx); // CHIPEvent::DependsOnList
  DependsOnList.clear();
}

void CHIPEvent::decreaseRefCount(std::string Reason) {
  LOCK(EventMtx); // CHIPEvent::Refc_
  // logDebug("CHIPEvent::decreaseRefCount() {} {} refc {}->{} REASON: {}",
  //          (void *)this, Msg.c_str(), *Refc_, *Refc_ - 1, Reason);
  if (*Refc_ > 0) {
    (*Refc_)--;
  } else {
    logError("CHIPEvent::decreaseRefCount() called when refc == 0");
  }
  // Destructor to be called by event monitor once backend is done using it
}
void CHIPEvent::increaseRefCount(std::string Reason) {
  LOCK(EventMtx); // CHIPEvent::Refc_
  // logDebug("CHIPEvent::increaseRefCount() {} {} refc {}->{} REASON: {}",
  //          (void *)this, Msg.c_str(), *Refc_, *Refc_ + 1, Reason);
  (*Refc_)++;
}

size_t CHIPEvent::getCHIPRefc() {
  LOCK(this->EventMtx); // CHIPEvent::Refc_
  return *Refc_;
}

// CHIPModuleflags_
//*************************************************************************************
void CHIPModule::consumeSPIRV() {
  FuncIL_ = (uint8_t *)Src_->getBinary().data();
  IlSize_ = Src_->getBinary().size();

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

CHIPKernel *CHIPModule::getKernelByName(const std::string &Name) {
  auto *Kernel = findKernel(Name);
  if (!Kernel) {
    std::string Msg = "Failed to find kernel via kernel name: " + Name;
    CHIPERR_LOG_AND_THROW(Msg, hipErrorLaunchFailure);
  }
  return Kernel;
}

bool CHIPModule::hasKernel(std::string Name) { return findKernel(Name); }

CHIPKernel *CHIPModule::getKernel(const void *HostFPtr) {
  logDebug("{} CHIPModule::getKernel({})", (void *)this, HostFPtr);
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

void CHIPModule::prepareDeviceVariablesNoLock(CHIPDevice *Device,
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

SPVFuncInfo *CHIPModule::findFunctionInfo(const std::string &FName) {
  return FuncInfos_.count(FName) ? FuncInfos_.at(FName).get() : nullptr;
}

// CHIPKernel
//*************************************************************************************
CHIPKernel::CHIPKernel(std::string HostFName, SPVFuncInfo *FuncInfo)
    : HostFName_(HostFName), FuncInfo_(FuncInfo) {}
CHIPKernel::~CHIPKernel(){};
std::string CHIPKernel::getName() { return HostFName_; }
const void *CHIPKernel::getHostPtr() { return HostFPtr_; }
const void *CHIPKernel::getDevPtr() { return DevFPtr_; }

SPVFuncInfo *CHIPKernel::getFuncInfo() { return FuncInfo_; }

void CHIPKernel::setName(std::string HostFName) { HostFName_ = HostFName; }
void CHIPKernel::setHostPtr(const void *HostFPtr) { HostFPtr_ = HostFPtr; }
void CHIPKernel::setDevPtr(const void *DevFPtr) { DevFPtr_ = DevFPtr; }

// CHIPExecItem
//*************************************************************************************
void CHIPExecItem::copyArgs(void **Args) {
  for (int i = 0; i < getNumArgs(); i++) {
    Args_.push_back(Args[i]);
  }
}

CHIPExecItem::CHIPExecItem(dim3 GridDim, dim3 BlockDim, size_t SharedMem,
                           hipStream_t ChipQueue)
    : SharedMem_(SharedMem), GridDim_(GridDim), BlockDim_(BlockDim),
      ChipQueue_(static_cast<CHIPQueue *>(ChipQueue)){};

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

CHIPDevice::~CHIPDevice() {
  LOCK(DeviceMtx); // CHIPDevice::ChipQueues_
  logDebug("~CHIPDevice() {}", (void *)this);
  while (this->ChipQueues_.size() > 0) {
    delete ChipQueues_[0];
    ChipQueues_.erase(ChipQueues_.begin());
  }

  delete LegacyDefaultQueue;
  LegacyDefaultQueue = nullptr;
}
CHIPQueue *CHIPDevice::getLegacyDefaultQueue() { return LegacyDefaultQueue; }

CHIPQueue *CHIPDevice::getDefaultQueue() {
#ifdef HIP_API_PER_THREAD_DEFAULT_STREAM
  return getPerThreadDefaultQueue();
#else
  return getLegacyDefaultQueue();
#endif
}

bool CHIPDevice::isPerThreadStreamUsed() {
  LOCK(DeviceMtx); // CHIPDevice::PerThreadStreamUsed
  return PerThreadStreamUsed_;
}

bool CHIPDevice::isPerThreadStreamUsedNoLock() { return PerThreadStreamUsed_; }

void CHIPDevice::setPerThreadStreamUsed(bool Status) {
  LOCK(DeviceMtx); // CHIPDevice::PerThreadStreamUsed
  PerThreadStreamUsed_ = Status;
}

CHIPQueue *CHIPDevice::getPerThreadDefaultQueue() {
  LOCK(DeviceMtx); // CHIPDevice::PerThreadStreamUsed
  return getPerThreadDefaultQueueNoLock();
}

CHIPQueue *CHIPDevice::getPerThreadDefaultQueueNoLock() {
  if (!PerThreadDefaultQueue.get()) {
    logDebug("PerThreadDefaultQueue is null.. Creating a new queue.");
    PerThreadDefaultQueue =
        std::unique_ptr<CHIPQueue>(Backend->createCHIPQueue(this));
    PerThreadStreamUsed_ = true;
    PerThreadDefaultQueue.get()->PerThreadQueueForDevice = this;
  }

  return PerThreadDefaultQueue.get();
}

std::vector<CHIPKernel *> CHIPDevice::getKernels() {
  std::vector<CHIPKernel *> ChipKernels;
  for (auto &Kv : SrcModToCompiledMod_) {
    for (CHIPKernel *Kernel : Kv.second->getKernels())
      ChipKernels.push_back(Kernel);
  }
  return ChipKernels;
}

std::string CHIPDevice::getName() { return std::string(HipDeviceProps_.name); }

void CHIPDevice::init() {
  LOCK(DeviceMtx) // CHIPDevice::LegacyDefaultQueue
  std::call_once(PropsPopulated_, &CHIPDevice::populateDevicePropertiesImpl,
                 this);
  if (!AllocationTracker)
    AllocationTracker = new CHIPAllocationTracker(
        HipDeviceProps_.totalGlobalMem, HipDeviceProps_.name);

  CHIPQueueFlags Flags;
  int Priority = 1; // TODO : set a default
  LegacyDefaultQueue = createQueue(Flags, Priority);
}

void CHIPDevice::copyDeviceProperties(hipDeviceProp_t *Prop) {
  logDebug("CHIPDevice->copy_device_properties()");
  if (Prop)
    std::memcpy(Prop, &this->HipDeviceProps_, sizeof(hipDeviceProp_t));
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

void CHIPDevice::eraseModule(CHIPModule *Module) {
  LOCK(DeviceMtx); // SrcModToCompiledMod_
  for (auto &Kv : SrcModToCompiledMod_)
    if (Kv.second == Module) {
      delete Module;
      SrcModToCompiledMod_.erase(Kv.first);
      break;
    }
}

void CHIPDevice::addQueue(CHIPQueue *ChipQueue) {
  LOCK(DeviceMtx) // writing CHIPDevice::ChipQueues_
  logDebug("{} CHIPDevice::addQueue({})", (void *)this, (void *)ChipQueue);

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
  LOCK(Backend->EventsMtx); // trackImpl CHIPBackend::Events
  LOCK(EventMtx);           // writing bool CHIPEvent::TrackCalled_
  if (!TrackCalled_) {
    Backend->Events.push_back(this);
    TrackCalled_ = true;
  }
}

CHIPQueue *CHIPDevice::createQueueAndRegister(CHIPQueueFlags Flags,
                                              int Priority) {

  auto ChipQueue = createQueue(Flags, Priority);
  // Add the queue handle to the device and the Backend
  addQueue(ChipQueue);
  return ChipQueue;
}

CHIPQueue *CHIPDevice::createQueueAndRegister(const uintptr_t *NativeHandles,
                                              const size_t NumHandles) {
  auto ChipQueue = createQueue(NativeHandles, NumHandles);
  // Add the queue handle to the device and the Backend
  addQueue(ChipQueue);
  return ChipQueue;
}

std::vector<CHIPQueue *> &CHIPDevice::getQueues() {
  LOCK(DeviceMtx); // reading CHIPDevice::ChipQueues_
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

void CHIPDevice::removeContext(CHIPContext *CHIPContext) {}

bool CHIPDevice::removeQueue(CHIPQueue *ChipQueue) {
  /**
   * If commands are still executing on the specified stream, some may complete
   * execution before the queue is deleted. The queue may be destroyed while
   * some commands are still inflight, or may wait for all commands queued to
   * the stream before destroying it.
   *
   * Choosing not to call Queue->finish()
   */
  LOCK(DeviceMtx) // reading CHIPDevice::ChipQueues_
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

  delete ChipQueue;
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

/// Prepares device variables for a module for which the host pointer
/// is member of.
void CHIPDevice::prepareDeviceVariables(HostPtr Ptr) {
  if (auto *Mod = getOrCreateModule(Ptr)) {
    LOCK(DeviceVarMtx); // CHIPModule::prepareDeviceVariablesNoLock()
    logDebug("Prepare variables in module {}", static_cast<const void *>(Mod));
    Mod->prepareDeviceVariablesNoLock(this, getDefaultQueue());
  }
}

void CHIPDevice::invalidateDeviceVariables() {
  // CHIPDevice::SrcModToCompiledMod_
  // CHIPModule::invalidateDeviceVariablesNoLock()
  LOCK(DeviceVarMtx); // CHIPDevice::SrcModToCompiledMod_
  logTrace("invalidate device variables.");
  for (auto &Kv : SrcModToCompiledMod_)
    Kv.second->invalidateDeviceVariablesNoLock();
}

void CHIPDevice::deallocateDeviceVariables() {
  // CHIPDevice::SrcModToCompiledMod_
  // CHIPModule::deallocateDeviceVariablesNoLock()
  LOCK(DeviceVarMtx); // CHIPDevice::SrcModToCompiledMod_
  logTrace("Deallocate storage for device variables.");
  for (auto &Kv : SrcModToCompiledMod_)
    Kv.second->deallocateDeviceVariablesNoLock(this);
}

/// Get compiled module associated with the host pointer 'Ptr'. Return
/// nullptr if 'Ptr' is not associated with any module.
CHIPModule *CHIPDevice::getOrCreateModule(HostPtr Ptr) {
  {
    LOCK(DeviceVarMtx); // CHIPDevice::HostPtrToCompiledMod_
    if (HostPtrToCompiledMod_.count(Ptr))
      return HostPtrToCompiledMod_[Ptr];
  }

  // The module, which the 'Ptr' is member of, might not be compiled yet.
  auto *SrcMod = getSPVRegister().getSource(Ptr);
  if (!SrcMod) // Client gave an invalid pointer?
    return nullptr;

  // Found the source module, now compile it.
  auto *Mod = getOrCreateModule(*SrcMod);

#ifndef NDEBUG
  {
    LOCK(DeviceVarMtx); // CHIPDevice::HostPtrToCompiledMod_
    assert((!Mod || (HostPtrToCompiledMod_.count(Ptr) &&
                     HostPtrToCompiledMod_[Ptr] == Mod)) &&
           "Forgot to map the host pointers");
  }
#endif

  return Mod;
}

/// Get compiled module for the source module 'SrcMod'.
CHIPModule *CHIPDevice::getOrCreateModule(const SPVModule &SrcMod) {
  LOCK(DeviceVarMtx); // CHIPDevice::SrcModToCompiledMod_
                      // CHIPDevice::HostPtrToCompiledMod_
                      // CHIPDevice::DeviceVarLookup_

  // Check if we have already created the module for the source.
  if (SrcModToCompiledMod_.count(&SrcMod))
    return SrcModToCompiledMod_[&SrcMod];

  logDebug("Compile module {}", static_cast<const void *>(&SrcMod));

  auto *Module = compile(SrcMod);
  if (!Module) // Probably a compile error.
    return nullptr;

  // Bind host pointers to their backend counterparts.
  for (const auto &Info : SrcMod.Kernels) {
    std::string NameTmp(Info.Name.begin(), Info.Name.end());
    CHIPKernel *Kernel = Module->getKernelByName(NameTmp);
    assert(Kernel && "Kernel went missing?");
    Kernel->setHostPtr(Info.Ptr);
    HostPtrToCompiledMod_[Info.Ptr] = Module;
  }

  for (const auto &Info : SrcMod.Variables) {
    // Global device variables in the original HIP sources have been
    // converted by a global variable pass (HipGlobalVariables.cpp)
    // and they are accessible through specially named shadow
    // kernels.
    std::string NameTmp(Info.Name.begin(), Info.Name.end());
    std::string VarInfoKernelName = std::string(ChipVarInfoPrefix) + NameTmp;

    if (!Module->hasKernel(VarInfoKernelName)) {
      // The kernel compilation pipe is allowed to remove device-side unused
      // global variables from the device modules. This is utilized in the
      // abort implementation to signal that abort is not called in the
      // module. The lack of the variable in the device module is used as a
      // quick (and dirty) way to not query for the global flag value after
      // each kernel execution (reading of which requires kernel launches).
      logTrace(
          "Device variable {} not found in the module -- removed as unused?",
          Info.Name);
      continue;
    }
    auto *Var = new CHIPDeviceVar(&Info);
    Module->addDeviceVariable(Var);

    DeviceVarLookup_.insert(std::make_pair(Info.Ptr, Var));
    HostPtrToCompiledMod_[Info.Ptr] = Module;
  }

  SrcModToCompiledMod_.insert(std::make_pair(&SrcMod, Module));
  return Module;
}

// CHIPContext
//*************************************************************************************
CHIPContext::CHIPContext() {}
CHIPContext::~CHIPContext() {
  logDebug("~CHIPContext() {}", (void *)this);
  delete ChipDevice_;
}

void CHIPContext::syncQueues(CHIPQueue *TargetQueue) {
  auto Dev = Backend->getActiveDevice();
  LOCK(Dev->DeviceMtx); // CHIPDevice::ChipQueues_ via getQueuesNoLock()

  auto DefaultQueue = Dev->getDefaultQueue();
#ifdef HIP_API_PER_THREAD_DEFAULT_STREAM
  // The per-thread default stream is an implicit stream local to both the
  // thread and the CUcontext, and which does not synchronize with other
  // streams (just like explcitly created streams). The per-thread default
  // stream is not a non-blocking stream and will synchronize with the
  // legacy default stream if both are used in a program.

  // since HIP_API_PER_THREAD_DEFAULT_STREAM is enabled, there is no legacy
  // default stream thus no syncronization necessary
  if (TargetQueue == DefaultQueue)
    return;
#endif
  std::vector<CHIPQueue *> QueuesToSyncWith;

  // The per-thread default stream is not a non-blocking stream and will
  // synchronize with the legacy default stream if both are used in a program
  if (Dev->isPerThreadStreamUsedNoLock()) {
    if (TargetQueue == Dev->getPerThreadDefaultQueueNoLock())
      QueuesToSyncWith.push_back(DefaultQueue);
    else if (TargetQueue == Dev->getLegacyDefaultQueue())
      QueuesToSyncWith.push_back(Dev->getPerThreadDefaultQueueNoLock());
  }

  // Always sycn with all blocking queues
  for (auto Queue : Dev->getQueuesNoLock()) {
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
    SyncQueuesEvent = TargetQueue->enqueueBarrierImpl(&EventsToWaitOn);
    SyncQueuesEvent->Msg = "barrierSyncQueue";
    TargetQueue->updateLastEvent(SyncQueuesEvent);
  } else { // blocking stream must wait until default stream is done
    auto Ev = DefaultQueue->getLastEvent();
    if (Ev)
      EventsToWaitOn.push_back(Ev);
    SyncQueuesEvent = TargetQueue->enqueueBarrierImpl(&EventsToWaitOn);
    SyncQueuesEvent->Msg = "barrierSyncQueue";
    TargetQueue->updateLastEvent(SyncQueuesEvent);
  }
  SyncQueuesEvent->track();
}

CHIPDevice *CHIPContext::getDevice() {
  assert(this->ChipDevice_);
  return ChipDevice_;
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
  void *AllocatedPtr, *HostPtr = nullptr;
  // TOOD hipCtx - use the device with which this context is associated
  CHIPDevice *ChipDev = Backend->getActiveDevice();
  if (!Flags.isDefault()) {
    if (Flags.isMapped())
      MemType = hipMemoryType::hipMemoryTypeHost;
    if (Flags.isCoherent())
      UNIMPLEMENTED(nullptr);
    if (Flags.isNonCoherent())
      // UNIMPLEMENTED(nullptr);
      if (Flags.isNumaUser())
        UNIMPLEMENTED(nullptr);
    if (Flags.isPortable())
      UNIMPLEMENTED(nullptr);
    if (Flags.isWriteCombined())
      logWarn("hipHostAllocWriteCombined is not supported. Ignoring.");
    // UNIMPLEMENTED(nullptr);
  }

  if (Size > ChipDev->getMaxMallocSize()) {
    logCritical("Requested allocation of {} exceeds the maximum size of a "
                "single allocation of {}",
                Size, ChipDev->getMaxMallocSize());
    CHIPERR_LOG_AND_THROW(
        "Allocation size exceeds limits for a single allocation",
        hipErrorOutOfMemory);
  }
  assert(ChipDev->getContext() == this);

  assert(ChipDev->AllocationTracker && "AllocationTracker was not created!");
  if (!ChipDev->AllocationTracker->reserveMem(Size))
    return nullptr;
  AllocatedPtr = allocateImpl(Size, Alignment, MemType);
  if (AllocatedPtr == nullptr)
    ChipDev->AllocationTracker->releaseMemReservation(Size);

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

  auto Dev = getDevice();
  // Free all the memory reservations on each device
  Dev->AllocationTracker->releaseMemReservation(
      Dev->AllocationTracker->TotalMemSize);
  AllocatedPtrs_.clear();

  getDevice()->reset();
}

hipError_t CHIPContext::free(void *Ptr) {
  CHIPDevice *ChipDev = Backend->getActiveDevice();
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
int CHIPBackend::getPerThreadQueuesActive() {
  LOCK(Backend->BackendMtx); // Prevent adding/removing devices while iterating
  int Active = 0;
  for (auto Dev : getDevices()) {
    if (Dev->isPerThreadStreamUsed()) {
      Active++;
    }
  }
  return Active;
}
int CHIPBackend::getQueuePriorityRange() {
  assert(MinQueuePriority_);
  return MinQueuePriority_;
}

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

CHIPBackend::CHIPBackend() {
  logDebug("CHIPBackend Base Constructor");
  Logger = spdlog::default_logger();
};

CHIPBackend::~CHIPBackend() {
  logDebug("CHIPBackend Destructor. Deleting all pointers.");
  if (StaleEventMonitor_)
    StaleEventMonitor_->stop();
  if (CallbackEventMonitor_)
    CallbackEventMonitor_->stop();
  Events.clear();
  for (auto &Ctx : ChipContexts) {
    Backend->removeContext(Ctx);
    delete Ctx;
  }
}

void CHIPBackend::waitForThreadExit() {
  /**
   * If the main thread just creates a bunch of other threads and tries to exit
   * right away, it could be the case that all those threads are not yet done
   * with initialization. In particular, these threads might not have yet
   * created their per-thread queues which is how we keep track of threads.
   *
   * So we just wait for 0.5 seconds before starting to check for thread exit.
   */
  pthread_yield();
  // TODO fix-255 is there a better way to do this?
  unsigned long long int sleepMicroSeconds = 500000;
  usleep(sleepMicroSeconds);

  while (true) {
    {
      auto NumPerThreadQueuesActive = Backend->getPerThreadQueuesActive();
      if (!NumPerThreadQueuesActive)
        break;

      logDebug(
          "CHIPBackend::waitForThreadExit() per-thread queues still active "
          "{}. Sleeping for 1s..",
          NumPerThreadQueuesActive);
    }
    sleep(1);
  }

  // Cleanup all queues
  {
    LOCK(Backend->BackendMtx); // prevent devices from being destrpyed

    for (auto Dev : Backend->getDevices()) {
      Dev->getLegacyDefaultQueue()->updateLastEvent(nullptr);
      int NumQueues = Dev->getQueues().size();
      if (NumQueues) {
        logWarn("Not all user created streams have been destoyed... Queues "
                "remaining: {}",
                NumQueues);
        logWarn("Make sure to call hipStreamDestroy() for all queues that have "
                "been created via hipStreamCreate()");
        logWarn("Removing user-created streams without calling a destructor");
        Dev->getQueues().clear();
        if (Backend->Events.size()) {
          logWarn("Clearing Event list {}", Backend->Events.size());
          Backend->Events.clear();
        }
      }
      /**
       * Skip setting LastEvent for these queues. At this point, the main() has
       * exited and the memory allocated for these queues has already been
       * freed.
       *
       */
    }
  }
}
void CHIPBackend::initialize(std::string PlatformStr, std::string DeviceTypeStr,
                             std::string DeviceIdStr) {
  initializeImpl(PlatformStr, DeviceTypeStr, DeviceIdStr);
  CustomJitFlags = read_env_var("CHIP_JIT_FLAGS", false);
  if (ChipContexts.size() == 0) {
    std::string Msg = "No CHIPContexts were initialized";
    CHIPERR_LOG_AND_THROW(Msg, hipErrorInitializationError);
  }

  PrimaryContext = ChipContexts[0];
  setActiveContext(
      ChipContexts[0]); // pushes primary context to context stack for thread 0
}

void CHIPBackend::setActiveContext(CHIPContext *ChipContext) {
  ChipCtxStack.push(ChipContext);
}

void CHIPBackend::setActiveDevice(CHIPDevice *ChipDevice) {
  Backend->setActiveContext(ChipDevice->getContext());
}

CHIPContext *CHIPBackend::getActiveContext() {
  // assert(ChipCtxStack.size() > 0 && "Context stack is empty");
  if (ChipCtxStack.size() == 0) {
    logDebug("Context stack is empty for thread {}", pthread_self());
    ChipCtxStack.push(PrimaryContext);
  }
  return ChipCtxStack.top();
};

CHIPDevice *CHIPBackend::getActiveDevice() {
  CHIPContext *Ctx = getActiveContext();
  return Ctx->getDevice();
};

std::vector<CHIPDevice *> CHIPBackend::getDevices() {
  std::vector<CHIPDevice *> Devices;
  for (auto Ctx : ChipContexts) {
    Devices.push_back(Ctx->getDevice());
  }

  return Devices;
}

size_t CHIPBackend::getNumDevices() { return ChipContexts.size(); }

void CHIPBackend::removeContext(CHIPContext *ChipContext) {
  auto ContextFound =
      std::find(ChipContexts.begin(), ChipContexts.end(), ChipContext);
  if (ContextFound != ChipContexts.end()) {
    ChipContexts.erase(ContextFound);
  }
}

void CHIPBackend::addContext(CHIPContext *ChipContext) {
  ChipContexts.push_back(ChipContext);
}

hipError_t CHIPBackend::configureCall(dim3 Grid, dim3 Block, size_t SharedMem,
                                      hipStream_t ChipQueue) {
  logDebug("CHIPBackend->configureCall(grid=({},{},{}), block=({},{},{}), "
           "shared={}, q={}",
           Grid.x, Grid.y, Grid.z, Block.x, Block.y, Block.z, SharedMem,
           (void *)ChipQueue);
  CHIPExecItem *ExecItem =
      Backend->createCHIPExecItem(Grid, Block, SharedMem, ChipQueue);
  ChipExecStack.push(ExecItem);

  return hipSuccess;
}

CHIPDevice *CHIPBackend::findDeviceMatchingProps(const hipDeviceProp_t *Props) {
  CHIPDevice *MatchedDevice = nullptr;
  int MaxMatchedCount = 0;
  for (auto &Dev : getDevices()) {
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
  auto Dev = Backend->getActiveDevice();
  LOCK(Dev->DeviceMtx); // CHIPDevice::ChipQueues_ via getQueuesNoLock()

  if (ChipQueue == hipStreamPerThread) {
    return Dev->getPerThreadDefaultQueueNoLock();
  } else if (ChipQueue == hipStreamLegacy) {
    return Dev->getLegacyDefaultQueue();
  } else if (ChipQueue == nullptr) {
    return Dev->getDefaultQueue();
  }
  std::vector<CHIPQueue *> AllQueues;
  if (Dev->isPerThreadStreamUsedNoLock())
    AllQueues.push_back(Dev->getPerThreadDefaultQueueNoLock());
  AllQueues.push_back(Dev->getLegacyDefaultQueue());

  for (auto &Dev : Dev->getQueuesNoLock()) {
    AllQueues.push_back(Dev);
  }

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
  logDebug("CHIPQueue() {}", (void *)this);
};

CHIPQueue::CHIPQueue(CHIPDevice *ChipDevice, CHIPQueueFlags Flags)
    : CHIPQueue(ChipDevice, Flags, 0){};

CHIPQueue::~CHIPQueue() {
  updateLastEvent(nullptr);
  if (PerThreadQueueForDevice) {
    PerThreadQueueForDevice->setPerThreadStreamUsed(false);
  }
};

///////// Enqueue Operations //////////
hipError_t CHIPQueue::memCopy(void *Dst, const void *Src, size_t Size) {
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif
  CHIPEvent *ChipEvent;
  // Scope this so that we release mutex for finish()
  {
    auto AllocInfoDst =
        Backend->getActiveDevice()->AllocationTracker->getAllocInfo(Dst);
    auto AllocInfoSrc =
        Backend->getActiveDevice()->AllocationTracker->getAllocInfo(Src);
    if (AllocInfoDst && AllocInfoDst->MemoryType == hipMemoryTypeHost)
      Backend->getActiveDevice()->getDefaultQueue()->MemUnmap(AllocInfoDst);
    if (AllocInfoSrc && AllocInfoSrc->MemoryType == hipMemoryTypeHost)
      Backend->getActiveDevice()->getDefaultQueue()->MemUnmap(AllocInfoSrc);

    ChipEvent = memCopyAsyncImpl(Dst, Src, Size);

    if (AllocInfoDst && AllocInfoDst->MemoryType == hipMemoryTypeHost)
      Backend->getActiveDevice()->getDefaultQueue()->MemMap(
          AllocInfoDst, CHIPQueue::MEM_MAP_TYPE::HOST_WRITE);
    if (AllocInfoSrc && AllocInfoSrc->MemoryType == hipMemoryTypeHost)
      Backend->getActiveDevice()->getDefaultQueue()->MemMap(
          AllocInfoSrc, CHIPQueue::MEM_MAP_TYPE::HOST_WRITE);

    ChipEvent->Msg = "memCopy";
    updateLastEvent(ChipEvent);
    this->finish();
  }
  ChipEvent->track();

  return hipSuccess;
}
void CHIPQueue::memCopyAsync(void *Dst, const void *Src, size_t Size) {
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif
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

    auto ChipEvent = memFillAsyncImpl(Dst, Size, Pattern, PatternSize);
    ChipEvent->Msg = "memFill";
    updateLastEvent(ChipEvent);
    ChipEvent->track();
    this->finish();
  }
  return;
}

void CHIPQueue::memFillAsync(void *Dst, size_t Size, const void *Pattern,
                             size_t PatternSize) {
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif

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
  auto ChipEvent = memCopy2DAsyncImpl(Dst, DPitch, Src, SPitch, Width, Height);
  ChipEvent->Msg = "memCopy2D";
  finish();
  updateLastEvent(ChipEvent);
  ChipEvent->track();
}

void CHIPQueue::memCopy2DAsync(void *Dst, size_t DPitch, const void *Src,
                               size_t SPitch, size_t Width, size_t Height) {
  {
#ifdef ENFORCE_QUEUE_SYNC
    ChipContext_->syncQueues(this);
#endif
    auto ChipEvent =
        memCopy2DAsyncImpl(Dst, DPitch, Src, SPitch, Width, Height);
    ChipEvent->Msg = "memCopy2DAsync";
    updateLastEvent(ChipEvent);
    ChipEvent->track();
    this->finish();
  }
  return;
}

void CHIPQueue::memCopy3D(void *Dst, size_t DPitch, size_t DSPitch,
                          const void *Src, size_t SPitch, size_t SSPitch,
                          size_t Width, size_t Height, size_t Depth) {
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif

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

  auto ChipEvent = memCopy3DAsyncImpl(Dst, DPitch, DSPitch, Src, SPitch,
                                      SSPitch, Width, Height, Depth);
  ChipEvent->Msg = "memCopy3DAsync";
  updateLastEvent(ChipEvent);
  ChipEvent->track();
}

void CHIPQueue::updateLastNode(CHIPGraphNode *NewNode) {
  if (LastNode_ != nullptr) {
    NewNode->addDependency(LastNode_);
  }
  LastNode_ = NewNode;
}

void CHIPQueue::initCaptureGraph() { CaptureGraph_ = new CHIPGraph(); }

CHIPEvent *CHIPQueue::RegisteredVarCopy(CHIPExecItem *ExecItem,
                                        MANAGED_MEM_STATE ExecState) {

  CHIPEvent *RegisterVarEvent = nullptr;
  const auto &FuncInfo = ExecItem->getKernel()->getFuncInfo();
  auto AllocTracker = Backend->getActiveDevice()->AllocationTracker;

  auto ArgVisitor = [&](const SPVFuncInfo::ClientArg &Arg) -> void {
    if (Arg.Kind != SPVTypeKind::Pointer)
      return;

    auto *PtrArgValue = const_cast<void *>(Arg.Data);
    auto *DevPtr = *reinterpret_cast<void **>(PtrArgValue);
    auto AllocInfo = AllocTracker->getAllocInfo(DevPtr);
    if (!AllocInfo) {
      logWarn(
          "Allocation info not found. Unregistered outside USM allocation?");
      // Previously, we used to assert here. However, this will fail for the USM
      // where a pointer is allocated using USM outside of CHIP-SPV. assert(0 &&
      // "Unexcepted internal error: allocation info not found");
      return;
    }
    void *HostPtr = AllocInfo->HostPtr;

    // If this is a shared pointer then we don't need to transfer data back
    if (AllocInfo->MemoryType == hipMemoryTypeUnified) {
      logDebug("MemoryType: unified -> skipping");
      return;
    }

    // required for OpenCL when fine-grain SVM is not availbale
    if (AllocInfo->MemoryType == hipMemoryTypeHost) {
      if (ExecState == MANAGED_MEM_STATE::PRE_KERNEL) {
        MemUnmap(AllocInfo);
      } else {
        MemMap(AllocInfo,
               CHIPQueue::MEM_MAP_TYPE::HOST_WRITE); // TODO fixOpenCLTests -
                                                     // print ptr
      }
      return;
    }

    if (HostPtr && AllocInfo->MemoryType == hipMemoryTypeManaged) {
      auto AllocInfo = AllocTracker->getAllocInfo(DevPtr);

      if (ExecState == MANAGED_MEM_STATE::PRE_KERNEL) {
        logDebug("A hipHostRegister argument was found. Appending a mem copy "
                 "Host {} -> Device {}",
                 DevPtr, HostPtr);
        RegisterVarEvent =
            this->memCopyAsyncImpl(DevPtr, HostPtr, AllocInfo->Size);
        RegisterVarEvent->Msg = "hipHostRegisterMemCpyHostToDev";
      } else {
        logDebug("A hipHostRegister argument was found. Appending a mem copy "
                 "Device {} -> Host {}",
                 DevPtr, HostPtr);
        RegisterVarEvent =
            this->memCopyAsyncImpl(HostPtr, DevPtr, AllocInfo->Size);
        RegisterVarEvent->Msg = "hipHostRegisterMemCpyDevToHost";
      }
      updateLastEvent(RegisterVarEvent);
    }
  };
  FuncInfo->visitClientArgs(ExecItem->getArgs(), ArgVisitor);

  return RegisterVarEvent;
}

void CHIPQueue::launch(CHIPExecItem *ExecItem) {
  std::stringstream InfoStr;
  InfoStr << "\nLaunching kernel " << ExecItem->getKernel()->getName() << "\n";
  InfoStr << "GridDim: <" << ExecItem->getGrid().x << ", "
          << ExecItem->getGrid().y << ", " << ExecItem->getGrid().z << ">";
  InfoStr << " BlockDim: <" << ExecItem->getBlock().x << ", "
          << ExecItem->getBlock().y << ", " << ExecItem->getBlock().z << ">\n";

  const auto &FuncInfo = *ExecItem->getKernel()->getFuncInfo();
  InfoStr << "NumArgs: " << FuncInfo.getNumKernelArgs() << "\n";
  auto Visitor = [&](const SPVFuncInfo::KernelArg &Arg) -> void {
    InfoStr << "Arg " << Arg.Index << ": " << Arg.getKindAsString() << " "
            << Arg.Size << " " << Arg.Data << "\n";
  };
  FuncInfo.visitKernelArgs(ExecItem->getArgs(), Visitor);

  logDebug("{}", InfoStr.str());

#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif

  auto TotalThreadsPerBlock =
      ExecItem->getBlock().x * ExecItem->getBlock().y * ExecItem->getBlock().z;
  auto DeviceProps = getDevice()->getDeviceProps();
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

  auto RegisteredVarInEvent =
      RegisteredVarCopy(ExecItem, MANAGED_MEM_STATE::PRE_KERNEL);
  auto LaunchEvent = launchImpl(ExecItem);
  auto RegisteredVarOutEvent =
      RegisteredVarCopy(ExecItem, MANAGED_MEM_STATE::POST_KERNEL);

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
  auto ChipEvent = enqueueBarrierImpl(EventsToWaitFor);
  ChipEvent->Msg = "enqueueBarrier";
  updateLastEvent(ChipEvent);
  ChipEvent->track();
  return ChipEvent;
}
CHIPEvent *CHIPQueue::enqueueMarker() {
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

  auto ChipEvent = memPrefetchImpl(Ptr, Count);
  ChipEvent->Msg = "memPrefetch";
  updateLastEvent(ChipEvent);
  ChipEvent->track();
}

void CHIPQueue::launchKernel(CHIPKernel *ChipKernel, dim3 NumBlocks,
                             dim3 DimBlocks, void **Args,
                             size_t SharedMemBytes) {
  LOCK(Backend->BackendMtx); // Prevent the breakup of RegisteredVarCopy in&out
  CHIPExecItem *ExecItem =
      Backend->createCHIPExecItem(NumBlocks, DimBlocks, SharedMemBytes, this);
  ExecItem->setKernel(ChipKernel);
  ExecItem->copyArgs(Args);
  ExecItem->setupAllArgs();
  launch(ExecItem);
  delete ExecItem;
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
    LOCK(Backend->CallbackQueueMtx); // CHIPBackend::CallbackQueue
    Backend->CallbackQueue.push(Callbackdata);
  }

  return;
}
