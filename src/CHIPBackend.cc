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
static void queueKernel(chipstar::Queue *Q, chipstar::Kernel *K,
                        void *Args[] = nullptr, dim3 GridDim = dim3(1),
                        dim3 BlockDim = dim3(1), size_t SharedMemSize = 0) {
  assert(Q);
  assert(K);
  // FIXME: Should construct backend specific exec item or make the exec
  //        item a backend agnostic class.
  chipstar::ExecItem *EI =
      ::Backend->createExecItem(GridDim, BlockDim, SharedMemSize, Q);
  EI->setKernel(K);

  EI->copyArgs(Args);
  EI->setupAllArgs();

  auto ChipQueue = EI->getQueue();
  if (!ChipQueue)
    CHIPERR_LOG_AND_THROW("Tried to launch kernel for an chipstar::ExecItem "
                          "which has a null queue",
                          hipErrorTbd);

  ChipQueue->launch(EI);
  delete EI;
}

/// Queue a shadow kernel for binding a device variable (a pointer) to
/// the given allocation.
static void queueVariableInfoShadowKernel(chipstar::Queue *Q,
                                          chipstar::Module *M,
                                          const chipstar::DeviceVar *Var,
                                          void *InfoBuffer) {
  assert(M && Var && InfoBuffer);
  auto *K = M->getKernelByName(std::string(ChipVarInfoPrefix) +
                               std::string(Var->getName()));
  assert(K && "chipstar::Module is missing a shadow kernel?");
  void *Args[] = {&InfoBuffer};
  queueKernel(Q, K, Args);
}

static void queueVariableBindShadowKernel(chipstar::Queue *Q,
                                          chipstar::Module *M,
                                          const chipstar::DeviceVar *Var) {
  assert(M && Var);
  auto *DevPtr = Var->getDevAddr();
  assert(DevPtr && "Space has not be allocated for a variable.");
  auto *K = M->getKernelByName(std::string(ChipVarBindPrefix) +
                               std::string(Var->getName()));
  assert(K && "chipstar::Module is missing a shadow kernel?");
  void *Args[] = {&DevPtr};
  queueKernel(Q, K, Args);
}

static void queueVariableInitShadowKernel(chipstar::Queue *Q,
                                          chipstar::Module *M,
                                          const chipstar::DeviceVar *Var) {
  assert(M && Var);
  auto *K = M->getKernelByName(std::string(ChipVarInitPrefix) +
                               std::string(Var->getName()));
  assert(K && "chipstar::Module is missing a shadow kernel?");
  queueKernel(Q, K);
}

chipstar::CallbackData::CallbackData(hipStreamCallback_t TheCallbackF,
                                     void *TheCallbackArgs,
                                     chipstar::Queue *TheChipQueue)
    : ChipQueue(TheChipQueue), CallbackArgs(TheCallbackArgs),
      CallbackF(TheCallbackF) {}

void chipstar::CallbackData::execute(hipError_t ResultFromDependency) {
  CallbackF(ChipQueue, ResultFromDependency, CallbackArgs);
}

// DeviceVar
// ************************************************************************
chipstar::DeviceVar::~DeviceVar() { assert(!DevAddr_ && "Memory leak?"); }

// chipstar::AllocTracker
// ************************************************************************
chipstar::AllocationTracker::AllocationTracker(size_t GlobalMemSize,
                                               std::string Name)
    : GlobalMemSize(GlobalMemSize), TotalMemSize(0), MaxMemUsed(0) {
  Name_ = Name;
}

chipstar::AllocationTracker::~AllocationTracker() {
  for (auto *Member : AllocInfos_)
    delete Member;
}

chipstar::AllocationInfo *
chipstar::AllocationTracker::getAllocInfo(const void *Ptr) {
  {
    LOCK( // chipstar::AllocTracker::PtrToAllocInfo_
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

bool chipstar::AllocationTracker::reserveMem(size_t Bytes) {
  LOCK(AllocationTrackerMtx); // reading chipstar::AllocTracker::GlobalMemSize
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

bool chipstar::AllocationTracker::releaseMemReservation(unsigned long Bytes) {
  LOCK(AllocationTrackerMtx); // reading chipstar::AllocTracker::GlobalMemSize
  if (TotalMemSize >= Bytes) {
    TotalMemSize -= Bytes;
    return true;
  }

  return false;
}

void chipstar::AllocationTracker::recordAllocation(
    void *DevPtr, void *HostPtr, hipDevice_t Device, size_t Size,
    chipstar::HostAllocFlags Flags, hipMemoryType MemoryType) {
  chipstar::AllocationInfo *AllocInfo = new chipstar::AllocationInfo{
      DevPtr, HostPtr, Size, Flags, Device, false, MemoryType};
  LOCK(AllocationTrackerMtx); // writing chipstar::AllocTracker::PtrToAllocInfo_
                              // chipstar::AllocTracker::AllocInfos_
  // TODO AllocInfo turned into class and constructor take care of this
  if (MemoryType == hipMemoryTypeHost) {
    AllocInfo->HostPtr = AllocInfo->DevPtr;
    // Map onto host so that the data can be potentially initialized on host
    ::Backend->getActiveDevice()->getDefaultQueue()->MemMap(
        AllocInfo, chipstar::Queue::MEM_MAP_TYPE::HOST_WRITE);
  }

  if (MemoryType == hipMemoryTypeUnified)
    AllocInfo->HostPtr = AllocInfo->DevPtr;

  AllocInfos_.insert(AllocInfo);

  if (DevPtr) {
    assert(!PtrToAllocInfo_.count(DevPtr) &&
           "Device pointer already recorded!");
    PtrToAllocInfo_[DevPtr] = AllocInfo;
  }
  if (HostPtr) {
    assert(!PtrToAllocInfo_.count(DevPtr) && "Host pointer already recorded!");
    PtrToAllocInfo_[HostPtr] = AllocInfo;
  }

  logDebug("chipstar::AllocationTracker::recordAllocation size: {} HOST {} DEV "
           "{} TYPE {}",
           Size, HostPtr, DevPtr, (unsigned)MemoryType);
  return;
}

chipstar::AllocationInfo *
chipstar::AllocationTracker::getAllocInfoCheckPtrRanges(void *DevPtr) {
  LOCK(AllocationTrackerMtx); // chipstar::AllocTracker::PtrToAllocInfo_
  for (auto &Info : PtrToAllocInfo_) {
    chipstar::AllocationInfo *AllocInfo = Info.second;
    void *Start = AllocInfo->DevPtr;
    void *End = (char *)Start + AllocInfo->Size;

    if (Start <= DevPtr && DevPtr < End)
      return AllocInfo;
  }

  return nullptr;
}

// chipstar::Event
// ************************************************************************

chipstar::Event::Event(chipstar::Context *Ctx, chipstar::EventFlags Flags)
    : EventStatus_(EVENT_STATUS_INIT), Flags_(Flags), ChipContext_(Ctx),
      Msg("") {}

void chipstar::Event::releaseDependencies() {
  assert(!Deleted_ && "chipstar::Event use after delete!");
  LOCK(EventMtx); // chipstar::Event::DependsOnList
  DependsOnList.clear();
}

// CHIPModuleflags_
//*************************************************************************************
void chipstar::Module::consumeSPIRV() {
  FuncIL_ = (uint8_t *)Src_->getBinary().data();
  IlSize_ = Src_->getBinary().size();

  // dump the SPIR-V source into current directory if CHIP_DUMP_SPIRV is set
  // dump here prior to parsing in case parsing crashes
  dumpSpirv(Src_->getBinary());

  // Parse the SPIR-V fat binary to retrieve kernel function
  size_t NumWords = IlSize_ / 4;
  BinaryData_ = new uint32_t[NumWords + 1];
  std::memcpy(BinaryData_, FuncIL_, IlSize_);
  // Extract kernel function information
  bool Res = parseSPIR(BinaryData_, NumWords, FuncInfos_);
  delete[] BinaryData_;
  if (!Res) {
    CHIPERR_LOG_AND_THROW("SPIR-V parsing failed", hipErrorUnknown);
  }
}

chipstar::Module::~Module() {}

void chipstar::Module::addKernel(chipstar::Kernel *Kernel) {
  ChipKernels_.push_back(Kernel);
}

void chipstar::Module::compileOnce(chipstar::Device *ChipDevice) {
  std::call_once(Compiled_, &chipstar::Module::compile, this, ChipDevice);
}

chipstar::Kernel *chipstar::Module::findKernel(const std::string &Name) {
  auto KernelFound = std::find_if(ChipKernels_.begin(), ChipKernels_.end(),
                                  [&Name](chipstar::Kernel *Kernel) {
                                    return Kernel->getName().compare(Name) == 0;
                                  });
  return KernelFound == ChipKernels_.end() ? nullptr : *KernelFound;
}

chipstar::Kernel *chipstar::Module::getKernelByName(const std::string &Name) {
  auto *Kernel = findKernel(Name);
  if (!Kernel) {
    std::string Msg = "Failed to find kernel via kernel name: " + Name;
    CHIPERR_LOG_AND_THROW(Msg, hipErrorLaunchFailure);
  }
  return Kernel;
}

bool chipstar::Module::hasKernel(std::string Name) { return findKernel(Name); }

chipstar::Kernel *chipstar::Module::getKernel(const void *HostFPtr) {
  logDebug("{} chipstar::Module::getKernel({})", (void *)this, HostFPtr);
  for (auto &Kernel : ChipKernels_)
    logDebug("chip kernel: {} {}", Kernel->getHostPtr(), Kernel->getName());
  auto KernelFound = std::find_if(ChipKernels_.begin(), ChipKernels_.end(),
                                  [HostFPtr](chipstar::Kernel *Kernel) {
                                    return Kernel->getHostPtr() == HostFPtr;
                                  });
  if (KernelFound == ChipKernels_.end()) {
    std::string Msg = "Failed to find kernel via host pointer";
    CHIPERR_LOG_AND_THROW(Msg, hipErrorLaunchFailure);
  }

  return *KernelFound;
}

std::vector<chipstar::Kernel *> &chipstar::Module::getKernels() {
  return ChipKernels_;
}

chipstar::DeviceVar *chipstar::Module::getGlobalVar(const char *VarName) {
  auto VarFound = std::find_if(ChipVars_.begin(), ChipVars_.end(),
                               [VarName](chipstar::DeviceVar *Var) {
                                 return Var->getName().compare(VarName) == 0;
                               });
  if (VarFound == ChipVars_.end()) {
    logDebug("Failed to find global variable by name: {}",
             std::string(VarName));
    return nullptr;
  }

  return *VarFound;
}

hipError_t
chipstar::Module::allocateDeviceVariablesNoLock(chipstar::Device *Device,
                                                chipstar::Queue *Queue) {
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
  std::vector<std::pair<chipstar::DeviceVar *, CHIPVarInfo *>> VarInfos;
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

void chipstar::Module::prepareDeviceVariablesNoLock(chipstar::Device *Device,
                                                    chipstar::Queue *Queue) {
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

void chipstar::Module::invalidateDeviceVariablesNoLock() {
  DeviceVariablesInitialized_ = false;
}

void chipstar::Module::deallocateDeviceVariablesNoLock(
    chipstar::Device *Device) {
  invalidateDeviceVariablesNoLock();
  for (auto *Var : ChipVars_) {
    auto Err = Device->getContext()->free(Var->getDevAddr());
    (void)Err;
    Var->setDevAddr(nullptr);
  }
  DeviceVariablesAllocated_ = false;
}

SPVFuncInfo *chipstar::Module::findFunctionInfo(const std::string &FName) {
  return FuncInfos_.count(FName) ? FuncInfos_.at(FName).get() : nullptr;
}

// Kernel
//*************************************************************************************
chipstar::Kernel::Kernel(std::string HostFName, SPVFuncInfo *FuncInfo)
    : HostFName_(HostFName), FuncInfo_(FuncInfo) {}
chipstar::Kernel::~Kernel(){};
std::string chipstar::Kernel::getName() { return HostFName_; }
const void *chipstar::Kernel::getHostPtr() { return HostFPtr_; }
const void *chipstar::Kernel::getDevPtr() { return DevFPtr_; }

SPVFuncInfo *chipstar::Kernel::getFuncInfo() { return FuncInfo_; }

void chipstar::Kernel::setName(std::string HostFName) {
  HostFName_ = HostFName;
}
void chipstar::Kernel::setHostPtr(const void *HostFPtr) {
  HostFPtr_ = HostFPtr;
}
void chipstar::Kernel::setDevPtr(const void *DevFPtr) { DevFPtr_ = DevFPtr; }

// chipstar::ArgSpillBuffer
//*****************************************************************************

chipstar::ArgSpillBuffer::~ArgSpillBuffer() { (void)Ctx_->free(DeviceBuffer_); }

void chipstar::ArgSpillBuffer::computeAndReserveSpace(
    const SPVFuncInfo &KernelInfo) {
  size_t Offset = 0;
  size_t MaxAlignment = 1;
  auto Visitor = [&](const SPVFuncInfo::KernelArg &Arg) -> void {
    if (Arg.Kind != SPVTypeKind::PODByRef)
      return;
    // FIXME: Extract alignment requirement for the argument value
    //        from SPIR-V, store it in FuncInfo and read it
    //        here. Using now an arbitrarily chosen value.
    size_t Alignment = 32; // Chose sizeof(double4) as the alignment.
    Offset = roundUp(Offset, Alignment);
    ArgIndexToOffset_.insert(std::make_pair(Arg.Index, Offset));
    MaxAlignment = std::max<size_t>(MaxAlignment, Alignment);
    Offset += Arg.Size;
  };
  KernelInfo.visitKernelArgs(Visitor);

  Size_ = Offset;
  HostBuffer_ = std::make_unique<char[]>(Size_);
  DeviceBuffer_ = static_cast<char *>(
      Ctx_->allocate(Size_, MaxAlignment, hipMemoryTypeDevice));
}

void *chipstar::ArgSpillBuffer ::allocate(const SPVFuncInfo::Arg &Arg) {
  assert(HostBuffer_ && DeviceBuffer_ &&
         "Forgot to call computeAndReserveSpace()?");
  auto Offset = ArgIndexToOffset_[Arg.Index];
  auto *HostPtr = HostBuffer_.get() + Offset;
  assert(Arg.Data);
  std::memcpy(HostPtr, Arg.Data, Arg.Size);
  return DeviceBuffer_ + Offset;
}

// ExecItem
//*************************************************************************************
void chipstar::ExecItem::copyArgs(void **Args) {
  for (int i = 0; i < getNumArgs(); i++) {
    Args_.push_back(Args[i]);
  }
}

chipstar::ExecItem::ExecItem(dim3 GridDim, dim3 BlockDim, size_t SharedMem,
                             hipStream_t ChipQueue)
    : SharedMem_(SharedMem), GridDim_(GridDim), BlockDim_(BlockDim),
      ChipQueue_(static_cast<chipstar::Queue *>(ChipQueue)){};

dim3 chipstar::ExecItem::getBlock() { return BlockDim_; }
dim3 chipstar::ExecItem::getGrid() { return GridDim_; }
size_t chipstar::ExecItem::getSharedMem() { return SharedMem_; }
chipstar::Queue *chipstar::ExecItem::getQueue() { return ChipQueue_; }
// Device
//*************************************************************************************
chipstar::Device::Device(chipstar::Context *Ctx, int DeviceIdx)
    : Ctx_(Ctx), Idx_(DeviceIdx) {
  LegacyDefaultQueue = nullptr;
  PerThreadDefaultQueue = nullptr;
  // Avoid indeterminate values.
  std::memset(&HipDeviceProps_, 0, sizeof(HipDeviceProps_));
}

chipstar::Device::~Device() {
  LOCK(DeviceMtx); // chipstar::Device::ChipQueues_
  logDebug("~Device() {}", (void *)this);
  while (this->ChipQueues_.size() > 0) {
    delete ChipQueues_[0];
    ChipQueues_.erase(ChipQueues_.begin());
  }

  delete LegacyDefaultQueue;
  LegacyDefaultQueue = nullptr;
}
chipstar::Queue *chipstar::Device::getLegacyDefaultQueue() {
  return LegacyDefaultQueue;
}

chipstar::Queue *chipstar::Device::getDefaultQueue() {
#ifdef HIP_API_PER_THREAD_DEFAULT_STREAM
  return getPerThreadDefaultQueue();
#else
  return getLegacyDefaultQueue();
#endif
}

bool chipstar::Device::isPerThreadStreamUsed() {
  LOCK(DeviceMtx); // chipstar::Device::PerThreadStreamUsed
  return PerThreadStreamUsed_;
}

bool chipstar::Device::isPerThreadStreamUsedNoLock() {
  return PerThreadStreamUsed_;
}

void chipstar::Device::setPerThreadStreamUsed(bool Status) {
  LOCK(DeviceMtx); // chipstar::Device::PerThreadStreamUsed
  PerThreadStreamUsed_ = Status;
}

chipstar::Queue *chipstar::Device::getPerThreadDefaultQueue() {
  LOCK(DeviceMtx); // chipstar::Device::PerThreadStreamUsed
  return getPerThreadDefaultQueueNoLock();
}

chipstar::Queue *chipstar::Device::getPerThreadDefaultQueueNoLock() {
  if (!PerThreadDefaultQueue.get()) {
    logDebug("PerThreadDefaultQueue is null.. Creating a new queue.");
    PerThreadDefaultQueue =
        std::unique_ptr<chipstar::Queue>(::Backend->createCHIPQueue(this));
    PerThreadStreamUsed_ = true;
    PerThreadDefaultQueue.get()->PerThreadQueueForDevice = this;
  }

  return PerThreadDefaultQueue.get();
}

std::vector<chipstar::Kernel *> chipstar::Device::getKernels() {
  std::vector<chipstar::Kernel *> ChipKernels;
  for (auto &Kv : SrcModToCompiledMod_) {
    for (chipstar::Kernel *Kernel : Kv.second->getKernels())
      ChipKernels.push_back(Kernel);
  }
  return ChipKernels;
}

std::string chipstar::Device::getName() {
  return std::string(HipDeviceProps_.name);
}

void chipstar::Device::init() {
  LOCK(DeviceMtx) // chipstar::Device::LegacyDefaultQueue
  std::call_once(PropsPopulated_,
                 &chipstar::Device::populateDevicePropertiesImpl, this);
  if (!AllocTracker)
    AllocTracker = new chipstar::AllocationTracker(
        HipDeviceProps_.totalGlobalMem, HipDeviceProps_.name);

  chipstar::QueueFlags Flags;
  int Priority = 1; // TODO : set a default
  LegacyDefaultQueue = createQueue(Flags, Priority);
}

void chipstar::Device::copyDeviceProperties(hipDeviceProp_t *Prop) {
  logDebug("Device->copy_device_properties()");
  if (Prop)
    std::memcpy(Prop, &this->HipDeviceProps_, sizeof(hipDeviceProp_t));
}

chipstar::Context *chipstar::Device::getContext() { return Ctx_; }
int chipstar::Device::getDeviceId() { return Idx_; }

chipstar::DeviceVar *chipstar::Device::getStatGlobalVar(const void *HostPtr) {
  if (DeviceVarLookup_.count(HostPtr)) {
    auto *Var = DeviceVarLookup_[HostPtr];
    assert(Var->getDevAddr() && "Missing device pointer.");
    return Var;
  }
  return nullptr;
}

chipstar::DeviceVar *chipstar::Device::getGlobalVar(const void *HostPtr) {
  if (auto *Found = getDynGlobalVar(HostPtr))
    return Found;

  if (auto *Found = getStatGlobalVar(HostPtr))
    return Found;

  return nullptr;
}

int chipstar::Device::getAttr(hipDeviceAttribute_t Attr) {
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
        "Device::getAttr(hipDeviceAttributeCanUseStreamWaitValue path "
        "unimplemented",
        hipErrorTbd);
    break;
  default:
    CHIPERR_LOG_AND_THROW("Device::getAttr asked for an unkown attribute",
                          hipErrorInvalidValue);
  }
  return -1;
}

size_t chipstar::Device::getGlobalMemSize() {
  return HipDeviceProps_.totalGlobalMem;
}

void chipstar::Device::eraseModule(chipstar::Module *Module) {
  LOCK(DeviceMtx); // SrcModToCompiledMod_
  for (auto &Kv : SrcModToCompiledMod_)
    if (Kv.second == Module) {
      delete Module;
      SrcModToCompiledMod_.erase(Kv.first);
      break;
    }
}

void chipstar::Device::addQueue(chipstar::Queue *ChipQueue) {
  LOCK(DeviceMtx) // writing chipstar::Device::ChipQueues_
  logDebug("{} Device::addQueue({})", (void *)this, (void *)ChipQueue);

  auto QueueFound =
      std::find(ChipQueues_.begin(), ChipQueues_.end(), ChipQueue);
  if (QueueFound == ChipQueues_.end()) {
    ChipQueues_.push_back(ChipQueue);
  } else {
    CHIPERR_LOG_AND_THROW("Tried to add a queue to the backend which was "
                          "already present in the backend queue list",
                          hipErrorTbd);
  }
  logDebug("Queue {} added to the queue vector for device {} ",
           (void *)ChipQueue, (void *)this);

  return;
}

chipstar::Queue *
chipstar::Device::createQueueAndRegister(chipstar::QueueFlags Flags,
                                         int Priority) {

  auto ChipQueue = createQueue(Flags, Priority);
  // Add the queue handle to the device and the Backend
  addQueue(ChipQueue);
  return ChipQueue;
}

chipstar::Queue *
chipstar::Device::createQueueAndRegister(const uintptr_t *NativeHandles,
                                         const size_t NumHandles) {
  auto ChipQueue = createQueue(NativeHandles, NumHandles);
  // Add the queue handle to the device and the Backend
  addQueue(ChipQueue);
  return ChipQueue;
}

std::vector<chipstar::Queue *> &chipstar::Device::getQueues() {
  LOCK(DeviceMtx); // reading chipstar::Device::ChipQueues_
  return ChipQueues_;
}

hipError_t chipstar::Device::setPeerAccess(chipstar::Device *Peer, int Flags,
                                           bool CanAccessPeer) {
  UNIMPLEMENTED(hipSuccess);
}

int chipstar::Device::getPeerAccess(chipstar::Device *PeerDevice) {
  UNIMPLEMENTED(0);
}

void chipstar::Device::setCacheConfig(hipFuncCache_t Cfg) { UNIMPLEMENTED(); }

void chipstar::Device::setFuncCacheConfig(const void *Func,
                                          hipFuncCache_t Cfg) {
  UNIMPLEMENTED();
}

hipFuncCache_t chipstar::Device::getCacheConfig() {
  UNIMPLEMENTED(hipFuncCachePreferNone);
}

hipSharedMemConfig chipstar::Device::getSharedMemConfig() {
  UNIMPLEMENTED(hipSharedMemBankSizeDefault);
}

void chipstar::Device::removeContext(chipstar::Context *Context) {}

bool chipstar::Device::removeQueue(chipstar::Queue *ChipQueue) {
  /**
   * If commands are still executing on the specified stream, some may complete
   * execution before the queue is deleted. The queue may be destroyed while
   * some commands are still inflight, or may wait for all commands queued to
   * the stream before destroying it.
   *
   * Choosing not to call Queue->finish()
   */
  LOCK(DeviceMtx) // reading chipstar::Device::ChipQueues_
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

void chipstar::Device::setSharedMemConfig(hipSharedMemConfig Cfg) {
  UNIMPLEMENTED();
}

size_t chipstar::Device::getUsedGlobalMem() {
  return AllocTracker->TotalMemSize;
}

bool chipstar::Device::hasPCIBusId(int PciDomainID, int PciBusID,
                                   int PciDeviceID) {
  auto T1 = this->HipDeviceProps_.pciBusID == PciBusID;
  auto T2 = this->HipDeviceProps_.pciDomainID == PciDomainID;
  auto T3 = this->HipDeviceProps_.pciDeviceID == PciDeviceID;

  return (T1 && T2 && T3);
}

/// Prepares device variables for a module for which the host pointer
/// is member of.
void chipstar::Device::prepareDeviceVariables(HostPtr Ptr) {
  if (auto *Mod = getOrCreateModule(Ptr)) {
    LOCK(DeviceVarMtx); // chipstar::Module::prepareDeviceVariablesNoLock()
    logDebug("Prepare variables in module {}", static_cast<const void *>(Mod));
    Mod->prepareDeviceVariablesNoLock(this, getDefaultQueue());
  }
}

void chipstar::Device::invalidateDeviceVariables() {
  // Device::SrcModToCompiledMod_
  // chipstar::Module::invalidateDeviceVariablesNoLock()
  LOCK(DeviceVarMtx); // Device::SrcModToCompiledMod_
  logTrace("invalidate device variables.");
  for (auto &Kv : SrcModToCompiledMod_)
    Kv.second->invalidateDeviceVariablesNoLock();
}

void chipstar::Device::deallocateDeviceVariables() {
  // chipstar::Device::SrcModToCompiledMod_
  // chipstar::Module::deallocateDeviceVariablesNoLock()
  LOCK(DeviceVarMtx); // chipstar::Device::SrcModToCompiledMod_
  logTrace("Deallocate storage for device variables.");
  for (auto &Kv : SrcModToCompiledMod_)
    Kv.second->deallocateDeviceVariablesNoLock(this);
}

/// Get compiled module associated with the host pointer 'Ptr'. Return
/// nullptr if 'Ptr' is not associated with any module.
chipstar::Module *chipstar::Device::getOrCreateModule(HostPtr Ptr) {
  {
    LOCK(DeviceVarMtx); // chipstar::Device::HostPtrToCompiledMod_
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
    LOCK(DeviceVarMtx); // chipstar::Device::HostPtrToCompiledMod_
    assert((!Mod || (HostPtrToCompiledMod_.count(Ptr) &&
                     HostPtrToCompiledMod_[Ptr] == Mod)) &&
           "Forgot to map the host pointers");
  }
#endif

  return Mod;
}

/// Get compiled module for the source module 'SrcMod'.
chipstar::Module *chipstar::Device::getOrCreateModule(const SPVModule &SrcMod) {
  LOCK(DeviceVarMtx); // chipstar::Device::SrcModToCompiledMod_
                      // chipstar::Device::HostPtrToCompiledMod_
                      // chipstar::Device::DeviceVarLookup_

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
    chipstar::Kernel *Kernel = Module->getKernelByName(NameTmp);
    assert(Kernel && "chipstar::Kernel went missing?");
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
    auto *Var = new chipstar::DeviceVar(&Info);
    Module->addDeviceVariable(Var);

    DeviceVarLookup_.insert(std::make_pair(Info.Ptr, Var));
    HostPtrToCompiledMod_[Info.Ptr] = Module;
  }

  SrcModToCompiledMod_.insert(std::make_pair(&SrcMod, Module));
  return Module;
}

// Context
//*************************************************************************************
chipstar::Context::Context() {}
chipstar::Context::~Context() { logDebug("~Context() {}", (void *)this); }

void chipstar::Context::syncQueues(chipstar::Queue *TargetQueue) {
  auto Dev = ::Backend->getActiveDevice();
  LOCK(Dev->DeviceMtx); // chipstar::Device::ChipQueues_ via getQueuesNoLock()

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
  std::vector<chipstar::Queue *> QueuesToSyncWith;

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
  std::vector<std::shared_ptr<chipstar::Event>> EventsToWaitOn;

  std::shared_ptr<chipstar::Event> SyncQueuesEvent;
  if (TargetQueue == DefaultQueue) {
    for (auto &q : QueuesToSyncWith) {
      auto Ev = q->getLastEvent();
      if (Ev)
        EventsToWaitOn.push_back(Ev);
    }
    SyncQueuesEvent = TargetQueue->enqueueBarrierImpl(EventsToWaitOn);
    SyncQueuesEvent->Msg = "barrierSyncQueue";
    TargetQueue->updateLastEvent(SyncQueuesEvent);
  } else { // blocking stream must wait until default stream is done
    auto Ev = DefaultQueue->getLastEvent();
    if (Ev)
      EventsToWaitOn.push_back(Ev);
    SyncQueuesEvent = TargetQueue->enqueueBarrierImpl(EventsToWaitOn);
    SyncQueuesEvent->Msg = "barrierSyncQueue";
    TargetQueue->updateLastEvent(SyncQueuesEvent);
  }
  ::Backend->trackEvent(SyncQueuesEvent);
}

chipstar::Device *chipstar::Context::getDevice() {
  assert(this->ChipDevice_);
  return ChipDevice_;
}

void *chipstar::Context::allocate(size_t Size, hipMemoryType MemType) {
  return allocate(Size, 0, MemType, chipstar::HostAllocFlags());
}

void *chipstar::Context::allocate(size_t Size, size_t Alignment,
                                  hipMemoryType MemType) {
  return allocate(Size, Alignment, MemType, chipstar::HostAllocFlags());
}

void *chipstar::Context::allocate(size_t Size, size_t Alignment,
                                  hipMemoryType MemType,
                                  chipstar::HostAllocFlags Flags) {
  void *AllocatedPtr, *HostPtr = nullptr;
  // TOOD hipCtx - use the device with which this context is associated
  chipstar::Device *ChipDev = ::Backend->getActiveDevice();
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

  assert(ChipDev->AllocTracker &&
         "chipstar::AllocationTracker was not created!");
  if (!ChipDev->AllocTracker->reserveMem(Size))
    return nullptr;
  AllocatedPtr = allocateImpl(Size, Alignment, MemType);
  if (AllocatedPtr == nullptr)
    ChipDev->AllocTracker->releaseMemReservation(Size);

  ChipDev->AllocTracker->recordAllocation(
      AllocatedPtr, HostPtr, ChipDev->getDeviceId(), Size, Flags, MemType);

  return AllocatedPtr;
}

unsigned int chipstar::Context::getFlags() { return Flags_; }

void chipstar::Context::setFlags(unsigned int Flags) { Flags_ = Flags; }

void chipstar::Context::reset() {
  logDebug("Resetting Context: deleting allocations");
  // Free all allocations in this context
  for (auto &Ptr : AllocatedPtrs_)
    freeImpl(Ptr);

  auto Dev = getDevice();
  // Free all the memory reservations on each device
  Dev->AllocTracker->releaseMemReservation(Dev->AllocTracker->TotalMemSize);
  AllocatedPtrs_.clear();

  getDevice()->reset();
}

hipError_t chipstar::Context::free(void *Ptr) {
  chipstar::Device *ChipDev = ::Backend->getActiveDevice();
  chipstar::AllocationInfo *AllocInfo =
      ChipDev->AllocTracker->getAllocInfo(Ptr);
  if (!AllocInfo)
    // HIP API doc says we should return hipErrorInvalidDevicePointer but HIP
    // test suite excepts hipErrorInvalidValue. Go with the latter.
    return hipErrorInvalidValue;

  ChipDev->AllocTracker->releaseMemReservation(AllocInfo->Size);
  ChipDev->AllocTracker->eraseRecord(AllocInfo);
  freeImpl(Ptr);
  return hipSuccess;
}

// Backend
//*************************************************************************************
int chipstar::Backend::getPerThreadQueuesActive() {
  LOCK(
      ::Backend->BackendMtx); // Prevent adding/removing devices while iterating
  int Active = 0;
  for (auto Dev : getDevices()) {
    if (Dev->isPerThreadStreamUsed()) {
      Active++;
    }
  }
  return Active;
}
int chipstar::Backend::getQueuePriorityRange() {
  assert(MinQueuePriority_);
  return MinQueuePriority_;
}

std::string chipstar::Backend::getJitFlags() {
  std::string Flags;
  if (CustomJitFlags != "") {
    Flags = CustomJitFlags;
  } else {
    Flags = getDefaultJitFlags();
  }
  logDebug("JIT compiler flags: {}", Flags);
  return Flags;
}

chipstar::Backend::Backend() {
  logDebug("Backend Base Constructor");
  Logger = spdlog::default_logger();
};

chipstar::Backend::~Backend() {
  logDebug("Backend Destructor. Deleting all pointers.");
  //   assert(Events.size() == 0);
  Events.clear();
  UserEvents.clear();
  for (auto &Ctx : ChipContexts) {
    ::Backend->removeContext(Ctx);
    delete Ctx;
  }
}

void chipstar::Backend::trackEvent(
    const std::shared_ptr<chipstar::Event> &Event) {
  LOCK(::Backend->EventsMtx); // trackImpl Backend::Events
  LOCK(Event->EventMtx);      // writing bool chipstar::Event::TrackCalled_
  //   assert(!isUserEvent() && "Attemped to track a user event!");
  //   assert(!Deleted_ && "chipstar::Event use after delete!");
  //   assert(!TrackCalled_ && "chipstar::Event already tracked!");

  logDebug("Tracking chipstar::Event {} in Backend::Events", (void *)this);
  // check if event is already present in Backend->Events
  for (auto &E : ::Backend->Events) {
      if (E.get() == Event.get()) {
      logError("chipstar::Event {} already tracked in Backend::Events",
                  (void *)this);
      assert(false && "chipstar::Event already tracked in Backend::Events");
      }
  }
  ::Backend->Events.push_back(Event);
  Event->markTracked();
}

void chipstar::Backend::waitForThreadExit() {
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
      auto NumPerThreadQueuesActive = ::Backend->getPerThreadQueuesActive();
      if (!NumPerThreadQueuesActive)
        break;

      logDebug("Backend::waitForThreadExit() per-thread queues still active "
               "{}. Sleeping for 1s..",
               NumPerThreadQueuesActive);
    }
    sleep(1);
  }

  // Cleanup all queues
  {
    LOCK(::Backend->BackendMtx); // prevent devices from being destrpyed
    for (auto Dev : ::Backend->getDevices()) {
      Dev->getLegacyDefaultQueue()->updateLastEvent(nullptr);
      LOCK(Dev->DeviceMtx);       // CHIPBackend::Events
      LOCK(::Backend->EventsMtx); // CHIPBackend::Events
      int NumQueues = Dev->getQueuesNoLock().size();
      if (NumQueues) {
        logWarn("Not all user created streams have been destoyed... Queues "
                "remaining: {}",
                NumQueues);
        logWarn("Make sure to call hipStreamDestroy() for all queues that have "
                "been created via hipStreamCreate()");
        logWarn("Removing user-created streams without calling a destructor");
        Dev->getQueuesNoLock().clear();
        if (::Backend->Events.size()) {
          logWarn("Clearing chipstar::Event list {}", ::Backend->Events.size());
          ::Backend->Events.clear();
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
void chipstar::Backend::initialize(std::string PlatformStr,
                                   std::string DeviceTypeStr,
                                   std::string DeviceIdStr) {
  initializeImpl(PlatformStr, DeviceTypeStr, DeviceIdStr);
  CustomJitFlags = readEnvVar("CHIP_JIT_FLAGS", false);
  if (ChipContexts.size() == 0) {
    std::string Msg = "No CHIPContexts were initialized";
    CHIPERR_LOG_AND_THROW(Msg, hipErrorInitializationError);
  }

  PrimaryContext = ChipContexts[0];
  setActiveContext(
      ChipContexts[0]); // pushes primary context to context stack for thread 0
}

void chipstar::Backend::setActiveContext(chipstar::Context *ChipContext) {
  ChipCtxStack.push(ChipContext);
}

void chipstar::Backend::setActiveDevice(chipstar::Device *ChipDevice) {
  ::Backend->setActiveContext(ChipDevice->getContext());
}

chipstar::Context *chipstar::Backend::getActiveContext() {
  // assert(ChipCtxStack.size() > 0 && "Context stack is empty");
  if (ChipCtxStack.size() == 0) {
    logDebug("Context stack is empty for thread {}", pthread_self());
    ChipCtxStack.push(PrimaryContext);
  }
  return ChipCtxStack.top();
};

chipstar::Device *chipstar::Backend::getActiveDevice() {
  chipstar::Context *Ctx = getActiveContext();
  return Ctx->getDevice();
};

std::vector<chipstar::Device *> chipstar::Backend::getDevices() {
  std::vector<chipstar::Device *> Devices;
  for (auto Ctx : ChipContexts) {
    Devices.push_back(Ctx->getDevice());
  }

  return Devices;
}

size_t chipstar::Backend::getNumDevices() { return ChipContexts.size(); }

void chipstar::Backend::removeContext(chipstar::Context *ChipContext) {
  auto ContextFound =
      std::find(ChipContexts.begin(), ChipContexts.end(), ChipContext);
  if (ContextFound != ChipContexts.end()) {
    ChipContexts.erase(ContextFound);
  }
}

void chipstar::Backend::addContext(chipstar::Context *ChipContext) {
  ChipContexts.push_back(ChipContext);
}

hipError_t chipstar::Backend::configureCall(dim3 Grid, dim3 Block,
                                            size_t SharedMem,
                                            hipStream_t ChipQueue) {
  logDebug("Backend->configureCall(grid=({},{},{}), block=({},{},{}), "
           "shared={}, q={}",
           Grid.x, Grid.y, Grid.z, Block.x, Block.y, Block.z, SharedMem,
           (void *)ChipQueue);
  chipstar::ExecItem *ExecItem =
      ::Backend->createExecItem(Grid, Block, SharedMem, ChipQueue);
  ChipExecStack.push(ExecItem);

  return hipSuccess;
}

chipstar::Device *
chipstar::Backend::findDeviceMatchingProps(const hipDeviceProp_t *Props) {
  chipstar::Device *MatchedDevice = nullptr;
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

chipstar::Queue *chipstar::Backend::findQueue(chipstar::Queue *ChipQueue) {
  auto Dev = ::Backend->getActiveDevice();
  LOCK(Dev->DeviceMtx); // chipstar::Device::ChipQueues_ via getQueuesNoLock()

  if (ChipQueue == hipStreamPerThread) {
    return Dev->getPerThreadDefaultQueueNoLock();
  } else if (ChipQueue == hipStreamLegacy) {
    return Dev->getLegacyDefaultQueue();
  } else if (ChipQueue == nullptr) {
    return Dev->getDefaultQueue();
  }
  std::vector<chipstar::Queue *> AllQueues;
  if (Dev->isPerThreadStreamUsedNoLock())
    AllQueues.push_back(Dev->getPerThreadDefaultQueueNoLock());
  AllQueues.push_back(Dev->getLegacyDefaultQueue());

  for (auto &Dev : Dev->getQueuesNoLock()) {
    AllQueues.push_back(Dev);
  }

  auto QueueFound = std::find(AllQueues.begin(), AllQueues.end(), ChipQueue);
  if (QueueFound == AllQueues.end())
    CHIPERR_LOG_AND_THROW("Backend::findQueue() was given a non-nullptr "
                          "queue but this queue "
                          "was not found among the backend queues.",
                          hipErrorTbd);
  return *QueueFound;
}

// Queue
//*************************************************************************************
chipstar::Queue::Queue(chipstar::Device *ChipDevice, chipstar::QueueFlags Flags,
                       int Priority)
    : Priority_(Priority), QueueFlags_(Flags), ChipDevice_(ChipDevice) {
  ChipContext_ = ChipDevice->getContext();
  logDebug("Queue() {}", (void *)this);
};

chipstar::Queue::Queue(chipstar::Device *ChipDevice, chipstar::QueueFlags Flags)
    : Queue(ChipDevice, Flags, 0){};

chipstar::Queue::~Queue() {
  updateLastEvent(nullptr);
  if (PerThreadQueueForDevice) {
    PerThreadQueueForDevice->setPerThreadStreamUsed(false);
  }
};

///////// Enqueue Operations //////////
hipError_t chipstar::Queue::memCopy(void *Dst, const void *Src, size_t Size) {
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif
  std::shared_ptr<chipstar::Event> ChipEvent;
  // Scope this so that we release mutex for finish()
  {
    auto AllocInfoDst =
        ::Backend->getActiveDevice()->AllocTracker->getAllocInfo(Dst);
    auto AllocInfoSrc =
        ::Backend->getActiveDevice()->AllocTracker->getAllocInfo(Src);
    if (AllocInfoDst && AllocInfoDst->MemoryType == hipMemoryTypeHost)
      ::Backend->getActiveDevice()->getDefaultQueue()->MemUnmap(AllocInfoDst);
    if (AllocInfoSrc && AllocInfoSrc->MemoryType == hipMemoryTypeHost)
      ::Backend->getActiveDevice()->getDefaultQueue()->MemUnmap(AllocInfoSrc);

    ChipEvent = memCopyAsyncImpl(Dst, Src, Size);

    if (AllocInfoDst && AllocInfoDst->MemoryType == hipMemoryTypeHost)
      ::Backend->getActiveDevice()->getDefaultQueue()->MemMap(
          AllocInfoDst, chipstar::Queue::MEM_MAP_TYPE::HOST_READ_WRITE);
    if (AllocInfoSrc && AllocInfoSrc->MemoryType == hipMemoryTypeHost)
      ::Backend->getActiveDevice()->getDefaultQueue()->MemMap(
          AllocInfoSrc, chipstar::Queue::MEM_MAP_TYPE::HOST_READ_WRITE);

    ChipEvent->Msg = "memCopy";
    updateLastEvent(ChipEvent);
    this->finish();
  }
  ::Backend->trackEvent(ChipEvent);

  return hipSuccess;
}
void chipstar::Queue::memCopyAsync(void *Dst, const void *Src, size_t Size) {
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif
  std::shared_ptr<chipstar::Event> ChipEvent;
  // Scope this so that we release mutex for finish()
  {
    auto AllocInfoDst =
        ::Backend->getActiveDevice()->AllocTracker->getAllocInfo(Dst);
    auto AllocInfoSrc =
        ::Backend->getActiveDevice()->AllocTracker->getAllocInfo(Src);
    if (AllocInfoDst && AllocInfoDst->MemoryType == hipMemoryTypeHost)
      this->MemUnmap(AllocInfoDst);
    if (AllocInfoSrc && AllocInfoSrc->MemoryType == hipMemoryTypeHost)
      this->MemUnmap(AllocInfoSrc);

    ChipEvent = memCopyAsyncImpl(Dst, Src, Size);

    if (AllocInfoDst && AllocInfoDst->MemoryType == hipMemoryTypeHost)
      this->MemMap(AllocInfoDst,
                   chipstar::Queue::MEM_MAP_TYPE::HOST_READ_WRITE);
    if (AllocInfoSrc && AllocInfoSrc->MemoryType == hipMemoryTypeHost)
      this->MemMap(AllocInfoSrc,
                   chipstar::Queue::MEM_MAP_TYPE::HOST_READ_WRITE);

    ChipEvent->Msg = "memCopyAsync";
    updateLastEvent(ChipEvent);
  }
  ::Backend->trackEvent(ChipEvent);
}

void chipstar::Queue::memFill(void *Dst, size_t Size, const void *Pattern,
                              size_t PatternSize) {
  {
#ifdef ENFORCE_QUEUE_SYNC
    ChipContext_->syncQueues(this);
#endif

    std::shared_ptr<chipstar::Event> ChipEvent =
        memFillAsyncImpl(Dst, Size, Pattern, PatternSize);
    ChipEvent->Msg = "memFill";
    updateLastEvent(ChipEvent);
    ::Backend->trackEvent(ChipEvent);
    this->finish();
  }
  return;
}

void chipstar::Queue::memFillAsync(void *Dst, size_t Size, const void *Pattern,
                                   size_t PatternSize) {
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif

  std::shared_ptr<chipstar::Event> ChipEvent =
      memFillAsyncImpl(Dst, Size, Pattern, PatternSize);
  ChipEvent->Msg = "memFillAsync";
  updateLastEvent(ChipEvent);
  ::Backend->trackEvent(ChipEvent);
}
void chipstar::Queue::memCopy2D(void *Dst, size_t DPitch, const void *Src,
                                size_t SPitch, size_t Width, size_t Height) {
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif
  std::shared_ptr<chipstar::Event> ChipEvent =
      memCopy2DAsyncImpl(Dst, DPitch, Src, SPitch, Width, Height);
  ChipEvent->Msg = "memCopy2D";
  finish();
  updateLastEvent(ChipEvent);
  ::Backend->trackEvent(ChipEvent);
}

void chipstar::Queue::memCopy2DAsync(void *Dst, size_t DPitch, const void *Src,
                                     size_t SPitch, size_t Width,
                                     size_t Height) {
  {
#ifdef ENFORCE_QUEUE_SYNC
    ChipContext_->syncQueues(this);
#endif
    std::shared_ptr<chipstar::Event> ChipEvent =
        memCopy2DAsyncImpl(Dst, DPitch, Src, SPitch, Width, Height);
    ChipEvent->Msg = "memCopy2DAsync";
    updateLastEvent(ChipEvent);
    ::Backend->trackEvent(ChipEvent);
    this->finish();
  }
  return;
}

void chipstar::Queue::memCopy3D(void *Dst, size_t DPitch, size_t DSPitch,
                                const void *Src, size_t SPitch, size_t SSPitch,
                                size_t Width, size_t Height, size_t Depth) {
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif

  std::shared_ptr<chipstar::Event> ChipEvent = memCopy3DAsyncImpl(
      Dst, DPitch, DSPitch, Src, SPitch, SSPitch, Width, Height, Depth);
  ChipEvent->Msg = "memCopy3D";
  finish();
  updateLastEvent(ChipEvent);
  ::Backend->trackEvent(ChipEvent);
}

void chipstar::Queue::memCopy3DAsync(void *Dst, size_t DPitch, size_t DSPitch,
                                     const void *Src, size_t SPitch,
                                     size_t SSPitch, size_t Width,
                                     size_t Height, size_t Depth) {
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif

  std::shared_ptr<chipstar::Event> ChipEvent = memCopy3DAsyncImpl(
      Dst, DPitch, DSPitch, Src, SPitch, SSPitch, Width, Height, Depth);
  ChipEvent->Msg = "memCopy3DAsync";
  updateLastEvent(ChipEvent);
  ::Backend->trackEvent(ChipEvent);
}

void chipstar::Queue::updateLastNode(CHIPGraphNode *NewNode) {
  if (LastNode_ != nullptr) {
    NewNode->addDependency(LastNode_);
  }
  LastNode_ = NewNode;
}

void chipstar::Queue::initCaptureGraph() { CaptureGraph_ = new CHIPGraph(); }

std::shared_ptr<chipstar::Event>
chipstar::Queue::RegisteredVarCopy(chipstar::ExecItem *ExecItem,
                                   MANAGED_MEM_STATE ExecState) {

  // TODO: Inspect kernel code for indirect allocation accesses. If
  //       the kernel does not have any, we only need inspect kernels
  //       pointer arguments for allocations to be synchronized.

  std::vector<std::shared_ptr<chipstar::Event>> CopyEvents;
  auto PreKernel = ExecState == MANAGED_MEM_STATE::PRE_KERNEL;
  auto &AllocTracker = ::Backend->getActiveDevice()->AllocTracker;
  auto ArgVisitor = [&](const chipstar::AllocationInfo &AllocInfo) -> void {
    if (AllocInfo.MemoryType == hipMemoryTypeHost) {
      logDebug("Sync host memory {} ({})", AllocInfo.HostPtr,
               (PreKernel ? "Unmap" : "Map"));
      if (PreKernel)
        MemUnmap(&AllocInfo);
      else
        MemMap(&AllocInfo, chipstar::Queue::MEM_MAP_TYPE::HOST_READ_WRITE);
    } else if (AllocInfo.HostPtr &&
               AllocInfo.MemoryType == hipMemoryTypeManaged) {
      void *Src = PreKernel ? AllocInfo.HostPtr : AllocInfo.DevPtr;
      void *Dst = PreKernel ? AllocInfo.DevPtr : AllocInfo.HostPtr;
      logDebug("Sync managed memory {} -> {} ({})", Src, Dst,
               (PreKernel ? "host-to-device" : "device-to-host"));
      CopyEvents.push_back(this->memCopyAsyncImpl(Dst, Src, AllocInfo.Size));
    }
  };
  AllocTracker->visitAllocations(ArgVisitor);

  if (CopyEvents.empty())
    return nullptr;

  // We don't need to updateLastEvent because this function is always part of
  // the kernel launch sequence which update the last event anyway
  // chipstar::Event* BackEvent = CopyEvents.back();
  //   updateLastEvent(std::shared_ptr<chipstar::Event>(BackEvent));

  for (auto Ev : CopyEvents)
    ::Backend->trackEvent(Ev);

  return CopyEvents.back();
}

void chipstar::Queue::launch(chipstar::ExecItem *ExItem) {
  std::stringstream InfoStr;
  InfoStr << "\nLaunching kernel " << ExItem->getKernel()->getName() << "\n";
  InfoStr << "GridDim: <" << ExItem->getGrid().x << ", " << ExItem->getGrid().y
          << ", " << ExItem->getGrid().z << ">";
  InfoStr << " BlockDim: <" << ExItem->getBlock().x << ", "
          << ExItem->getBlock().y << ", " << ExItem->getBlock().z << ">\n";
  InfoStr << "SharedMem: " << ExItem->getSharedMem() << "\n";

  const auto &FuncInfo = *ExItem->getKernel()->getFuncInfo();
  InfoStr << "NumArgs: " << FuncInfo.getNumKernelArgs() << "\n";
  auto Visitor = [&](const SPVFuncInfo::KernelArg &Arg) -> void {
    InfoStr << "Arg " << Arg.Index << ": " << Arg.getKindAsString() << " "
            << Arg.Size << " " << Arg.Data;
    if (Arg.Kind == SPVTypeKind::Pointer && !Arg.isWorkgroupPtr()) {
      void *PtrVal = *static_cast<void **>(const_cast<void *>(Arg.Data));
      InfoStr << " (" << PtrVal << ")";
    }
    InfoStr << "\n";
  };
  FuncInfo.visitKernelArgs(ExItem->getArgs(), Visitor);

  // Making this log info since hipLaunchKernel doesn't know enough about args
  logInfo("{}", InfoStr.str());

#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif

  auto TotalThreadsPerBlock =
      ExItem->getBlock().x * ExItem->getBlock().y * ExItem->getBlock().z;
  auto DeviceProps = getDevice()->getDeviceProps();
  auto MaxTotalThreadsPerBlock = DeviceProps.maxThreadsPerBlock;

  if (TotalThreadsPerBlock > MaxTotalThreadsPerBlock) {
    logCritical("Requested total local size {} exceeds HW limit {}",
                TotalThreadsPerBlock, MaxTotalThreadsPerBlock);
    CHIPERR_LOG_AND_THROW("Requested local size exceeds HW max",
                          hipErrorLaunchFailure);
  }

  if (ExItem->getBlock().x > DeviceProps.maxThreadsDim[0] ||
      ExItem->getBlock().y > DeviceProps.maxThreadsDim[1] ||
      ExItem->getBlock().z > DeviceProps.maxThreadsDim[2]) {
    logCritical(
        "Requested local size dimension ({}, {}, {}) exceeds max ({}, {}, {})",
        ExItem->getBlock().x, ExItem->getBlock().y, ExItem->getBlock().z,
        DeviceProps.maxThreadsDim[0], DeviceProps.maxThreadsDim[1],
        DeviceProps.maxThreadsDim[2]);
    CHIPERR_LOG_AND_THROW("Requested local size exceeds HW max",
                          hipErrorLaunchFailure);
  }

  std::shared_ptr<chipstar::Event> RegisteredVarInEvent =
      RegisteredVarCopy(ExItem, MANAGED_MEM_STATE::PRE_KERNEL);
  std::shared_ptr<chipstar::Event> LaunchEvent = launchImpl(ExItem);
  std::shared_ptr<chipstar::Event> RegisteredVarOutEvent =
      RegisteredVarCopy(ExItem, MANAGED_MEM_STATE::POST_KERNEL);

  RegisteredVarOutEvent ? updateLastEvent(RegisteredVarOutEvent)
                        : updateLastEvent(LaunchEvent);

  ::Backend->trackEvent(LaunchEvent);
}

std::shared_ptr<chipstar::Event> chipstar::Queue::enqueueBarrier(
    const std::vector<std::shared_ptr<chipstar::Event>> &EventsToWaitFor) {
  std::shared_ptr<chipstar::Event> ChipEvent =
      std::shared_ptr<chipstar::Event>(enqueueBarrierImpl(EventsToWaitFor));
  ChipEvent->Msg = "enqueueBarrier";
  updateLastEvent(ChipEvent);
  ::Backend->trackEvent(ChipEvent);
  return ChipEvent;
}
std::shared_ptr<chipstar::Event> chipstar::Queue::enqueueMarker() {
  std::shared_ptr<chipstar::Event> ChipEvent =
      std::shared_ptr<chipstar::Event>(enqueueMarkerImpl());
  ChipEvent->Msg = "enqueueMarker";
  updateLastEvent(ChipEvent);
  ::Backend->trackEvent(ChipEvent);
  return ChipEvent;
}

void chipstar::Queue::memPrefetch(const void *Ptr, size_t Count) {
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif

  std::shared_ptr<chipstar::Event> ChipEvent =
      std::shared_ptr<chipstar::Event>(memPrefetchImpl(Ptr, Count));
  ChipEvent->Msg = "memPrefetch";
  updateLastEvent(ChipEvent);
  ::Backend->trackEvent(ChipEvent);
}

void chipstar::Queue::launchKernel(chipstar::Kernel *ChipKernel, dim3 NumBlocks,
                                   dim3 DimBlocks, void **Args,
                                   size_t SharedMemBytes) {
  LOCK(
      ::Backend->BackendMtx); // Prevent the breakup of RegisteredVarCopy in&out
  chipstar::ExecItem *ExItem =
      ::Backend->createExecItem(NumBlocks, DimBlocks, SharedMemBytes, this);
  ExItem->setKernel(ChipKernel);
  ExItem->copyArgs(Args);
  ExItem->setupAllArgs();
  launch(ExItem);
  delete ExItem;
}

///////// End Enqueue Operations //////////

chipstar::Device *chipstar::Queue::getDevice() {
  if (ChipDevice_ == nullptr) {
    std::string Msg = "chip_device is null";
    CHIPERR_LOG_AND_THROW(Msg, hipErrorLaunchFailure);
  }

  return ChipDevice_;
}

chipstar::QueueFlags chipstar::Queue::getFlags() { return QueueFlags_; }
int chipstar::Queue::getPriority() { return Priority_; }
void chipstar::Queue::addCallback(hipStreamCallback_t Callback,
                                  void *UserData) {
  chipstar::CallbackData *Callbackdata =
      ::Backend->createCallbackData(Callback, UserData, this);

  {
    LOCK(::Backend->CallbackQueueMtx); // Backend::CallbackQueue
    ::Backend->CallbackQueue.push(Callbackdata);
  }

  return;
}

//   template <class GraphNodeType, class... ArgTypes>
//   bool chipstar::Queue::captureIntoGraph(ArgTypes... ArgsPack) {
//     if (getCaptureStatus() == hipStreamCaptureStatusActive) {
//       auto Graph = getCaptureGraph();
//       auto Node = new GraphNodeType(ArgsPack...);
//       updateLastNode(Node);
//       Graph->addNode(Node);
//       return true;
//     }
//     return false;
//   }

CHIPGraph *chipstar::Queue::getCaptureGraph() const {
  return static_cast<CHIPGraph *>(CaptureGraph_);
}