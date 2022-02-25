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
/// the give n allocation.
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

CHIPCallbackData::CHIPCallbackData(hipStreamCallback_t CallbackF,
                                   void *CallbackArgs, CHIPQueue *ChipQueue)
    : CallbackF(CallbackF), CallbackArgs(CallbackArgs), ChipQueue(ChipQueue) {}

// void CHIPEventMonitor::monitor() {
//   logDebug("CHIPEventMonitor::monitor()");
//   CHIPCallbackData *CallbackData;
//   while (Backend->getCallback(&CallbackData)) {
//     CallbackData->GpuReady->updateFinishStatus();
//     if (CallbackData->GpuReady->getEventStatus() != EVENT_STATUS_RECORDED)
//       continue;
//     CallbackData->execute(hipSuccess);
//     CallbackData->CpuCallbackComplete->hostSignal();
//     CallbackData->GpuAck->wait();
//     delete CallbackData;
//   }

//   // no more callback events left, free up the thread
//   delete this;
//   pthread_yield();
// }

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
  // TODO free up all the pointers in ptr_set
  UNIMPLEMENTED();
}

AllocationInfo *CHIPAllocationTracker::getByHostPtr(const void *HostPtr) {
  auto Found = HostToDev_.find(const_cast<void *>(HostPtr));
  if (Found == HostToDev_.end()) {
    return nullptr;
  }
  return getByDevPtr(Found->second);
}
AllocationInfo *CHIPAllocationTracker::getByDevPtr(const void *DevPtr) {
  auto Ptr = const_cast<void *>(DevPtr);
  logDebug("dev_to_allocation_info size: {}", DevToAllocInfo_.size());
  auto Count = DevToAllocInfo_.count(Ptr);
  if (Count == 0)
    return nullptr;

  return &DevToAllocInfo_[const_cast<void *>(DevPtr)];
}

bool CHIPAllocationTracker::reserveMem(size_t Bytes) {
  std::lock_guard<std::mutex> Lock(Mtx_);
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
  std::lock_guard<std::mutex> Lock(Mtx_);
  if (TotalMemSize >= Bytes) {
    TotalMemSize -= Bytes;
    return true;
  }

  return false;
}

void CHIPAllocationTracker::recordAllocation(void *DevPtr, size_t Size) {
  AllocationInfo AllocInfo{DevPtr, Size};
  DevToAllocInfo_[DevPtr] = AllocInfo;
  logDebug("CHIPAllocationTracker::recordAllocation size: {}",
           DevToAllocInfo_.size());
  return;
}

// CHIPEvent
// ************************************************************************

void CHIPEvent::recordStream(CHIPQueue *ChipQueue) {
  logDebug("CHIPEvent::recordStream()");
  std::lock_guard<std::mutex> Lock(Mtx);
  assert(ChipQueue->getLastEvent() != nullptr);
  this->takeOver(ChipQueue->getLastEvent());
  EventStatus_ = EVENT_STATUS_RECORDING;
}

CHIPEvent::CHIPEvent(CHIPContext *Ctx, CHIPEventFlags Flags)
    : EventStatus_(EVENT_STATUS_INIT), Flags_(Flags), ChipContext_(Ctx),
      Refc_(new size_t(1)) {}

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

CHIPKernel *CHIPModule::getKernel(std::string Name) {
  auto KernelFound = std::find_if(ChipKernels_.begin(), ChipKernels_.end(),
                                  [Name](CHIPKernel *Kernel) {
                                    return Kernel->getName().compare(Name) == 0;
                                  });
  if (KernelFound == ChipKernels_.end()) {
    std::string Msg = "Failed to find kernel via kernel name: " + Name;
    CHIPERR_LOG_AND_THROW(Msg, hipErrorLaunchFailure);
  }

  return *KernelFound;
}

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
    std::string Msg =
        "Failed to find global variable by name: " + std::string(VarName);
    CHIPERR_LOG_AND_THROW(Msg, hipErrorLaunchFailure);
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
  CHIPVarInfo *VarInfoBufD =
      (CHIPVarInfo *)Ctx->allocate(VarInfoBufSize, CHIPMemoryType::Shared);
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
    Var->setDevAddr(Ctx->allocate(Size, Alignment, CHIPMemoryType::Shared));
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
  allocateDeviceVariablesNoLock(Device, Queue);

  // Mark initialized if the module does not have any device variables.
  DeviceVariablesInitialized_ |= ChipVars_.empty();

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
    Device->getContext()->free(Var->getDevAddr());
    Var->setDevAddr(nullptr);
  }
  DeviceVariablesAllocated_ = false;
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
    : GridDim_(GridDim), BlockDim_(BlockDim), SharedMem_(SharedMem),
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
CHIPDevice::CHIPDevice(CHIPContext *Ctx) : Ctx_(Ctx) {}

CHIPDevice::CHIPDevice() {
  logDebug("Device {} is {}: name \"{}\" \n", Idx_, (void *)this,
           HipDeviceProps_.name);
}
CHIPDevice::~CHIPDevice() {}

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

std::string CHIPDevice::getName() {
  populateDeviceProperties();
  return std::string(HipDeviceProps_.name);
}

void CHIPDevice::populateDeviceProperties() {
  std::call_once(PropsPopulated_, &CHIPDevice::populateDevicePropertiesImpl,
                 this);
  if (!AllocationTracker)
    AllocationTracker = new CHIPAllocationTracker(
        HipDeviceProps_.totalGlobalMem, HipDeviceProps_.name);
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
  int *Pi;
  hipDeviceProp_t Prop = {0};
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
  case hipDeviceAttributeMaxTexture2DWidth:
    return Prop.maxTexture2D[0];
    break;
  case hipDeviceAttributeMaxTexture2DHeight:
    return Prop.maxTexture2D[1];
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
    *reinterpret_cast<unsigned int **>(Pi) = Prop.hdpMemFlushCntl;
    break;
  case hipDeviceAttributeHdpRegFlushCntl:
    *reinterpret_cast<unsigned int **>(Pi) = Prop.hdpRegFlushCntl;
    break;
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
  auto *Var = new CHIPDeviceVar(Name, Size);
  auto ModuleIter = ChipModules.find(ModuleStr);
  assert(ModuleIter != ChipModules.end() && "Module was not registered!");
  auto *Module = ModuleIter->second;
  Module->addDeviceVariable(Var);
  DeviceVarLookup_.insert(std::make_pair(HostPtr, Var));
}

void CHIPDevice::addQueue(CHIPQueue *ChipQueue) {
  auto QueueFound =
      std::find(ChipQueues_.begin(), ChipQueues_.end(), ChipQueue);
  if (QueueFound == ChipQueues_.end())
    ChipQueues_.push_back(ChipQueue);
  return;
}

CHIPQueue *CHIPDevice::addQueue(unsigned int Flags, int Priority) {
  auto ChipQueue = addQueueImpl(Flags, Priority);
  return ChipQueue;
}

std::vector<CHIPQueue *> CHIPDevice::getQueues() { return ChipQueues_; }

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
  auto FoundQueue =
      std::find(ChipQueues_.begin(), ChipQueues_.end(), ChipQueue);
  if (FoundQueue == ChipQueues_.end()) {
    std::string Msg =
        "Tried to remove a queue for a device but the queue was not found in "
        "device queue list";
    CHIPERR_LOG_AND_THROW(Msg, hipErrorUnknown);
  }

  ChipQueues_.erase(FoundQueue);
  return true;
}

void CHIPDevice::setSharedMemConfig(hipSharedMemConfig Cfg) { UNIMPLEMENTED(); }

size_t CHIPDevice::getUsedGlobalMem() {
  return AllocationTracker->TotalMemSize;
}

bool CHIPDevice::hasPCIBusId(int PciDomainID, int PciBusID, int PciDeviceID) {
  populateDeviceProperties();
  auto T1 = this->HipDeviceProps_.pciBusID == PciBusID;
  auto T2 = this->HipDeviceProps_.pciDomainID == PciDomainID;
  auto T3 = this->HipDeviceProps_.pciDeviceID == PciDeviceID;

  return (T1 && T2 && T3);
}

CHIPQueue *CHIPDevice::getActiveQueue() { return ChipQueues_[0]; }

hipError_t CHIPDevice::allocateDeviceVariables() {
  std::lock_guard<std::mutex> Lock(Mtx_);
  logTrace("Allocate storage for device variables.");
  for (auto I : ChipModules) {
    auto Status =
        I.second->allocateDeviceVariablesNoLock(this, getActiveQueue());
    if (Status != hipSuccess)
      return Status;
  }
  return hipSuccess;
}

void CHIPDevice::initializeDeviceVariables() {
  std::lock_guard<std::mutex> Lock(Mtx_);
  logTrace("Initialize device variables.");
  for (auto Module : ChipModules)
    Module.second->initializeDeviceVariablesNoLock(this, getActiveQueue());
}

void CHIPDevice::invalidateDeviceVariables() {
  std::lock_guard<std::mutex> Lock(Mtx_);
  logTrace("invalidate device variables.");
  for (auto Module : ChipModules)
    Module.second->invalidateDeviceVariablesNoLock();
}

void CHIPDevice::deallocateDeviceVariables() {
  std::lock_guard<std::mutex> Lock(Mtx_);
  logTrace("Deallocate storage for device variables.");
  for (auto Module : ChipModules)
    Module.second->deallocateDeviceVariablesNoLock(this);
}

// CHIPContext
//*************************************************************************************
CHIPContext::CHIPContext() {}
CHIPContext::~CHIPContext() {}

void CHIPContext::syncQueues(CHIPQueue *TargetQueue) {
  logDebug("CHIPContext::syncQueues()");
  std::vector<CHIPQueue *> Queues = Backend->getQueues();
  std::vector<CHIPQueue *> QueuesBlocking;

  // // Default queue gets created add init - always 0th in queue list
  CHIPQueue *DefaultQueue = Queues[0];
  Queues.erase(Queues.begin());

  for (auto &Queue : Queues)
    if (Queue->getQueueType() == CHIPQueueType::Blocking)
      QueuesBlocking.push_back(Queue);
  logDebug("Num blocking queues: {}", QueuesBlocking.size());

  // default stream waits on all blocking streams to complete
  std::vector<CHIPEvent *> EventsToWaitOn;

  if (TargetQueue == DefaultQueue) {
    for (auto &q : QueuesBlocking)
      EventsToWaitOn.push_back(q->getLastEvent());
    auto E = DefaultQueue->enqueueBarrierImpl(&EventsToWaitOn);
    TargetQueue->setLastEvent(E);
  } else { // blocking stream must wait until default stream is done
    EventsToWaitOn.push_back(DefaultQueue->getLastEvent());
    auto E = TargetQueue->enqueueBarrierImpl(&EventsToWaitOn);
    TargetQueue->setLastEvent(E);
  }
}

void CHIPContext::addDevice(CHIPDevice *ChipDevice) {
  logDebug("CHIPContext.add_device() {}", ChipDevice->getName());
  ChipDevices_.push_back(ChipDevice);
}

std::vector<CHIPDevice *> &CHIPContext::getDevices() {
  if (ChipDevices_.size() == 0)
    logWarn("CHIPContext.get_devices() was called but chip_devices is empty");
  return ChipDevices_;
}

std::vector<CHIPQueue *> &CHIPContext::getQueues() {
  if (ChipQueues_.size() == 0) {
    std::string Msg = "No queus in this context";
    CHIPERR_LOG_AND_THROW(Msg, hipErrorUnknown);
  }
  return ChipQueues_;
}
void CHIPContext::addQueue(CHIPQueue *ChipQueue) {
  logDebug("CHIPContext.add_queue()");
  ChipQueues_.push_back(ChipQueue);
}
hipStream_t CHIPContext::findQueue(hipStream_t Stream) {
  std::vector<CHIPQueue *> Queues = getQueues();
  if (Stream == nullptr)
    return Backend->getActiveQueue();

  auto QueueFound = std::find(Queues.begin(), Queues.end(), Stream);
  if (QueueFound == Queues.end())
    return nullptr;
  return *QueueFound;
}

void CHIPContext::finishAll() {
  for (CHIPQueue *Queue : ChipQueues_)
    Queue->finish();
}

void *CHIPContext::allocate(size_t Size) {
  return allocate(Size, 0, CHIPMemoryType::Shared);
}

void *CHIPContext::allocate(size_t Size, CHIPMemoryType MemType) {
  return allocate(Size, 0, MemType);
}
void *CHIPContext::allocate(size_t Size, size_t Alignment,
                            CHIPMemoryType MemType) {
  std::lock_guard<std::mutex> Lock(Mtx);
  void *AllocatedPtr;

  CHIPDevice *ChipDev = Backend->getActiveDevice();
  assert(ChipDev->getContext() == this);

  assert(ChipDev->AllocationTracker && "AllocationTracker was not created!");
  if (!ChipDev->AllocationTracker->reserveMem(Size))
    return nullptr;
  AllocatedPtr = allocateImpl(Size, Alignment, MemType);
  if (AllocatedPtr == nullptr)
    ChipDev->AllocationTracker->releaseMemReservation(Size);

  ChipDev->AllocationTracker->recordAllocation(AllocatedPtr, Size);

  return AllocatedPtr;
}

hipError_t CHIPContext::findPointerInfo(hipDeviceptr_t *Base, size_t *Size,
                                        hipDeviceptr_t Ptr) {
  // allocation_info *info = Backend->AllocationTracker.getByDevPtr(dptr);
  AllocationInfo *Info =
      Backend->getActiveDevice()->AllocationTracker->getByDevPtr(Ptr);
  if (!Info)
    return hipErrorInvalidDevicePointer;
  *Base = Info->BasePtr;
  *Size = Info->Size;
  return hipSuccess;
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

  // TODO Is all the state reset?
}

CHIPContext *CHIPContext::retain() { UNIMPLEMENTED(nullptr); }

hipError_t CHIPContext::free(void *Ptr) {
  CHIPDevice *ChipDev = Backend->getActiveDevice();
  AllocationInfo *Info = ChipDev->AllocationTracker->getByDevPtr(Ptr);
  if (!Info)
    return hipErrorInvalidDevicePointer;

  ChipDev->AllocationTracker->releaseMemReservation(Info->Size);
  freeImpl(Ptr);
  return hipSuccess;
}

// CHIPBackend
//*************************************************************************************

void CHIPBackend::uninitialize() {
  logDebug("CHIPBackend::uninitialize()");
  {
    std::lock_guard<std::mutex> Lock(Backend->EventsMtx);
    for (auto q : Backend->getQueues()) {
      auto Ev = q->getLastEvent();
      std::lock_guard<std::mutex> Lock(Ev->Mtx);
      if (Ev->getCHIPRefc() == 2)
        Ev->decreaseRefCount();
    }
  }
  // Give a chance for StaleEventMonitor to cleanup
  pthread_yield();
  sleep(1); // TODO Is there a better way to do this?

  logDebug("Remaining {} events that haven't been collected:",
           Backend->Events.size());
  for (auto E : Backend->Events)
    logDebug("{} status= {} refc={}", E->Msg, E->getEventStatusStr(),
             E->getCHIPRefc());
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

CHIPBackend::CHIPBackend() { logDebug("CHIPBackend Base Constructor"); };
CHIPBackend::~CHIPBackend() {
  logDebug("CHIPBackend Destructor. Deleting all pointers.");
  while (!ChipExecStack.empty())
    ChipExecStack.pop();
  while (!CallbackStack.empty())
    CallbackStack.pop();

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
  setActiveDevice(ChipDevices[0]);
}

void CHIPBackend::setActiveDevice(CHIPDevice *ChipDevice) {
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
  ActiveQ_ = ChipDevice->getActiveQueue();
}
std::vector<CHIPQueue *> &CHIPBackend::getQueues() { return ChipQueues; }
CHIPQueue *CHIPBackend::getActiveQueue() {
  if (ActiveQ_ == nullptr) {
    std::string Msg = "Active queue is null";
    CHIPERR_LOG_AND_THROW(Msg, hipErrorUnknown);
  }
  return ActiveQ_;
};

CHIPContext *CHIPBackend::getActiveContext() {
  if (ActiveCtx_ == nullptr) {
    std::string Msg = "Active context is null";
    CHIPERR_LOG_AND_THROW(Msg, hipErrorUnknown);
  }
  return ActiveCtx_;
};

CHIPDevice *CHIPBackend::getActiveDevice() {
  if (ActiveDev_ == nullptr) {
    CHIPERR_LOG_AND_THROW(
        "CHIPBackend.getActiveDevice() was called but active_ctx is null",
        hipErrorUnknown);
  }
  return ActiveDev_;
};

std::vector<CHIPDevice *> &CHIPBackend::getDevices() { return ChipDevices; }

size_t CHIPBackend::getNumDevices() { return ChipDevices.size(); }
std::vector<std::string *> &CHIPBackend::getModulesStr() { return ModulesStr_; }

void CHIPBackend::addContext(CHIPContext *ChipContext) {
  ChipContexts.push_back(ChipContext);
}
void CHIPBackend::addQueue(CHIPQueue *ChipQueue) {
  logDebug("CHIPBackend.add_queue()");
  ChipQueues.push_back(ChipQueue);
}
void CHIPBackend::addDevice(CHIPDevice *ChipDevice) {
  logDebug("CHIPDevice.add_device() {}", ChipDevice->getName());
  ChipDevices.push_back(ChipDevice);
}

void CHIPBackend::registerModuleStr(std::string *ModuleStr) {
  logDebug("CHIPBackend->register_module()");
  std::lock_guard<std::mutex> Lock(Mtx_);
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
  std::lock_guard<std::mutex> Lock(Mtx_);
  logDebug("CHIPBackend->configureCall(grid=({},{},{}), block=({},{},{}), "
           "shared={}, q={}",
           Grid.x, Grid.y, Grid.z, Block.x, Block.y, Block.z, SharedMem,
           (void *)ChipQueue);
  if (ChipQueue == nullptr)
    ChipQueue = getActiveQueue();
  CHIPExecItem *ExecItem = new CHIPExecItem(Grid, Block, SharedMem, ChipQueue);
  ChipExecStack.push(ExecItem);

  return hipSuccess;
}

hipError_t CHIPBackend::setArg(const void *Arg, size_t Size, size_t Offset) {
  logDebug("CHIPBackend->set_arg()");
  std::lock_guard<std::mutex> Lock(Mtx_);
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
    hipDeviceProp_t CurrentProp = {0};
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
  if (ChipQueue == nullptr) {
    logDebug("CHIPBackend::findQueue() was given a nullptr. Returning default "
             "queue");
    return Backend->getActiveQueue();
  }
  auto Queues = Backend->getActiveDevice()->getQueues();
  auto QueueFound = std::find(Queues.begin(), Queues.end(), ChipQueue);
  if (QueueFound == Queues.end())
    CHIPERR_LOG_AND_THROW(
        "CHIPBackend::findQueue() was given a non-nullptr queue but this queue "
        "was not found among the backend queues.",
        hipErrorTbd);
  return *QueueFound;
}

// hipError_t CHIPBackend::removeModule(CHIPModule *chip_module){};
// CHIPModule *CHIPBackend::addModule(std::string *module_str) {
//   for (auto &ctx : chip_contexts)
//     for (auto &dev : ctx->getDevices()) dev->addModule(modules_str);
//}
// CHIPQueue
//*************************************************************************************
CHIPQueue::CHIPQueue(CHIPDevice *ChipDevice, unsigned int Flags, int Priority)
    : ChipDevice_(ChipDevice), Flags_(Flags), Priority_(Priority) {
  ChipContext_ = ChipDevice->getContext();
  QueueType_ = CHIPQueueType{Flags};
};
CHIPQueue::CHIPQueue(CHIPDevice *ChipDevice, unsigned int Flags)
    : CHIPQueue(ChipDevice, Flags, 0){};
CHIPQueue::CHIPQueue(CHIPDevice *ChipDevice) : CHIPQueue(ChipDevice, 0, 0){};
CHIPQueue::~CHIPQueue(){};

///////// Enqueue Operations //////////
CHIPEvent *CHIPQueue::memCopyImpl(void *Dst, const void *Src, size_t Size) {
  auto ChipEvent = memCopyAsyncImpl(Dst, Src, Size);
  finish();
  return ChipEvent;
}
hipError_t CHIPQueue::memCopy(void *Dst, const void *Src, size_t Size) {
  std::lock_guard<std::mutex> Lock(Mtx);
  std::lock_guard<std::mutex> LockEvents(Backend->EventsMtx);
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif
  auto ChipEvent = memCopyImpl(Dst, Src, Size);
  ChipEvent->Msg = "memCopy";
  updateLastEvent(ChipEvent);
  this->finish();
  return hipSuccess;
}
hipError_t CHIPQueue::memCopyAsync(void *Dst, const void *Src, size_t Size) {
  std::lock_guard<std::mutex> LockEvents(Backend->EventsMtx);
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif
  auto ChipEvent = memCopyAsyncImpl(Dst, Src, Size);
  ChipEvent->Msg = "memCopyAsync";
  updateLastEvent(ChipEvent);
  return hipSuccess;
}
void CHIPQueue::memFill(void *Dst, size_t Size, const void *Pattern,
                        size_t PatternSize) {
  std::lock_guard<std::mutex> Lock(Mtx);
  std::lock_guard<std::mutex> LockEvents(Backend->EventsMtx);
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif
  auto ChipEvent = memFillImpl(Dst, Size, Pattern, PatternSize);
  ChipEvent->Msg = "memFill";
  updateLastEvent(ChipEvent);
}
CHIPEvent *CHIPQueue::memFillImpl(void *Dst, size_t Size, const void *Pattern,
                                  size_t PatternSize) {
  auto ChipEvent = memFillAsyncImpl(Dst, Size, Pattern, PatternSize);
  finish();
  return ChipEvent;
}
void CHIPQueue::memFillAsync(void *Dst, size_t Size, const void *Pattern,
                             size_t PatternSize) {
  std::lock_guard<std::mutex> Lock(Mtx);
  std::lock_guard<std::mutex> LockEvents(Backend->EventsMtx);
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif
  auto ChipEvent = memFillAsyncImpl(Dst, Size, Pattern, PatternSize);
  ChipEvent->Msg = "memFillAsync";
  updateLastEvent(ChipEvent);
}
void CHIPQueue::memCopy2D(void *Dst, size_t DPitch, const void *Src,
                          size_t SPitch, size_t Width, size_t Height) {
  std::lock_guard<std::mutex> Lock(Mtx);
  std::lock_guard<std::mutex> LockEvents(Backend->EventsMtx);
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif
  auto ChipEvent = memCopy2DAsyncImpl(Dst, DPitch, Src, SPitch, Width, Height);
  ChipEvent->Msg = "memCopy2D";
  finish();
  updateLastEvent(ChipEvent);
}
CHIPEvent *CHIPQueue::memCopy2DImpl(void *Dst, size_t DPitch, const void *Src,
                                    size_t SPitch, size_t Width,
                                    size_t Height) {

  auto ChipEvent = memCopy2DAsyncImpl(Dst, DPitch, Src, SPitch, Width, Height);
  finish();
  return ChipEvent;
}
void CHIPQueue::memCopy2DAsync(void *Dst, size_t DPitch, const void *Src,
                               size_t SPitch, size_t Width, size_t Height) {
  std::lock_guard<std::mutex> Lock(Mtx);
  std::lock_guard<std::mutex> LockEvents(Backend->EventsMtx);
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif
  auto ChipEvent = memCopy2DAsyncImpl(Dst, DPitch, Src, SPitch, Width, Height);
  ChipEvent->Msg = "memCopy2DAsync";
  updateLastEvent(ChipEvent);
}
void CHIPQueue::memCopy3D(void *Dst, size_t DPitch, size_t DSPitch,
                          const void *Src, size_t SPitch, size_t SSPitch,
                          size_t Width, size_t Height, size_t Depth) {
  std::lock_guard<std::mutex> Lock(Mtx);
  std::lock_guard<std::mutex> LockEvents(Backend->EventsMtx);
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif
  auto ChipEvent = memCopy3DAsyncImpl(Dst, DPitch, DSPitch, Src, SPitch,
                                      SSPitch, Width, Height, Depth);
  ChipEvent->Msg = "memCopy3D";
  finish();
  updateLastEvent(ChipEvent);
}
CHIPEvent *CHIPQueue::memCopy3DImpl(void *Dst, size_t DPitch, size_t DSPitch,
                                    const void *Src, size_t SPithc,
                                    size_t SSPitch, size_t Width, size_t Height,
                                    size_t Depth) {
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif
  auto ChipEvent = memCopy3DAsyncImpl(Dst, DPitch, DSPitch, Src, SPithc,
                                      SSPitch, Width, Height, Depth);
  finish();
  return ChipEvent;
}
void CHIPQueue::memCopy3DAsync(void *Dst, size_t DPitch, size_t DSPitch,
                               const void *Src, size_t SPitch, size_t SSPitch,
                               size_t Width, size_t Height, size_t Depth) {
  std::lock_guard<std::mutex> Lock(Mtx);
  std::lock_guard<std::mutex> LockEvents(Backend->EventsMtx);
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif
  auto ChipEvent = memCopy3DAsyncImpl(Dst, DPitch, DSPitch, Src, SPitch,
                                      SSPitch, Width, Height, Depth);
  ChipEvent->Msg = "memCopy3DAsync";
  updateLastEvent(ChipEvent);
}
void CHIPQueue::memCopyToTexture(CHIPTexture *TexObj, void *Src) {
  std::lock_guard<std::mutex> Lock(Mtx);
  std::lock_guard<std::mutex> LockEvents(Backend->EventsMtx);
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif
  auto ChipEvent = memCopyToTextureImpl(TexObj, Src);
  ChipEvent->Msg = "memCopyToTexture";
  updateLastEvent(ChipEvent);
}
void CHIPQueue::launch(CHIPExecItem *ExecItem) {
  std::lock_guard<std::mutex> Lock(Mtx);
  std::lock_guard<std::mutex> LockEvents(Backend->EventsMtx);
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif
  auto ChipEvent = launchImpl(ExecItem);
  ChipEvent->Msg = "launch";
  updateLastEvent(ChipEvent);

  size_t NumArgs = ExecItem->getKernel()->getFuncInfo()->ArgTypeInfo.size();

  auto AllocTracker = Backend->getActiveDevice()->AllocationTracker;
  auto Args = ExecItem->getArgsPointer();
  for (int i = 0; i < ExecItem->getNumArgs(); i++) {

    void **k = reinterpret_cast<void **>(Args[i]);
    void *DevPtr = reinterpret_cast<void *>(*k);
    void *HostPtr = AllocTracker->getAssociatedHostPtr(DevPtr);

    if (HostPtr) {
      auto AllocInfo = AllocTracker->getByDevPtr(DevPtr);
      logDebug("A hipHostRegister argument was found. Appending a mem copy "
               "back to the host {} -> {}",
               DevPtr, HostPtr);
      auto Ev = this->memCopyImpl(HostPtr, DevPtr, AllocInfo->Size);
      Ev->Msg = "hipHostRegisterMemCpy";
      updateLastEvent(Ev);
    }
  }
}
CHIPEvent *
CHIPQueue::enqueueBarrier(std::vector<CHIPEvent *> *EventsToWaitFor) {
  std::lock_guard<std::mutex> Lock(Mtx);
  auto ChipEvent = enqueueBarrierImpl(EventsToWaitFor);
  ChipEvent->Msg = "enqueueBarrier";
  updateLastEvent(ChipEvent);
  return ChipEvent;
}
CHIPEvent *CHIPQueue::enqueueMarker() {
  std::lock_guard<std::mutex> Lock(Mtx);
  auto ChipEvent = enqueueMarkerImpl();
  ChipEvent->Msg = "enqueueMarker";
  updateLastEvent(ChipEvent);
  return ChipEvent;
}

void CHIPQueue::memPrefetch(const void *Ptr, size_t Count) {
  std::lock_guard<std::mutex> Lock(Mtx);
  std::lock_guard<std::mutex> LockEvents(Backend->EventsMtx);
#ifdef ENFORCE_QUEUE_SYNC
  ChipContext_->syncQueues(this);
#endif
  auto ChipEvent = memPrefetchImpl(Ptr, Count);
  ChipEvent->Msg = "memPrefetch";
  updateLastEvent(ChipEvent);
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

unsigned int CHIPQueue::getFlags() { return Flags_; }
int CHIPQueue::getPriorityRange(int LowerOrUpper) { UNIMPLEMENTED(0); }
int CHIPQueue::getPriority() { UNIMPLEMENTED(0); }
bool CHIPQueue::addCallback(hipStreamCallback_t Callback, void *UserData) {
  CHIPCallbackData *Callbackdata =
      Backend->createCallbackData(Callback, UserData, this);

  {
    std::lock_guard<std::mutex> Lock(Backend->CallbackStackMtx);
    Backend->CallbackStack.push(Callbackdata);
  }

  // Setup event handling on the CPU side
  {
    std::lock_guard<std::mutex> Lock(Mtx);
    if (!EventMonitor_)
      EventMonitor_ = Backend->createCallbackEventMonitor();
  }

  return true;
}

bool CHIPQueue::query() { UNIMPLEMENTED(true); }
