#include "CHIPBackend.hh"

#include <utility>

/// Queue a kernel for retrieving information about the device variable.
static void queueKernel(CHIPQueue* Q, CHIPKernel* K, void* Args[] = nullptr,
                        dim3 GridDim = dim3(1), dim3 BlockDim = dim3(1),
                        size_t SharedMemSize = 0) {
  assert(Q);
  assert(K);
  // FIXME: Should construct backend specific exec item or make the exec
  //        item a backend agnostic class.
  CHIPExecItem EI(GridDim, BlockDim, SharedMemSize, Q);
  EI.setArgPointer(Args);
  EI.launchByDevicePtr(K);
}

/// Queue a shadow kernel for binding a device variable (a pointer) to
/// the give n allocation.
static void queueVariableInfoShadowKernel(CHIPQueue* Q, CHIPModule* M,
                                          const CHIPDeviceVar* Var,
                                          void* InfoBuffer) {
  assert(M && Var && InfoBuffer);
  auto* K = M->getKernel(std::string(ChipVarInfoPrefix) + Var->getName());
  assert(K && "Module is missing a shadow kernel?");
  void* Args[] = {&InfoBuffer};
  queueKernel(Q, K, Args);
}

static void queueVariableBindShadowKernel(CHIPQueue* Q, CHIPModule* M,
                                          const CHIPDeviceVar* Var) {
  assert(M && Var);
  auto* DevPtr = Var->getDevAddr();
  assert(DevPtr && "Space has not be allocated for a variable.");
  auto* K = M->getKernel(std::string(ChipVarBindPrefix) + Var->getName());
  assert(K && "Module is missing a shadow kernel?");
  void* Args[] = {&DevPtr};
  queueKernel(Q, K, Args);
}

static void queueVariableInitShadowKernel(CHIPQueue* Q, CHIPModule* M,
                                          const CHIPDeviceVar* Var) {
  assert(M && Var);
  auto* K = M->getKernel(std::string(ChipVarInitPrefix) + Var->getName());
  assert(K && "Module is missing a shadow kernel?");
  queueKernel(Q, K);
}

CHIPCallbackData::CHIPCallbackData(hipStreamCallback_t callback_f_,
                                   void* callback_args_, CHIPQueue* chip_queue_)
    : CallbackF(callback_f_), CallbackArgs(callback_args_),
      ChipQueue(chip_queue_) {
  setup();
}

void CHIPCallbackData::setup() {
  CHIPContext* ctx = ChipQueue->getContext();
  GpuReady = Backend->createCHIPEvent(ctx);
  CpuCallbackComplete = Backend->createCHIPEvent(ctx);
  GpuAck = Backend->createCHIPEvent(ctx);

  auto gpu_ready = ChipQueue->enqueueBarrier(nullptr);

  std::vector<CHIPEvent*> evs = {CpuCallbackComplete};
  ChipQueue->enqueueBarrier(&evs);

  CHIPEvent* gpu_ack = ChipQueue->enqueueMarker();
}

void CHIPEventMonitor::monitor() {
  logDebug("CHIPEventMonitor::monitor()");
  CHIPCallbackData* cb;
  while (Backend->getCallback(&cb)) {
    cb->GpuReady->wait();
    cb->execute(hipSuccess);
    cb->CpuCallbackComplete->hostSignal();
    cb->GpuAck->wait();
    delete cb;
  }

  // no more callback events left, free up the thread
  delete this;
  pthread_yield();
}

// CHIPDeviceVar
// ************************************************************************
CHIPDeviceVar::CHIPDeviceVar(std::string TheName, size_t TheSize)
    : Name_(TheName), Size_(TheSize) {}

CHIPDeviceVar::~CHIPDeviceVar() { assert(!DevAddr_ && "Memory leak?"); }

// CHIPAllocationTracker
// ************************************************************************
CHIPAllocationTracker::CHIPAllocationTracker(size_t global_mem_size_,
                                             std::string name_)
    : GlobalMemSize(global_mem_size_), TotalMemSize(0), MaxMemUsed(0) {
  Name_ = name_;
}
CHIPAllocationTracker::~CHIPAllocationTracker() {
  // TODO free up all the pointers in ptr_set
  UNIMPLEMENTED();
}

AllocationInfo* CHIPAllocationTracker::getByHostPtr(const void* host_ptr) {
  auto found = HostToDev_.find(const_cast<void*>(host_ptr));
  if (found == HostToDev_.end()) {
    CHIPERR_LOG_AND_THROW("Unable to find allocation info for host pointer",
                          hipErrorInvalidSymbol);
  }
  return getByDevPtr(found->second);
}
AllocationInfo* CHIPAllocationTracker::getByDevPtr(const void* dev_ptr) {
  auto ptr = const_cast<void*>(dev_ptr);
  logDebug("dev_to_allocation_info size: {}", DevToAllocInfo_.size());
  auto c = DevToAllocInfo_.count(ptr);
  if (c == 0)
    CHIPERR_LOG_AND_THROW("pointer not found on device", hipErrorTbd);

  return &DevToAllocInfo_[const_cast<void*>(dev_ptr)];
}

bool CHIPAllocationTracker::reserveMem(size_t bytes) {
  std::lock_guard<std::mutex> Lock(Mtx_);
  if (bytes <= (GlobalMemSize - TotalMemSize)) {
    TotalMemSize += bytes;
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

bool CHIPAllocationTracker::releaseMemReservation(unsigned long bytes) {
  std::lock_guard<std::mutex> Lock(Mtx_);
  if (TotalMemSize >= bytes) {
    TotalMemSize -= bytes;
    return true;
  }

  return false;
}

void CHIPAllocationTracker::recordAllocation(void* dev_ptr, size_t size_) {
  AllocationInfo alloc_info{dev_ptr, size_};
  DevToAllocInfo_[dev_ptr] = alloc_info;
  logDebug("CHIPAllocationTracker::recordAllocation size: {}",
           DevToAllocInfo_.size());
  return;
}

// CHIPEvent
// ************************************************************************

void CHIPEvent::recordStream(CHIPQueue* chip_queue) {
  logDebug("CHIPEvent::recordStream()");
  std::lock_guard<std::mutex> Lock(Mtx_);
  assert(chip_queue->getLastEvent() != nullptr);
  this->takeOver(chip_queue->getLastEvent());
  EventStatus_ = EVENT_STATUS_RECORDING;
}

CHIPEvent::CHIPEvent(CHIPContext* ctx_in, CHIPEventFlags flags_)
    : EventStatus_(EVENT_STATUS_INIT), Flags_(flags_), ChipContext_(ctx_in),
      Refc_(new size_t(1)) {
  ctx_in->Events.push_back(this);
}

// CHIPModuleflags_
//*************************************************************************************
void CHIPModule::consumeSPIRV() {
  FuncIL_ = (uint8_t*)Src_.data();
  IlSize_ = Src_.length();

  // Parse the SPIR-V fat binary to retrieve kernel function
  size_t numWords = IlSize_ / 4;
  BinaryData_ = new int32_t[numWords + 1];
  std::memcpy(BinaryData_, FuncIL_, IlSize_);
  // Extract kernel function information
  bool res = parseSPIR(BinaryData_, numWords, FuncInfos_);
  delete[] BinaryData_;
  if (!res) {
    CHIPERR_LOG_AND_THROW("SPIR-V parsing failed", hipErrorUnknown);
  }
}

CHIPModule::CHIPModule(std::string* module_str) {
  Src_ = *module_str;
  consumeSPIRV();
}
CHIPModule::CHIPModule(std::string&& module_str) {
  Src_ = module_str;
  consumeSPIRV();
}
CHIPModule::~CHIPModule() {}

void CHIPModule::addKernel(CHIPKernel* kernel) {
  ChipKernels_.push_back(kernel);
}

void CHIPModule::compileOnce(CHIPDevice* chip_dev) {
  std::call_once(Compiled_, &CHIPModule::compile, this, chip_dev);
}

CHIPKernel* CHIPModule::getKernel(std::string name) {
  auto kernel = std::find_if(
      ChipKernels_.begin(), ChipKernels_.end(),
      [name](CHIPKernel* k) { return k->getName().compare(name) == 0; });
  if (kernel == ChipKernels_.end()) {
    std::string msg = "Failed to find kernel via kernel name: " + name;
    CHIPERR_LOG_AND_THROW(msg, hipErrorLaunchFailure);
  }

  return *kernel;
}

CHIPKernel* CHIPModule::getKernel(const void* host_f_ptr) {
  for (auto& kernel : ChipKernels_)
    logDebug("chip kernel: {} {}", kernel->getHostPtr(), kernel->getName());
  auto kernel = std::find_if(
      ChipKernels_.begin(), ChipKernels_.end(),
      [host_f_ptr](CHIPKernel* k) { return k->getHostPtr() == host_f_ptr; });
  if (kernel == ChipKernels_.end()) {
    std::string msg = "Failed to find kernel via host pointer";
    CHIPERR_LOG_AND_THROW(msg, hipErrorLaunchFailure);
  }

  return *kernel;
}

std::vector<CHIPKernel*>& CHIPModule::getKernels() { return ChipKernels_; }

CHIPDeviceVar* CHIPModule::getGlobalVar(const char* var_name_) {
  auto var = std::find_if(ChipVars_.begin(), ChipVars_.end(),
                          [var_name_](CHIPDeviceVar* v) {
                            return v->getName().compare(var_name_) == 0;
                          });
  if (var == ChipVars_.end()) {
    std::string msg =
        "Failed to find global variable by name: " + std::string(var_name_);
    CHIPERR_LOG_AND_THROW(msg, hipErrorLaunchFailure);
  }

  return *var;
}

hipError_t CHIPModule::allocateDeviceVariablesNoLock(CHIPDevice* Device,
                                                     CHIPQueue* Queue) {
  // Mark as allocated if the module does not have any variables.
  DeviceVariablesAllocated_ |= ChipVars_.empty();

  if (DeviceVariablesAllocated_)
    return hipSuccess;

  logTrace("Allocate storage for device variables in module: {}", (void*)this);

  // TODO: catch any exception and abort as it's probably an unrecoverable
  //       condition?

  size_t VarInfoBufSize = sizeof(CHIPVarInfo) * ChipVars_.size();
  auto* Ctx = Device->getContext();
  CHIPVarInfo* VarInfoBufD =
      (CHIPVarInfo*)Ctx->allocate(VarInfoBufSize, CHIPMemoryType::Shared);
  assert(VarInfoBufD && "Could not allocate space for a shadow kernel.");
  auto VarInfoBufH = std::make_unique<CHIPVarInfo[]>(ChipVars_.size());

  // Gather information for storage allocation.
  std::vector<std::pair<CHIPDeviceVar*, CHIPVarInfo*>> VarInfos;
  for (auto* Var : ChipVars_) {
    auto I = VarInfos.size();
    queueVariableInfoShadowKernel(Queue, this, Var, &VarInfoBufD[I]);
    VarInfos.push_back(std::make_pair(Var, &VarInfoBufH[I]));
  }
  Queue->memCopyAsync(VarInfoBufH.get(), VarInfoBufD, VarInfoBufSize);
  Queue->finish();

  // Allocate storage for the device variables.
  for (auto& VarInfo : VarInfos) {
    size_t Size = (*VarInfo.second)[0];
    size_t Alignment = (*VarInfo.second)[1];
    size_t HasInitializer = (*VarInfo.second)[2];
    assert(Size && "Unexpected zero sized device variable.");
    assert(Alignment && "Unexpected alignment requirement.");

    auto* Var = VarInfo.first;
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

void CHIPModule::initializeDeviceVariablesNoLock(CHIPDevice* Device,
                                                 CHIPQueue* Queue) {
  allocateDeviceVariablesNoLock(Device, Queue);

  // Mark initialized if the module does not have any device variables.
  DeviceVariablesInitialized_ |= ChipVars_.empty();

  if (DeviceVariablesInitialized_) {
    // Can't be initialized if no storage is not allocated.
    assert(DeviceVariablesAllocated_ && "Should have storage.");
    return;
  }

  logTrace("Initialize device variables in module: {}", (void*)this);

  bool QueuedKernels = false;
  for (auto* Var : ChipVars_) {
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

void CHIPModule::deallocateDeviceVariablesNoLock(CHIPDevice* Device) {
  invalidateDeviceVariablesNoLock();
  for (auto* Var : ChipVars_) {
    Device->getContext()->free(Var->getDevAddr());
    Var->setDevAddr(nullptr);
  }
  DeviceVariablesAllocated_ = false;
}

// CHIPKernel
//*************************************************************************************
CHIPKernel::CHIPKernel(std::string host_f_name_, OCLFuncInfo* func_info_)
    : HostFName_(host_f_name_), FuncInfo_(func_info_) {}
CHIPKernel::~CHIPKernel(){};
std::string CHIPKernel::getName() { return HostFName_; }
const void* CHIPKernel::getHostPtr() { return HostFPtr_; }
const void* CHIPKernel::getDevPtr() { return DevFPtr_; }

OCLFuncInfo* CHIPKernel::getFuncInfo() { return FuncInfo_; }

void CHIPKernel::setName(std::string host_f_name_) {
  HostFName_ = host_f_name_;
}
void CHIPKernel::setHostPtr(const void* host_f_ptr_) {
  HostFPtr_ = host_f_ptr_;
}
void CHIPKernel::setDevPtr(const void* dev_f_ptr_) { DevFPtr_ = dev_f_ptr_; }

// CHIPExecItem
//*************************************************************************************
CHIPExecItem::CHIPExecItem(dim3 grid_dim_, dim3 block_dim_, size_t shared_mem_,
                           hipStream_t chip_queue_)
    : GridDim_(grid_dim_), BlockDim_(block_dim_), SharedMem_(shared_mem_),
      ChipQueue_(chip_queue_){};
CHIPExecItem::~CHIPExecItem(){};

std::vector<uint8_t> CHIPExecItem::getArgData() { return ArgData_; }

void CHIPExecItem::setArg(const void* arg, size_t size, size_t offset) {
  if ((offset + size) > ArgData_.size())
    ArgData_.resize(offset + size + 1024);

  std::memcpy(ArgData_.data() + offset, arg, size);
  logDebug("CHIPExecItem.setArg() on {} size {} offset {}\n", (void*)this, size,
           offset);
  OffsetSizes_.push_back(std::make_tuple(offset, size));
}

CHIPEvent* CHIPExecItem::launchByDevicePtr(CHIPKernel* K) {
  this->ChipKernel_ = K;
  return ChipQueue_->launch(this);
}

CHIPEvent* CHIPExecItem::launchByHostPtr(const void* hostPtr) {
  logTrace("launchByHostPtr");
  if (ChipQueue_ == nullptr) {
    std::string msg = "Tried to launch CHIPExecItem but its queue is null";
    CHIPERR_LOG_AND_THROW(msg, hipErrorLaunchFailure);
  }

  CHIPDevice* dev = ChipQueue_->getDevice();
  return launchByDevicePtr(dev->findKernelByHostPtr(hostPtr));
}

dim3 CHIPExecItem::getBlock() { return BlockDim_; }
dim3 CHIPExecItem::getGrid() { return GridDim_; }
CHIPKernel* CHIPExecItem::getKernel() { return ChipKernel_; }
size_t CHIPExecItem::getSharedMem() { return SharedMem_; }
CHIPQueue* CHIPExecItem::getQueue() { return ChipQueue_; }
// CHIPDevice
//*************************************************************************************
CHIPDevice::CHIPDevice(CHIPContext* ctx_) : Ctx_(ctx_) {}

CHIPDevice::CHIPDevice() {
  logDebug("Device {} is {}: name \"{}\" \n", Idx, (void*)this,
           HipDeviceProps_.name);
}
CHIPDevice::~CHIPDevice() {}

std::vector<CHIPKernel*> CHIPDevice::getKernels() {
  std::vector<CHIPKernel*> kernels;
  for (auto module_it : ChipModules) {
    auto* module = module_it.second;
    for (CHIPKernel* kernel : module->getKernels())
      kernels.push_back(kernel);
  }
  return kernels;
}

std::unordered_map<const std::string*, CHIPModule*>& CHIPDevice::getModules() {
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
void CHIPDevice::copyDeviceProperties(hipDeviceProp_t* prop) {
  logDebug("CHIPDevice->copy_device_properties()");
  if (prop)
    std::memcpy(prop, &this->HipDeviceProps_, sizeof(hipDeviceProp_t));
}

CHIPKernel* CHIPDevice::findKernelByHostPtr(const void* hostPtr) {
  logDebug("CHIPDevice::findKernelByHostPtr({})", hostPtr);
  std::vector<CHIPKernel*> chip_kernels = getKernels();
  if (chip_kernels.size() == 0) {
    CHIPERR_LOG_AND_THROW("chip_kernels is empty for this device",
                          hipErrorLaunchFailure);
  }
  logDebug("Listing Kernels for device {}", getName());
  for (auto& kernel : chip_kernels) {
    logDebug("Kernel name: {} host_f_ptr: {}", kernel->getName(),
             kernel->getHostPtr());
  }

  auto found_kernel = std::find_if(chip_kernels.begin(), chip_kernels.end(),
                                   [&hostPtr](CHIPKernel* kernel) {
                                     return kernel->getHostPtr() == hostPtr;
                                   });

  if (found_kernel == chip_kernels.end()) {
    std::string msg =
        "Tried to find kernel by host pointer but kernel was not found";
    CHIPERR_LOG_AND_THROW(msg, hipErrorLaunchFailure);
  } else {
    logDebug("Found kernel {} with host pointer {}", (*found_kernel)->getName(),
             (*found_kernel)->getHostPtr());
  }

  return *found_kernel;
}

CHIPContext* CHIPDevice::getContext() { return Ctx_; }
int CHIPDevice::getDeviceId() { return Idx; }

CHIPDeviceVar* CHIPDevice::getStatGlobalVar(const void* HostPtr) {
  if (DeviceVarLookup_.count(HostPtr)) {
    auto* Var = DeviceVarLookup_[HostPtr];
    assert(Var->getDevAddr() && "Missing device pointer.");
    return Var;
  }
  return nullptr;
}

CHIPDeviceVar* CHIPDevice::getGlobalVar(const void* HostPtr) {
  if (auto* Found = getDynGlobalVar(HostPtr))
    return Found;

  if (auto* Found = getStatGlobalVar(HostPtr))
    return Found;

  return nullptr;
}

int CHIPDevice::getAttr(hipDeviceAttribute_t attr) {
  int* pi;
  hipDeviceProp_t prop = {0};
  copyDeviceProperties(&prop);

  switch (attr) {
  case hipDeviceAttributeMaxThreadsPerBlock:
    return prop.maxThreadsPerBlock;
    break;
  case hipDeviceAttributeMaxBlockDimX:
    return prop.maxThreadsDim[0];
    break;
  case hipDeviceAttributeMaxBlockDimY:
    return prop.maxThreadsDim[1];
    break;
  case hipDeviceAttributeMaxBlockDimZ:
    return prop.maxThreadsDim[2];
    break;
  case hipDeviceAttributeMaxGridDimX:
    return prop.maxGridSize[0];
    break;
  case hipDeviceAttributeMaxGridDimY:
    return prop.maxGridSize[1];
    break;
  case hipDeviceAttributeMaxGridDimZ:
    return prop.maxGridSize[2];
    break;
  case hipDeviceAttributeMaxSharedMemoryPerBlock:
    return prop.sharedMemPerBlock;
    break;
  case hipDeviceAttributeTotalConstantMemory:
    return prop.totalConstMem;
    break;
  case hipDeviceAttributeWarpSize:
    return prop.warpSize;
    break;
  case hipDeviceAttributeMaxRegistersPerBlock:
    return prop.regsPerBlock;
    break;
  case hipDeviceAttributeClockRate:
    return prop.clockRate;
    break;
  case hipDeviceAttributeMemoryClockRate:
    return prop.memoryClockRate;
    break;
  case hipDeviceAttributeMemoryBusWidth:
    return prop.memoryBusWidth;
    break;
  case hipDeviceAttributeMultiprocessorCount:
    return prop.multiProcessorCount;
    break;
  case hipDeviceAttributeComputeMode:
    return prop.computeMode;
    break;
  case hipDeviceAttributeL2CacheSize:
    return prop.l2CacheSize;
    break;
  case hipDeviceAttributeMaxThreadsPerMultiProcessor:
    return prop.maxThreadsPerMultiProcessor;
    break;
  case hipDeviceAttributeComputeCapabilityMajor:
    return prop.major;
    break;
  case hipDeviceAttributeComputeCapabilityMinor:
    return prop.minor;
    break;
  case hipDeviceAttributePciBusId:
    return prop.pciBusID;
    break;
  case hipDeviceAttributeConcurrentKernels:
    return prop.concurrentKernels;
    break;
  case hipDeviceAttributePciDeviceId:
    return prop.pciDeviceID;
    break;
  case hipDeviceAttributeMaxSharedMemoryPerMultiprocessor:
    return prop.maxSharedMemoryPerMultiProcessor;
    break;
  case hipDeviceAttributeIsMultiGpuBoard:
    return prop.isMultiGpuBoard;
    break;
  case hipDeviceAttributeCooperativeLaunch:
    return prop.cooperativeLaunch;
    break;
  case hipDeviceAttributeCooperativeMultiDeviceLaunch:
    return prop.cooperativeMultiDeviceLaunch;
    break;
  case hipDeviceAttributeIntegrated:
    return prop.integrated;
    break;
  case hipDeviceAttributeMaxTexture1DWidth:
    return prop.maxTexture1D;
    break;
  case hipDeviceAttributeMaxTexture2DWidth:
    return prop.maxTexture2D[0];
    break;
  case hipDeviceAttributeMaxTexture2DHeight:
    return prop.maxTexture2D[1];
    break;
  case hipDeviceAttributeMaxTexture3DWidth:
    return prop.maxTexture3D[0];
    break;
  case hipDeviceAttributeMaxTexture3DHeight:
    return prop.maxTexture3D[1];
    break;
  case hipDeviceAttributeMaxTexture3DDepth:
    return prop.maxTexture3D[2];
    break;
  case hipDeviceAttributeHdpMemFlushCntl:
    *reinterpret_cast<unsigned int**>(pi) = prop.hdpMemFlushCntl;
    break;
  case hipDeviceAttributeHdpRegFlushCntl:
    *reinterpret_cast<unsigned int**>(pi) = prop.hdpRegFlushCntl;
    break;
  case hipDeviceAttributeMaxPitch:
    return prop.memPitch;
    break;
  case hipDeviceAttributeTextureAlignment:
    return prop.textureAlignment;
    break;
  case hipDeviceAttributeTexturePitchAlignment:
    return prop.texturePitchAlignment;
    break;
  case hipDeviceAttributeKernelExecTimeout:
    return prop.kernelExecTimeoutEnabled;
    break;
  case hipDeviceAttributeCanMapHostMemory:
    return prop.canMapHostMemory;
    break;
  case hipDeviceAttributeEccEnabled:
    return prop.ECCEnabled;
    break;
  case hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc:
    return prop.cooperativeMultiDeviceUnmatchedFunc;
    break;
  case hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim:
    return prop.cooperativeMultiDeviceUnmatchedGridDim;
    break;
  case hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim:
    return prop.cooperativeMultiDeviceUnmatchedBlockDim;
    break;
  case hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem:
    return prop.cooperativeMultiDeviceUnmatchedSharedMem;
    break;
  case hipDeviceAttributeAsicRevision:
    return prop.asicRevision;
    break;
  case hipDeviceAttributeManagedMemory:
    return prop.managedMemory;
    break;
  case hipDeviceAttributeDirectManagedMemAccessFromHost:
    return prop.directManagedMemAccessFromHost;
    break;
  case hipDeviceAttributeConcurrentManagedAccess:
    return prop.concurrentManagedAccess;
    break;
  case hipDeviceAttributePageableMemoryAccess:
    return prop.pageableMemoryAccess;
    break;
  case hipDeviceAttributePageableMemoryAccessUsesHostPageTables:
    return prop.pageableMemoryAccessUsesHostPageTables;
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

void CHIPDevice::registerFunctionAsKernel(std::string* module_str,
                                          const void* host_f_ptr,
                                          const char* host_f_name) {
  CHIPModule* module = nullptr;
  if (ChipModules.count(module_str)) {
    module = ChipModules[module_str];
  } else {
    module = addModule(module_str);
    module->compileOnce(this);
  }

  CHIPKernel* kernel = module->getKernel(std::string(host_f_name));
  if (!kernel) {
    std::string msg = "Device " + getName() +
                      " tried to register host function " + host_f_name +
                      " but failed to find kernel with a matching name";
    CHIPERR_LOG_AND_THROW(msg, hipErrorLaunchFailure);
  }

  kernel->setHostPtr(host_f_ptr);

  logDebug("Device {}: successfully registered function {} as kernel {}",
           getName(), host_f_name, kernel->getName().c_str());
  return;
}

void CHIPDevice::registerDeviceVariable(std::string* ModuleStr,
                                        const void* HostPtr, const char* Name,
                                        size_t Size) {
  auto* Var = new CHIPDeviceVar(Name, Size);
  auto ModuleIter = ChipModules.find(ModuleStr);
  assert(ModuleIter != ChipModules.end() && "Module was not registered!");
  auto* Module = ModuleIter->second;
  Module->addDeviceVariable(Var);
  DeviceVarLookup_.insert(std::make_pair(HostPtr, Var));
}

void CHIPDevice::addQueue(CHIPQueue* chip_queue_) {
  auto queue_found =
      std::find(ChipQueues_.begin(), ChipQueues_.end(), chip_queue_);
  if (queue_found == ChipQueues_.end())
    ChipQueues_.push_back(chip_queue_);
  return;
}

CHIPQueue* CHIPDevice::addQueue(unsigned int flags, int priority) {
  auto q = addQueueImpl(flags, priority);
  return q;
}

std::vector<CHIPQueue*> CHIPDevice::getQueues() { return ChipQueues_; }

hipError_t CHIPDevice::setPeerAccess(CHIPDevice* peer, int flags,
                                     bool canAccessPeer) {
  UNIMPLEMENTED(hipSuccess);
}

int CHIPDevice::getPeerAccess(CHIPDevice* peerDevice) { UNIMPLEMENTED(0); }

void CHIPDevice::setCacheConfig(hipFuncCache_t cfg) { UNIMPLEMENTED(); }

void CHIPDevice::setFuncCacheConfig(const void* func, hipFuncCache_t config) {
  UNIMPLEMENTED();
}

hipFuncCache_t CHIPDevice::getCacheConfig() {
  UNIMPLEMENTED(hipFuncCachePreferNone);
}

hipSharedMemConfig CHIPDevice::getSharedMemConfig() {
  UNIMPLEMENTED(hipSharedMemBankSizeDefault);
}

bool CHIPDevice::removeQueue(CHIPQueue* q) {
  auto found_q = std::find(ChipQueues_.begin(), ChipQueues_.end(), q);
  if (found_q == ChipQueues_.end()) {
    std::string msg =
        "Tried to remove a queue for a device but the queue was not found in "
        "device queue list";
    CHIPERR_LOG_AND_THROW(msg, hipErrorUnknown);
  }

  ChipQueues_.erase(found_q);
  return true;
}

void CHIPDevice::setSharedMemConfig(hipSharedMemConfig config) {
  UNIMPLEMENTED();
}

size_t CHIPDevice::getUsedGlobalMem() {
  return AllocationTracker->TotalMemSize;
}

bool CHIPDevice::hasPCIBusId(int, int, int) { UNIMPLEMENTED(true); }

CHIPQueue* CHIPDevice::getActiveQueue() { return ChipQueues_[0]; }

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
  for (auto it : ChipModules)
    it.second->initializeDeviceVariablesNoLock(this, getActiveQueue());
}

void CHIPDevice::invalidateDeviceVariables() {
  std::lock_guard<std::mutex> Lock(Mtx_);
  logTrace("invalidate device variables.");
  for (auto it : ChipModules)
    it.second->invalidateDeviceVariablesNoLock();
}

void CHIPDevice::deallocateDeviceVariables() {
  std::lock_guard<std::mutex> Lock(Mtx_);
  logTrace("Deallocate storage for device variables.");
  for (auto it : ChipModules)
    it.second->deallocateDeviceVariablesNoLock(this);
}

// CHIPContext
//*************************************************************************************
CHIPContext::CHIPContext() {}
CHIPContext::~CHIPContext() {}

void CHIPContext::syncQueues(CHIPQueue* target_queue) {
  logDebug("CHIPContext::syncQueues()");
  std::vector<CHIPQueue*> queues = getQueues();
  std::vector<CHIPQueue*> blocking_queues;

  // Default queue gets created add init - always 0th in queue list
  CHIPQueue* default_queue = queues[0];
  queues.erase(queues.begin());

  for (auto& q : queues)
    if (q->getQueueType() == CHIPQueueType::Blocking)
      blocking_queues.push_back(q);
  logDebug("Num blocking queues: {}", blocking_queues.size());

  // default stream waits on all blocking streams to complete
  std::vector<CHIPEvent*> events_to_wait_on;
  CHIPEvent* signal;

  // if (target_queue == default_queue) {
  //   for (auto &q : blocking_queues)
  //     events_to_wait_on.push_back(q->getLastEvent());
  //   signal = target_queue->enqueueBarrierImpl(&events_to_wait_on);
  //   target_queue->LastEvent = signal;  // TODO: replace with
  // } else {  // blocking stream must wait until default stream is done
  //   events_to_wait_on.push_back(default_queue->LastEvent);
  //   signal = target_queue->enqueueBarrierImpl(&events_to_wait_on);
  //   target_queue->LastEvent = signal;  // TODO: replace with
  // }
}

void CHIPContext::addDevice(CHIPDevice* dev) {
  logDebug("CHIPContext.add_device() {}", dev->getName());
  ChipDevices_.push_back(dev);
}

std::vector<CHIPDevice*>& CHIPContext::getDevices() {
  if (ChipDevices_.size() == 0)
    logWarn("CHIPContext.get_devices() was called but chip_devices is empty");
  return ChipDevices_;
}

std::vector<CHIPQueue*>& CHIPContext::getQueues() {
  if (ChipQueues_.size() == 0) {
    std::string msg = "No queus in this context";
    CHIPERR_LOG_AND_THROW(msg, hipErrorUnknown);
  }
  return ChipQueues_;
}
void CHIPContext::addQueue(CHIPQueue* q) {
  logDebug("CHIPContext.add_queue()");
  ChipQueues_.push_back(q);
}
hipStream_t CHIPContext::findQueue(hipStream_t stream) {
  std::vector<CHIPQueue*> Queues = getQueues();
  if (stream == nullptr)
    return Backend->getActiveQueue();

  auto I = std::find(Queues.begin(), Queues.end(), stream);
  if (I == Queues.end())
    return nullptr;
  return *I;
}

void CHIPContext::finishAll() {
  for (CHIPQueue* q : ChipQueues_)
    q->finish();
}

void* CHIPContext::allocate(size_t size) {
  return allocate(size, 0, CHIPMemoryType::Shared);
}

void* CHIPContext::allocate(size_t size, CHIPMemoryType mem_type) {
  return allocate(size, 0, mem_type);
}
void* CHIPContext::allocate(size_t size, size_t alignment,
                            CHIPMemoryType mem_type) {
  std::lock_guard<std::mutex> Lock(Mtx);
  void* allocated_ptr;

  CHIPDevice* chip_dev = Backend->getActiveDevice();
  assert(chip_dev->getContext() == this);

  assert(chip_dev->AllocationTracker && "AllocationTracker was not created!");
  if (!chip_dev->AllocationTracker->reserveMem(size))
    return nullptr;
  allocated_ptr = allocateImpl(size, alignment, mem_type);
  if (allocated_ptr == nullptr)
    chip_dev->AllocationTracker->releaseMemReservation(size);

  chip_dev->AllocationTracker->recordAllocation(allocated_ptr, size);

  return allocated_ptr;
}

hipError_t CHIPContext::findPointerInfo(hipDeviceptr_t* pbase, size_t* psize,
                                        hipDeviceptr_t dptr) {
  // allocation_info *info = Backend->AllocationTracker.getByDevPtr(dptr);
  AllocationInfo* info =
      Backend->getActiveDevice()->AllocationTracker->getByDevPtr(dptr);
  if (!info)
    return hipErrorInvalidDevicePointer;
  *pbase = info->BasePtr;
  *psize = info->Size;
  return hipSuccess;
}

unsigned int CHIPContext::getFlags() { return Flags_; }

void CHIPContext::setFlags(unsigned int flags_) { Flags_ = flags_; }

void CHIPContext::reset() {
  logDebug("Resetting CHIPContext: deleting allocations");
  // Free all allocations in this context
  for (auto& ptr : AloocatedPtrs_)
    freeImpl(ptr);
  // Free all the memory reservations on each device
  for (auto& dev : ChipDevices_)
    dev->AllocationTracker->releaseMemReservation(
        dev->AllocationTracker->TotalMemSize);
  AloocatedPtrs_.clear();

  // TODO Is all the state reset?
}

CHIPContext* CHIPContext::retain() { UNIMPLEMENTED(nullptr); }

hipError_t CHIPContext::free(void* ptr) {
  CHIPDevice* chip_dev = Backend->getActiveDevice();
  AllocationInfo* info = chip_dev->AllocationTracker->getByDevPtr(ptr);
  if (!info)
    return hipErrorInvalidDevicePointer;

  chip_dev->AllocationTracker->releaseMemReservation(info->Size);
  freeImpl(ptr);
  return hipSuccess;
}

// CHIPBackend
//*************************************************************************************

std::string CHIPBackend::getJitFlags() {
  std::string flags;
  if (CustomJitFlags != "") {
    flags = CustomJitFlags;
  } else {
    flags = getDefaultJitFlags();
  }
  logDebug("JIT compiler flags: {}", flags);
  return flags;
}

CHIPBackend::CHIPBackend() { logDebug("CHIPBackend Base Constructor"); };
CHIPBackend::~CHIPBackend() {
  logDebug("CHIPBackend Destructor. Deleting all pointers.");
  ChipExecStack.empty();
  for (auto& ctx : ChipContexts)
    delete ctx;
  for (auto& q : ChipQueues)
    delete q;
  for (auto& mod : ModulesStr_)
    delete mod;
}

void CHIPBackend::initialize(std::string platform_str,
                             std::string device_type_str,
                             std::string device_ids_str) {
  initializeImpl(platform_str, device_type_str, device_ids_str);
  CustomJitFlags = read_env_var("CHIP_JIT_FLAGS", false);
  if (ChipDevices.size() == 0) {
    std::string msg = "No CHIPDevices were initialized";
    CHIPERR_LOG_AND_THROW(msg, hipErrorInitializationError);
  }
  setActiveDevice(ChipDevices[0]);
}

void CHIPBackend::setActiveDevice(CHIPDevice* chip_dev) {
  auto I = std::find(ChipDevices.begin(), ChipDevices.end(), chip_dev);
  if (I == ChipDevices.end()) {
    std::string msg =
        "Tried to set active device with CHIPDevice pointer that is not in "
        "CHIPBackend::chip_devices";
    CHIPERR_LOG_AND_THROW(msg, hipErrorLaunchFailure);
  };
  ActiveDev_ = chip_dev;
  ActiveCtx_ = chip_dev->getContext();
  ActiveQ_ = chip_dev->getActiveQueue();
}
std::vector<CHIPQueue*>& CHIPBackend::getQueues() { return ChipQueues; }
CHIPQueue* CHIPBackend::getActiveQueue() {
  if (ActiveQ_ == nullptr) {
    std::string msg = "Active queue is null";
    CHIPERR_LOG_AND_THROW(msg, hipErrorUnknown);
  }
  return ActiveQ_;
};

CHIPContext* CHIPBackend::getActiveContext() {
  if (ActiveCtx_ == nullptr) {
    std::string msg = "Active context is null";
    CHIPERR_LOG_AND_THROW(msg, hipErrorUnknown);
  }
  return ActiveCtx_;
};

CHIPDevice* CHIPBackend::getActiveDevice() {
  if (ActiveDev_ == nullptr) {
    CHIPERR_LOG_AND_THROW(
        "CHIPBackend.getActiveDevice() was called but active_ctx is null",
        hipErrorUnknown);
  }
  return ActiveDev_;
};

std::vector<CHIPDevice*>& CHIPBackend::getDevices() { return ChipDevices; }

size_t CHIPBackend::getNumDevices() { return ChipDevices.size(); }
std::vector<std::string*>& CHIPBackend::getModulesStr() { return ModulesStr_; }

void CHIPBackend::addContext(CHIPContext* ctx_in) {
  ChipContexts.push_back(ctx_in);
}
void CHIPBackend::addQueue(CHIPQueue* q_in) {
  logDebug("CHIPBackend.add_queue()");
  ChipQueues.push_back(q_in);
}
void CHIPBackend::addDevice(CHIPDevice* dev_in) {
  logDebug("CHIPDevice.add_device() {}", dev_in->getName());
  ChipDevices.push_back(dev_in);
}

void CHIPBackend::registerModuleStr(std::string* mod_str) {
  logDebug("CHIPBackend->register_module()");
  std::lock_guard<std::mutex> Lock(Mtx_);
  getModulesStr().push_back(mod_str);
}

void CHIPBackend::unregisterModuleStr(std::string* mod_str) {
  logDebug("CHIPBackend->unregister_module()");
  auto found_mod = std::find(ModulesStr_.begin(), ModulesStr_.end(), mod_str);
  if (found_mod != ModulesStr_.end()) {
    getModulesStr().erase(found_mod);
  } else {
    logWarn("Module {} not found in CHIPBackend.modules_str while trying to "
            "unregister",
            (void*)mod_str);
  }
}

hipError_t CHIPBackend::configureCall(dim3 grid, dim3 block, size_t shared,
                                      hipStream_t q) {
  std::lock_guard<std::mutex> Lock(Mtx_);
  logDebug("CHIPBackend->configureCall(grid=({},{},{}), block=({},{},{}), "
           "shared={}, q={}",
           grid.x, grid.y, grid.z, block.x, block.y, block.z, shared, (void*)q);
  if (q == nullptr)
    q = getActiveQueue();
  CHIPExecItem* ex = new CHIPExecItem(grid, block, shared, q);
  ChipExecStack.push(ex);

  return hipSuccess;
}

hipError_t CHIPBackend::setArg(const void* arg, size_t size, size_t offset) {
  logDebug("CHIPBackend->set_arg()");
  std::lock_guard<std::mutex> Lock(Mtx_);
  CHIPExecItem* ex = ChipExecStack.top();
  ex->setArg(arg, size, offset);

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

bool CHIPBackend::registerFunctionAsKernel(std::string* module_str,
                                           const void* host_f_ptr,
                                           const char* host_f_name) {
  logDebug("CHIPBackend.registerFunctionAsKernel()");
  for (auto& ctx : ChipContexts)
    for (auto& dev : ctx->getDevices())
      dev->registerFunctionAsKernel(module_str, host_f_ptr, host_f_name);
  return true;
}

void CHIPBackend::registerDeviceVariable(std::string* ModuleStr,
                                         const void* HostPtr, const char* Name,
                                         size_t Size) {
  for (auto* Ctx : ChipContexts)
    for (auto* Dev : Ctx->getDevices())
      Dev->registerDeviceVariable(ModuleStr, HostPtr, Name, Size);
}

CHIPDevice*
CHIPBackend::findDeviceMatchingProps(const hipDeviceProp_t* properties) {
  CHIPDevice* matched_device;
  int maxMatchedCount = 0;
  for (auto& dev : ChipDevices) {
    hipDeviceProp_t currentProp = {0};
    dev->copyDeviceProperties(&currentProp);
    int validPropCount = 0;
    int matchedCount = 0;
    if (properties->major != 0) {
      validPropCount++;
      if (currentProp.major >= properties->major) {
        matchedCount++;
      }
    }
    if (properties->minor != 0) {
      validPropCount++;
      if (currentProp.minor >= properties->minor) {
        matchedCount++;
      }
    }
    if (properties->totalGlobalMem != 0) {
      validPropCount++;
      if (currentProp.totalGlobalMem >= properties->totalGlobalMem) {
        matchedCount++;
      }
    }
    if (properties->sharedMemPerBlock != 0) {
      validPropCount++;
      if (currentProp.sharedMemPerBlock >= properties->sharedMemPerBlock) {
        matchedCount++;
      }
    }
    if (properties->maxThreadsPerBlock != 0) {
      validPropCount++;
      if (currentProp.maxThreadsPerBlock >= properties->maxThreadsPerBlock) {
        matchedCount++;
      }
    }
    if (properties->totalConstMem != 0) {
      validPropCount++;
      if (currentProp.totalConstMem >= properties->totalConstMem) {
        matchedCount++;
      }
    }
    if (properties->multiProcessorCount != 0) {
      validPropCount++;
      if (currentProp.multiProcessorCount >= properties->multiProcessorCount) {
        matchedCount++;
      }
    }
    if (properties->maxThreadsPerMultiProcessor != 0) {
      validPropCount++;
      if (currentProp.maxThreadsPerMultiProcessor >=
          properties->maxThreadsPerMultiProcessor) {
        matchedCount++;
      }
    }
    if (properties->memoryClockRate != 0) {
      validPropCount++;
      if (currentProp.memoryClockRate >= properties->memoryClockRate) {
        matchedCount++;
      }
    }
    if (properties->memoryBusWidth != 0) {
      validPropCount++;
      if (currentProp.memoryBusWidth >= properties->memoryBusWidth) {
        matchedCount++;
      }
    }
    if (properties->l2CacheSize != 0) {
      validPropCount++;
      if (currentProp.l2CacheSize >= properties->l2CacheSize) {
        matchedCount++;
      }
    }
    if (properties->regsPerBlock != 0) {
      validPropCount++;
      if (currentProp.regsPerBlock >= properties->regsPerBlock) {
        matchedCount++;
      }
    }
    if (properties->maxSharedMemoryPerMultiProcessor != 0) {
      validPropCount++;
      if (currentProp.maxSharedMemoryPerMultiProcessor >=
          properties->maxSharedMemoryPerMultiProcessor) {
        matchedCount++;
      }
    }
    if (properties->warpSize != 0) {
      validPropCount++;
      if (currentProp.warpSize >= properties->warpSize) {
        matchedCount++;
      }
    }
    if (validPropCount == matchedCount) {
      matched_device = matchedCount > maxMatchedCount ? dev : matched_device;
      maxMatchedCount = std::max(matchedCount, maxMatchedCount);
    }
  }
  return matched_device;
}

CHIPQueue* CHIPBackend::findQueue(CHIPQueue* q) {
  if (q == nullptr) {
    logDebug("CHIPBackend::findQueue() was given a nullptr. Returning default "
             "queue");
    return Backend->getActiveQueue();
  }
  auto queues = Backend->getActiveDevice()->getQueues();
  auto q_found = std::find(queues.begin(), queues.end(), q);
  if (q_found == queues.end())
    CHIPERR_LOG_AND_THROW(
        "CHIPBackend::findQueue() was given a non-nullptr queue but this queue "
        "was not found among the backend queues.",
        hipErrorTbd);
  return *q_found;
}

// hipError_t CHIPBackend::removeModule(CHIPModule *chip_module){};
// CHIPModule *CHIPBackend::addModule(std::string *module_str) {
//   for (auto &ctx : chip_contexts)
//     for (auto &dev : ctx->getDevices()) dev->addModule(modules_str);
//}
// CHIPQueue
//*************************************************************************************
CHIPQueue::CHIPQueue(CHIPDevice* chip_device_, unsigned int flags_,
                     int priority_)
    : ChipDevice_(chip_device_), Flags_(flags_), Priority_(priority_) {
  ChipContext_ = chip_device_->getContext();
  QueueType_ = CHIPQueueType{flags_};
};
CHIPQueue::CHIPQueue(CHIPDevice* chip_device_, unsigned int flags_)
    : CHIPQueue(chip_device_, flags_, 0){};
CHIPQueue::CHIPQueue(CHIPDevice* chip_device_)
    : CHIPQueue(chip_device_, 0, 0){};
CHIPQueue::~CHIPQueue(){};

///////// Enqueue Operations //////////
CHIPEvent* CHIPQueue::memCopyImpl(void* dst, const void* src, size_t size) {
  auto ev = memCopyAsyncImpl(dst, src, size);
  finish();
  return ev;
}
hipError_t CHIPQueue::memCopy(void* dst, const void* src, size_t size) {
  std::lock_guard<std::mutex> Lock(Mtx);
#ifdef ENFORCE_QUEUE_SYNC
  chip_context->syncQueues(this);
#endif
  auto ev = memCopyImpl(dst, src, size);
  ev->Msg = "memCopy";
  updateLastEvent(ev);
  return hipSuccess;
}
hipError_t CHIPQueue::memCopyAsync(void* dst, const void* src, size_t size) {
  std::lock_guard<std::mutex> Lock(Mtx);
#ifdef ENFORCE_QUEUE_SYNC
  chip_context->syncQueues(this);
#endif
  auto ev = memCopyAsyncImpl(dst, src, size);
  ev->Msg = "memCopyAsync";
  updateLastEvent(ev);
  return hipSuccess;
}
void CHIPQueue::memFill(void* dst, size_t size, const void* pattern,
                        size_t pattern_size) {
  std::lock_guard<std::mutex> Lock(Mtx);
#ifdef ENFORCE_QUEUE_SYNC
  chip_context->syncQueues(this);
#endif
  auto ev = memFillImpl(dst, size, pattern, pattern_size);
  ev->Msg = "memFill";
  updateLastEvent(ev);
}
CHIPEvent* CHIPQueue::memFillImpl(void* dst, size_t size, const void* pattern,
                                  size_t pattern_size) {
  auto ev = memFillAsyncImpl(dst, size, pattern, pattern_size);
  finish();
  return ev;
}
void CHIPQueue::memFillAsync(void* dst, size_t size, const void* pattern,
                             size_t pattern_size) {
  std::lock_guard<std::mutex> Lock(Mtx);
#ifdef ENFORCE_QUEUE_SYNC
  chip_context->syncQueues(this);
#endif
  auto ev = memFillAsyncImpl(dst, size, pattern, pattern_size);
  ev->Msg = "memFillAsync";
  updateLastEvent(ev);
}
void CHIPQueue::memCopy2D(void* dst, size_t dpitch, const void* src,
                          size_t spitch, size_t width, size_t height) {
  std::lock_guard<std::mutex> Lock(Mtx);
#ifdef ENFORCE_QUEUE_SYNC
  chip_context->syncQueues(this);
#endif
  auto ev = memCopy2DAsyncImpl(dst, dpitch, src, spitch, width, height);
  ev->Msg = "memCopy2D";
  finish();
  updateLastEvent(ev);
}
CHIPEvent* CHIPQueue::memCopy2DImpl(void* dst, size_t dpitch, const void* src,
                                    size_t spitch, size_t width,
                                    size_t height) {
#ifdef ENFORCE_QUEUE_SYNC
  chip_context->syncQueues(this);
#endif
  auto ev = memCopy2DAsyncImpl(dst, dpitch, src, spitch, width, height);
  finish();
  return ev;
}
void CHIPQueue::memCopy2DAsync(void* dst, size_t dpitch, const void* src,
                               size_t spitch, size_t width, size_t height) {
  std::lock_guard<std::mutex> Lock(Mtx);
#ifdef ENFORCE_QUEUE_SYNC
  chip_context->syncQueues(this);
#endif
  auto ev = memCopy2DAsyncImpl(dst, dpitch, src, spitch, width, height);
  ev->Msg = "memCopy2DAsync";
  updateLastEvent(ev);
}
void CHIPQueue::memCopy3D(void* dst, size_t dpitch, size_t dspitch,
                          const void* src, size_t spitch, size_t sspitch,
                          size_t width, size_t height, size_t depth) {
  std::lock_guard<std::mutex> Lock(Mtx);
#ifdef ENFORCE_QUEUE_SYNC
  chip_context->syncQueues(this);
#endif
  auto ev = memCopy3DAsyncImpl(dst, dpitch, dspitch, src, spitch, sspitch,
                               width, height, depth);
  ev->Msg = "memCopy3D";
  finish();
  updateLastEvent(ev);
}
CHIPEvent* CHIPQueue::memCopy3DImpl(void* dst, size_t dpitch, size_t dspitch,
                                    const void* src, size_t spitch,
                                    size_t sspitch, size_t width, size_t height,
                                    size_t depth) {
#ifdef ENFORCE_QUEUE_SYNC
  chip_context->syncQueues(this);
#endif
  auto ev = memCopy3DAsyncImpl(dst, dpitch, dspitch, src, spitch, sspitch,
                               width, height, depth);
  finish();
  return ev;
}
void CHIPQueue::memCopy3DAsync(void* dst, size_t dpitch, size_t dspitch,
                               const void* src, size_t spitch, size_t sspitch,
                               size_t width, size_t height, size_t depth) {
#ifdef ENFORCE_QUEUE_SYNC
  chip_context->syncQueues(this);
#endif
  auto ev = memCopy3DAsyncImpl(dst, dpitch, dspitch, src, spitch, sspitch,
                               width, height, depth);
  std::lock_guard<std::mutex> Lock(Mtx);
  ev->Msg = "memCopy3DAsync";
  updateLastEvent(ev);
}
void CHIPQueue::memCopyToTexture(CHIPTexture* texObj, void* src) {
  std::lock_guard<std::mutex> Lock(Mtx);
#ifdef ENFORCE_QUEUE_SYNC
  chip_context->syncQueues(this);
#endif
  auto ev = memCopyToTextureImpl(texObj, src);
  ev->Msg = "memCopyToTexture";
  updateLastEvent(ev);
}
CHIPEvent* CHIPQueue::launch(CHIPExecItem* exec_item) {
  // std::lock_guard<std::mutex> Lock(mtx);
#ifdef ENFORCE_QUEUE_SYNC
  chip_context->syncQueues(this);
#endif
  auto ev = launchImpl(exec_item);
  ev->Msg = "launch";
  updateLastEvent(ev);
  return ev;
}
CHIPEvent* CHIPQueue::enqueueBarrier(std::vector<CHIPEvent*>* eventsToWaitFor) {
  std::lock_guard<std::mutex> Lock(Mtx);
  auto ev = enqueueBarrierImpl(eventsToWaitFor);
  ev->Msg = "enqueueBarrier";
  updateLastEvent(ev);
  return ev;
}
CHIPEvent* CHIPQueue::enqueueMarker() {
  std::lock_guard<std::mutex> Lock(Mtx);
#ifdef ENFORCE_QUEUE_SYNC
  chip_context->syncQueues(this);
#endif
  auto ev = enqueueMarkerImpl();
  ev->Msg = "enqueueMarker";
  updateLastEvent(ev);
  return ev;
}

void CHIPQueue::memPrefetch(const void* ptr, size_t count) {
  std::lock_guard<std::mutex> Lock(Mtx);
#ifdef ENFORCE_QUEUE_SYNC
  chip_context->syncQueues(this);
#endif
  auto ev = memPrefetchImpl(ptr, count);
  ev->Msg = "memPrefetch";
  updateLastEvent(ev);
}

void CHIPQueue::launchHostFunc(const void* hostFunction, dim3 numBlocks,
                               dim3 dimBlocks, void** args,
                               size_t sharedMemBytes) {
  CHIPExecItem e(numBlocks, dimBlocks, sharedMemBytes, this);
  e.setArgPointer(args);
  auto ev = e.launchByHostPtr(hostFunction);
  ev->Msg = "launchHostFunc";
  updateLastEvent(ev);
}

void CHIPQueue::launchWithKernelParams(dim3 grid, dim3 block,
                                       unsigned int sharedMemBytes, void** args,
                                       CHIPKernel* kernel) {
  UNIMPLEMENTED();
}

void CHIPQueue::launchWithExtraParams(dim3 grid, dim3 block,
                                      unsigned int sharedMemBytes, void** extra,
                                      CHIPKernel* kernel) {
  UNIMPLEMENTED();
}

///////// End Enqueue Operations //////////

CHIPDevice* CHIPQueue::getDevice() {
  if (ChipDevice_ == nullptr) {
    std::string msg = "chip_device is null";
    CHIPERR_LOG_AND_THROW(msg, hipErrorLaunchFailure);
  }

  return ChipDevice_;
}

unsigned int CHIPQueue::getFlags() { return Flags_; }
// hipError_t CHIPQueue::memCopy(void *dst, const void *src, size_t size) {}
// hipError_t CHIPQueue::memCopyAsync(void *, void const *, unsigned long) {}

int CHIPQueue::getPriorityRange(int lower_or_upper) { UNIMPLEMENTED(0); }
int CHIPQueue::getPriority() { UNIMPLEMENTED(0); }
bool CHIPQueue::addCallback(hipStreamCallback_t callback, void* userData) {
  CHIPCallbackData* cb = Backend->createCallbackData(callback, userData, this);

  Backend->CallbackStack.push(cb);

  // Setup event handling on the CPU side
  if (!EventMonitor_)
    EventMonitor_ = Backend->createEventMonitor();
  return true;
}

bool CHIPQueue::query() { UNIMPLEMENTED(true); }
