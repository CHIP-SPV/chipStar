#include "CHIPBackend.hh"

#include <utility>

CHIPCallbackData::CHIPCallbackData(hipStreamCallback_t callback_f_,
                                   void *callback_args_, CHIPQueue *chip_queue_)
    : callback_f(callback_f_),
      callback_args(callback_args_),
      chip_queue(chip_queue_) {
  setup();
}

void CHIPCallbackData::setup() {
  CHIPContext *ctx = chip_queue->getContext();
  gpu_ready = Backend->createCHIPEvent(ctx);
  cpu_callback_complete = Backend->createCHIPEvent(ctx);
  gpu_ack = Backend->createCHIPEvent(ctx);

  auto gpu_ready = chip_queue->enqueueBarrier(nullptr);

  std::vector<CHIPEvent *> evs = {cpu_callback_complete};
  chip_queue->enqueueBarrier(&evs);

  CHIPEvent *gpu_ack = chip_queue->enqueueMarker();
}

void CHIPEventMonitor::monitor() {
  logTrace("CHIPEventMonitor::monitor()");
  std::lock_guard<std::mutex> Lock(mtx);
  CHIPCallbackData *cb;
  while (Backend->getCallback(&cb)) {
    cb->gpu_ready->wait();
    cb->execute(hipSuccess);
    cb->cpu_callback_complete->hostSignal();
    cb->gpu_ack->wait();
    delete cb;
  }

  // no more callback events left, free up the thread
  delete this;
  pthread_yield();
}

// CHIPDeviceVar
// ************************************************************************
CHIPDeviceVar::CHIPDeviceVar(std::string host_var_name_, void *dev_ptr_,
                             size_t size_)
    : host_var_name(host_var_name_), dev_ptr(dev_ptr_), size(size_) {}
CHIPDeviceVar::~CHIPDeviceVar() {}
size_t CHIPDeviceVar::getSize() { return size; }
std::string CHIPDeviceVar::getName() { return host_var_name; }
void *CHIPDeviceVar::getDevAddr() { return dev_ptr; }

// CHIPAllocationTracker
// ************************************************************************
CHIPAllocationTracker::CHIPAllocationTracker(size_t global_mem_size_,
                                             std::string name_)
    : global_mem_size(global_mem_size_), total_mem_used(0), max_mem_used(0) {
  name = name_;
}
CHIPAllocationTracker::~CHIPAllocationTracker() {
  // TODO free up all the pointers in ptr_set
  UNIMPLEMENTED();
}

allocation_info *CHIPAllocationTracker::getByHostPtr(const void *host_ptr) {
  auto found = host_to_dev.find(const_cast<void *>(host_ptr));
  if (found == host_to_dev.end()) {
    CHIPERR_LOG_AND_THROW("Unable to find allocation info for host pointer",
                          hipErrorInvalidSymbol);
  }
  return getByDevPtr(found->second);
}
allocation_info *CHIPAllocationTracker::getByDevPtr(const void *dev_ptr) {
  auto ptr = const_cast<void *>(dev_ptr);
  logDebug("dev_to_allocation_info size: {}", dev_to_allocation_info.size());
  auto c = dev_to_allocation_info.count(ptr);
  if (c == 0) CHIPERR_LOG_AND_THROW("pointer not found on device", hipErrorTbd);

  return &dev_to_allocation_info[const_cast<void *>(dev_ptr)];
}

bool CHIPAllocationTracker::reserveMem(size_t bytes) {
  std::lock_guard<std::mutex> Lock(mtx);
  if (bytes <= (global_mem_size - total_mem_used)) {
    total_mem_used += bytes;
    if (total_mem_used > max_mem_used) max_mem_used = total_mem_used;
    logDebug("Currently used memory on dev {}: {} M\n", name,
             (total_mem_used >> 20));
    return true;
  } else {
    CHIPERR_LOG_AND_THROW("Failed to allocate memory",
                          hipErrorMemoryAllocation);
  }
}

bool CHIPAllocationTracker::releaseMemReservation(unsigned long bytes) {
  std::lock_guard<std::mutex> Lock(mtx);
  if (total_mem_used >= bytes) {
    total_mem_used -= bytes;
    return true;
  }

  return false;
}

void CHIPAllocationTracker::recordAllocation(void *dev_ptr, size_t size_) {
  allocation_info alloc_info{dev_ptr, size_};
  dev_to_allocation_info[dev_ptr] = alloc_info;
  logDebug("CHIPAllocationTracker::recordAllocation size: {}",
           dev_to_allocation_info.size());
  return;
}

// CHIPEvent
// ************************************************************************

void CHIPEvent::recordStream(CHIPQueue *chip_queue) {
  logDebug("CHIPEvent::recordStream()");
  std::lock_guard<std::mutex> Lock(mtx);
  assert(chip_queue->getLastEvent() != nullptr);
  this->takeOver(chip_queue->getLastEvent());
  event_status = EVENT_STATUS_RECORDING;
}

CHIPEvent::CHIPEvent(CHIPContext *ctx_in, CHIPEventFlags flags_)
    : event_status(EVENT_STATUS_INIT),
      flags(flags_),
      chip_context(ctx_in),
      refc(new size_t(1)) {}

// CHIPModuleflags_
//*************************************************************************************
void CHIPModule::consumeSPIRV() {
  funcIL = (uint8_t *)src.data();
  ilSize = src.length();

  // Parse the SPIR-V fat binary to retrieve kernel function
  size_t numWords = ilSize / 4;
  binary_data = new int32_t[numWords + 1];
  std::memcpy(binary_data, funcIL, ilSize);
  // Extract kernel function information
  bool res = parseSPIR(binary_data, numWords, func_infos);
  delete[] binary_data;
  if (!res) {
    CHIPERR_LOG_AND_THROW("SPIR-V parsing failed", hipErrorUnknown);
  }
}

CHIPModule::CHIPModule(std::string *module_str) {
  src = *module_str;
  consumeSPIRV();
}
CHIPModule::CHIPModule(std::string &&module_str) {
  src = module_str;
  consumeSPIRV();
}
CHIPModule::~CHIPModule() {}

void CHIPModule::addKernel(CHIPKernel *kernel) {
  chip_kernels.push_back(kernel);
}

void CHIPModule::compileOnce(CHIPDevice *chip_dev) {
  std::call_once(compiled, &CHIPModule::compile, this, chip_dev);
}

CHIPKernel *CHIPModule::getKernel(std::string name) {
  auto kernel = std::find_if(
      chip_kernels.begin(), chip_kernels.end(),
      [name](CHIPKernel *k) { return k->getName().compare(name) == 0; });
  if (kernel == chip_kernels.end()) {
    std::string msg = "Failed to find kernel via kernel name: " + name;
    CHIPERR_LOG_AND_THROW(msg, hipErrorLaunchFailure);
  }

  return *kernel;
}

CHIPKernel *CHIPModule::getKernel(const void *host_f_ptr) {
  for (auto &kernel : chip_kernels)
    logTrace("chip kernel: {} {}", kernel->getHostPtr(), kernel->getName());
  auto kernel = std::find_if(
      chip_kernels.begin(), chip_kernels.end(),
      [host_f_ptr](CHIPKernel *k) { return k->getHostPtr() == host_f_ptr; });
  if (kernel == chip_kernels.end()) {
    std::string msg = "Failed to find kernel via host pointer";
    CHIPERR_LOG_AND_THROW(msg, hipErrorLaunchFailure);
  }

  return *kernel;
}

std::vector<CHIPKernel *> &CHIPModule::getKernels() { return chip_kernels; }

CHIPDeviceVar *CHIPModule::getGlobalVar(const char *var_name_) {
  auto var = std::find_if(chip_vars.begin(), chip_vars.end(),
                          [var_name_](CHIPDeviceVar *v) {
                            return v->getName().compare(var_name_) == 0;
                          });
  if (var == chip_vars.end()) {
    std::string msg =
        "Failed to find global variable by name: " + std::string(var_name_);
    CHIPERR_LOG_AND_THROW(msg, hipErrorLaunchFailure);
  }

  return *var;
}

// CHIPKernel
//*************************************************************************************
CHIPKernel::CHIPKernel(std::string host_f_name_, OCLFuncInfo *func_info_)
    : host_f_name(host_f_name_), func_info(func_info_) {}
CHIPKernel::~CHIPKernel(){};
std::string CHIPKernel::getName() { return host_f_name; }
const void *CHIPKernel::getHostPtr() { return host_f_ptr; }
const void *CHIPKernel::getDevPtr() { return dev_f_ptr; }

OCLFuncInfo *CHIPKernel::getFuncInfo() { return func_info; }

void CHIPKernel::setName(std::string host_f_name_) {
  host_f_name = host_f_name_;
}
void CHIPKernel::setHostPtr(const void *host_f_ptr_) {
  host_f_ptr = host_f_ptr_;
}
void CHIPKernel::setDevPtr(const void *dev_f_ptr_) { dev_f_ptr = dev_f_ptr_; }

// CHIPExecItem
//*************************************************************************************
CHIPExecItem::CHIPExecItem(dim3 grid_dim_, dim3 block_dim_, size_t shared_mem_,
                           hipStream_t chip_queue_)
    : grid_dim(grid_dim_),
      block_dim(block_dim_),
      shared_mem(shared_mem_),
      chip_queue(chip_queue_){};
CHIPExecItem::~CHIPExecItem(){};

std::vector<uint8_t> CHIPExecItem::getArgData() { return arg_data; }

void CHIPExecItem::setArg(const void *arg, size_t size, size_t offset) {
  if ((offset + size) > arg_data.size()) arg_data.resize(offset + size + 1024);

  std::memcpy(arg_data.data() + offset, arg, size);
  logDebug("CHIPExecItem.setArg() on {} size {} offset {}\n", (void *)this,
           size, offset);
  offset_sizes.push_back(std::make_tuple(offset, size));
}

CHIPEvent *CHIPExecItem::launchByHostPtr(const void *hostPtr) {
  logTrace("launchByHostPtr");
  if (chip_queue == nullptr) {
    std::string msg = "Tried to launch CHIPExecItem but its queue is null";
    CHIPERR_LOG_AND_THROW(msg, hipErrorLaunchFailure);
  }

  CHIPDevice *dev = chip_queue->getDevice();
  this->chip_kernel = dev->findKernelByHostPtr(hostPtr);
  return (chip_queue->launchImpl(this));
}

dim3 CHIPExecItem::getBlock() { return block_dim; }
dim3 CHIPExecItem::getGrid() { return grid_dim; }
CHIPKernel *CHIPExecItem::getKernel() { return chip_kernel; }
size_t CHIPExecItem::getSharedMem() { return shared_mem; }
CHIPQueue *CHIPExecItem::getQueue() { return chip_queue; }
// CHIPDevice
//*************************************************************************************
CHIPDevice::CHIPDevice(CHIPContext *ctx_) : ctx(ctx_) {}

CHIPDevice::CHIPDevice() {
  logDebug("Device {} is {}: name \"{}\" \n", idx, (void *)this,
           hip_device_props.name);
}
CHIPDevice::~CHIPDevice() {}

std::vector<CHIPKernel *> CHIPDevice::getKernels() {
  std::vector<CHIPKernel *> kernels;
  for (auto &module : chip_modules) {
    for (CHIPKernel *kernel : module->getKernels()) kernels.push_back(kernel);
  }
  return kernels;
}
std::vector<CHIPModule *> &CHIPDevice::getModules() { return chip_modules; }

std::string CHIPDevice::getName() {
  populateDeviceProperties();
  return std::string(hip_device_props.name);
}

void CHIPDevice::populateDeviceProperties() {
  std::call_once(propsPopulated, &CHIPDevice::populateDeviceProperties_, this);
  if (!allocation_tracker)
    allocation_tracker = new CHIPAllocationTracker(
        hip_device_props.totalGlobalMem, hip_device_props.name);
}
void CHIPDevice::copyDeviceProperties(hipDeviceProp_t *prop) {
  logTrace("CHIPDevice->copy_device_properties()");
  if (prop) std::memcpy(prop, &this->hip_device_props, sizeof(hipDeviceProp_t));
}

CHIPKernel *CHIPDevice::findKernelByHostPtr(const void *hostPtr) {
  logTrace("CHIPDevice::findKernelByHostPtr({})", hostPtr);
  std::vector<CHIPKernel *> chip_kernels = getKernels();
  if (chip_kernels.size() == 0) {
    CHIPERR_LOG_AND_THROW("chip_kernels is empty for this device",
                          hipErrorLaunchFailure);
  }
  logDebug("Listing Kernels for device {}", getName());
  for (auto &kernel : chip_kernels) {
    logTrace("Kernel name: {} host_f_ptr: {}", kernel->getName(),
             kernel->getHostPtr());
  }

  auto found_kernel = std::find_if(chip_kernels.begin(), chip_kernels.end(),
                                   [&hostPtr](CHIPKernel *kernel) {
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

CHIPContext *CHIPDevice::getContext() { return ctx; }
int CHIPDevice::getDeviceId() { return idx; }

CHIPDeviceVar *CHIPDevice::getStatGlobalVar(const char *var_name_) {
  logDebug("CHIPDeviceVar::getStatGlobalVar({})", var_name_);
  CHIPDeviceVar *found_var = nullptr;
  for (auto mod : chip_modules) {
    found_var = mod->getGlobalVar(var_name_);
    if (found_var) return found_var;
  }
  return nullptr;
}

CHIPDeviceVar *CHIPDevice::getGlobalVar(const char *var_name_) {
  auto found_dyn = getDynGlobalVar(var_name_);
  if (found_dyn) return found_dyn;

  auto found_stat = getStatGlobalVar(var_name_);
  if (found_stat) return found_stat;

  return nullptr;
}

int CHIPDevice::getAttr(hipDeviceAttribute_t attr) {
  int *pi;
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
      *reinterpret_cast<unsigned int **>(pi) = prop.hdpMemFlushCntl;
      break;
    case hipDeviceAttributeHdpRegFlushCntl:
      *reinterpret_cast<unsigned int **>(pi) = prop.hdpRegFlushCntl;
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

size_t CHIPDevice::getGlobalMemSize() {
  return hip_device_props.totalGlobalMem;
}

void CHIPDevice::registerFunctionAsKernel(std::string *module_str,
                                          const void *host_f_ptr,
                                          const char *host_f_name) {
  CHIPModule *chip_module;
  auto found = module_str_to_chip_map.count(module_str);
  if (found) {
    chip_module = module_str_to_chip_map[module_str];
  } else {
    chip_module = addModule(module_str);
    chip_module->compileOnce(this);  // Compile it
    module_str_to_chip_map[module_str] = chip_module;
    // TODO Place it in the Backend cache
  }
  CHIPKernel *kernel = chip_module->getKernel(std::string(host_f_name));
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

void CHIPDevice::addQueue(CHIPQueue *chip_queue_) {
  auto queue_found =
      std::find(chip_queues.begin(), chip_queues.end(), chip_queue_);
  if (queue_found == chip_queues.end()) chip_queues.push_back(chip_queue_);
  return;
}

CHIPQueue *CHIPDevice::addQueue(unsigned int flags, int priority) {
  auto q = addQueue_(flags, priority);
  return q;
}

std::vector<CHIPQueue *> CHIPDevice::getQueues() { return chip_queues; }

hipError_t CHIPDevice::setPeerAccess(CHIPDevice *peer, int flags,
                                     bool canAccessPeer) {
  UNIMPLEMENTED(hipSuccess);
}

int CHIPDevice::getPeerAccess(CHIPDevice *peerDevice) { UNIMPLEMENTED(0); }

void CHIPDevice::setCacheConfig(hipFuncCache_t cfg) { UNIMPLEMENTED(); }

void CHIPDevice::setFuncCacheConfig(const void *func, hipFuncCache_t config) {
  UNIMPLEMENTED();
}

hipFuncCache_t CHIPDevice::getCacheConfig() {
  UNIMPLEMENTED(hipFuncCachePreferNone);
}

hipSharedMemConfig CHIPDevice::getSharedMemConfig() {
  UNIMPLEMENTED(hipSharedMemBankSizeDefault);
}

bool CHIPDevice::removeQueue(CHIPQueue *q) {
  auto found_q = std::find(chip_queues.begin(), chip_queues.end(), q);
  if (found_q == chip_queues.end()) {
    std::string msg =
        "Tried to remove a queue for a device but the queue was not found in "
        "device queue list";
    CHIPERR_LOG_AND_THROW(msg, hipErrorUnknown);
  }

  chip_queues.erase(found_q);
  return true;
}

void CHIPDevice::setSharedMemConfig(hipSharedMemConfig config) {
  UNIMPLEMENTED();
}

size_t CHIPDevice::getUsedGlobalMem() {
  return allocation_tracker->total_mem_used;
}

bool CHIPDevice::hasPCIBusId(int, int, int) { UNIMPLEMENTED(true); }

CHIPQueue *CHIPDevice::getActiveQueue() { return chip_queues[0]; }
// CHIPContext
//*************************************************************************************
CHIPContext::CHIPContext() {}
CHIPContext::~CHIPContext() {}

void CHIPContext::syncQueues(CHIPQueue *target_queue) {
  logDebug("CHIPContext::syncQueues()");
  std::vector<CHIPQueue *> queues = getQueues();
  std::vector<CHIPQueue *> blocking_queues;

  // Default queue gets created add init - always 0th in queue list
  CHIPQueue *default_queue = queues[0];
  queues.erase(queues.begin());

  for (auto &q : queues)
    if (q->getQueueType() == CHIPQueueType::Blocking)
      blocking_queues.push_back(q);
  logDebug("Num blocking queues: {}", blocking_queues.size());

  // default stream waits on all blocking streams to complete
  std::vector<CHIPEvent *> events_to_wait_on;
  CHIPEvent *signal;

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

void CHIPContext::addDevice(CHIPDevice *dev) {
  logTrace("CHIPContext.add_device() {}", dev->getName());
  chip_devices.push_back(dev);
}

std::vector<CHIPDevice *> &CHIPContext::getDevices() {
  if (chip_devices.size() == 0)
    logWarn("CHIPContext.get_devices() was called but chip_devices is empty");
  return chip_devices;
}

std::vector<CHIPQueue *> &CHIPContext::getQueues() {
  if (chip_queues.size() == 0) {
    std::string msg = "No queus in this context";
    CHIPERR_LOG_AND_THROW(msg, hipErrorUnknown);
  }
  return chip_queues;
}
void CHIPContext::addQueue(CHIPQueue *q) {
  logTrace("CHIPContext.add_queue()");
  chip_queues.push_back(q);
}
hipStream_t CHIPContext::findQueue(hipStream_t stream) {
  std::vector<CHIPQueue *> Queues = getQueues();
  if (stream == nullptr) return Backend->getActiveQueue();

  auto I = std::find(Queues.begin(), Queues.end(), stream);
  if (I == Queues.end()) return nullptr;
  return *I;
}

void CHIPContext::finishAll() {
  for (CHIPQueue *q : chip_queues) q->finish();
}

void *CHIPContext::allocate(size_t size) {
  return allocate(size, 0, CHIPMemoryType::Shared);
}

void *CHIPContext::allocate(size_t size, CHIPMemoryType mem_type) {
  return allocate(size, 0, mem_type);
}
void *CHIPContext::allocate(size_t size, size_t alignment,
                            CHIPMemoryType mem_type) {
  std::lock_guard<std::mutex> Lock(mtx);
  void *allocated_ptr;

  CHIPDevice *chip_dev = Backend->getActiveDevice();
  assert(chip_dev->getContext() == this);

  assert(chip_dev->allocation_tracker && "AllocationTracker was not created!");
  if (!chip_dev->allocation_tracker->reserveMem(size)) return nullptr;
  allocated_ptr = allocate_(size, alignment, mem_type);
  if (allocated_ptr == nullptr)
    chip_dev->allocation_tracker->releaseMemReservation(size);

  chip_dev->allocation_tracker->recordAllocation(allocated_ptr, size);

  return allocated_ptr;
}

hipError_t CHIPContext::findPointerInfo(hipDeviceptr_t *pbase, size_t *psize,
                                        hipDeviceptr_t dptr) {
  // allocation_info *info = Backend->AllocationTracker.getByDevPtr(dptr);
  allocation_info *info =
      Backend->getActiveDevice()->allocation_tracker->getByDevPtr(dptr);
  if (!info) return hipErrorInvalidDevicePointer;
  *pbase = info->base_ptr;
  *psize = info->size;
  return hipSuccess;
}

unsigned int CHIPContext::getFlags() { return flags; }

void CHIPContext::setFlags(unsigned int flags_) { flags = flags_; }

void CHIPContext::reset() {
  logDebug("Resetting CHIPContext: deleting allocations");
  // Free all allocations in this context
  for (auto &ptr : allocated_ptrs) free_(ptr);
  // Free all the memory reservations on each device
  for (auto &dev : chip_devices)
    dev->allocation_tracker->releaseMemReservation(
        dev->allocation_tracker->total_mem_used);
  allocated_ptrs.clear();

  // TODO Is all the state reset?
}

CHIPContext *CHIPContext::retain() { UNIMPLEMENTED(nullptr); }

hipError_t CHIPContext::free(void *ptr) {
  CHIPDevice *chip_dev = Backend->getActiveDevice();
  allocation_info *info = chip_dev->allocation_tracker->getByDevPtr(ptr);
  if (!info) return hipErrorInvalidDevicePointer;

  chip_dev->allocation_tracker->releaseMemReservation(info->size);
  free_(ptr);
  return hipSuccess;
}

// CHIPBackend
//*************************************************************************************

std::string CHIPBackend::getJitFlags() {
  std::string flags;
  if (custom_jit_flags != "") {
    flags = custom_jit_flags;
  } else {
    flags = getDefaultJitFlags();
  }
  logDebug("JIT compiler flags: {}", flags);
  return flags;
}

CHIPBackend::CHIPBackend() { logDebug("CHIPBackend Base Constructor"); };
CHIPBackend::~CHIPBackend() {
  logDebug("CHIPBackend Destructor. Deleting all pointers.");
  chip_execstack.empty();
  for (auto &ctx : chip_contexts) delete ctx;
  for (auto &q : chip_queues) delete q;
  for (auto &mod : modules_str) delete mod;
}

void CHIPBackend::initialize(std::string platform_str,
                             std::string device_type_str,
                             std::string device_ids_str) {
  initialize_(platform_str, device_type_str, device_ids_str);
  custom_jit_flags = read_env_var("CHIP_JIT_FLAGS", false);
  if (chip_devices.size() == 0) {
    std::string msg = "No CHIPDevices were initialized";
    CHIPERR_LOG_AND_THROW(msg, hipErrorInitializationError);
  }
  setActiveDevice(chip_devices[0]);
}

void CHIPBackend::setActiveDevice(CHIPDevice *chip_dev) {
  auto I = std::find(chip_devices.begin(), chip_devices.end(), chip_dev);
  if (I == chip_devices.end()) {
    std::string msg =
        "Tried to set active device with CHIPDevice pointer that is not in "
        "CHIPBackend::chip_devices";
    CHIPERR_LOG_AND_THROW(msg, hipErrorLaunchFailure);
  };
  active_dev = chip_dev;
  active_ctx = chip_dev->getContext();
  active_q = chip_dev->getActiveQueue();
}
std::vector<CHIPQueue *> &CHIPBackend::getQueues() { return chip_queues; }
CHIPQueue *CHIPBackend::getActiveQueue() {
  if (active_q == nullptr) {
    std::string msg = "Active queue is null";
    CHIPERR_LOG_AND_THROW(msg, hipErrorUnknown);
  }
  return active_q;
};

CHIPContext *CHIPBackend::getActiveContext() {
  if (active_ctx == nullptr) {
    std::string msg = "Active context is null";
    CHIPERR_LOG_AND_THROW(msg, hipErrorUnknown);
  }
  return active_ctx;
};

CHIPDevice *CHIPBackend::getActiveDevice() {
  if (active_dev == nullptr) {
    CHIPERR_LOG_AND_THROW(
        "CHIPBackend.getActiveDevice() was called but active_ctx is null",
        hipErrorUnknown);
  }
  return active_dev;
};

std::vector<CHIPDevice *> &CHIPBackend::getDevices() { return chip_devices; }

size_t CHIPBackend::getNumDevices() { return chip_devices.size(); }
std::vector<std::string *> &CHIPBackend::getModulesStr() { return modules_str; }

void CHIPBackend::addContext(CHIPContext *ctx_in) {
  chip_contexts.push_back(ctx_in);
}
void CHIPBackend::addQueue(CHIPQueue *q_in) {
  logDebug("CHIPBackend.add_queue()");
  chip_queues.push_back(q_in);
}
void CHIPBackend::addDevice(CHIPDevice *dev_in) {
  logTrace("CHIPDevice.add_device() {}", dev_in->getName());
  chip_devices.push_back(dev_in);
}

void CHIPBackend::registerModuleStr(std::string *mod_str) {
  logTrace("CHIPBackend->register_module()");
  std::lock_guard<std::mutex> Lock(mtx);
  getModulesStr().push_back(mod_str);
}

void CHIPBackend::unregisterModuleStr(std::string *mod_str) {
  logTrace("CHIPBackend->unregister_module()");
  auto found_mod = std::find(modules_str.begin(), modules_str.end(), mod_str);
  if (found_mod != modules_str.end()) {
    getModulesStr().erase(found_mod);
  } else {
    logWarn(
        "Module {} not found in CHIPBackend.modules_str while trying to "
        "unregister",
        (void *)mod_str);
  }
}

hipError_t CHIPBackend::configureCall(dim3 grid, dim3 block, size_t shared,
                                      hipStream_t q) {
  std::lock_guard<std::mutex> Lock(mtx);
  logTrace(
      "CHIPBackend->configureCall(grid=({},{},{}), block=({},{},{}), "
      "shared={}, q={}",
      grid.x, grid.y, grid.z, block.x, block.y, block.z, shared, (void *)q);
  if (q == nullptr) q = getActiveQueue();
  CHIPExecItem *ex = new CHIPExecItem(grid, block, shared, q);
  chip_execstack.push(ex);

  return hipSuccess;
}

hipError_t CHIPBackend::setArg(const void *arg, size_t size, size_t offset) {
  logTrace("CHIPBackend->set_arg()");
  std::lock_guard<std::mutex> Lock(mtx);
  CHIPExecItem *ex = chip_execstack.top();
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

bool CHIPBackend::registerFunctionAsKernel(std::string *module_str,
                                           const void *host_f_ptr,
                                           const char *host_f_name) {
  logTrace("CHIPBackend.registerFunctionAsKernel()");
  for (auto &ctx : chip_contexts)
    for (auto &dev : ctx->getDevices())
      dev->registerFunctionAsKernel(module_str, host_f_ptr, host_f_name);
  return true;
}

CHIPDevice *CHIPBackend::findDeviceMatchingProps(
    const hipDeviceProp_t *properties) {
  CHIPDevice *matched_device;
  int maxMatchedCount = 0;
  for (auto &dev : chip_devices) {
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

CHIPQueue *CHIPBackend::findQueue(CHIPQueue *q) {
  if (q == nullptr) {
    logTrace(
        "CHIPBackend::findQueue() was given a nullptr. Returning default "
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
CHIPQueue::CHIPQueue(CHIPDevice *chip_device_, unsigned int flags_,
                     int priority_)
    : chip_device(chip_device_), flags(flags_), priority(priority_) {
  chip_context = chip_device_->getContext();
  queue_type = CHIPQueueType{flags_};
};
CHIPQueue::CHIPQueue(CHIPDevice *chip_device_, unsigned int flags_)
    : CHIPQueue(chip_device_, flags_, 0){};
CHIPQueue::CHIPQueue(CHIPDevice *chip_device_)
    : CHIPQueue(chip_device_, 0, 0){};
CHIPQueue::~CHIPQueue(){};

///////// Enqueue Operations //////////
CHIPEvent *CHIPQueue::memCopyImpl(void *dst, const void *src, size_t size) {
  auto ev = memCopyAsyncImpl(dst, src, size);
  finish();
  return ev;
}
hipError_t CHIPQueue::memCopy(void *dst, const void *src, size_t size) {
#ifdef ENFORCE_QUEUE_SYNC
  chip_context->syncQueues(this);
#endif
  auto ev = memCopyImpl(dst, src, size);
  ev->msg = "memCopy";
  updateLastEvent(ev);
  return hipSuccess;
}
hipError_t CHIPQueue::memCopyAsync(void *dst, const void *src, size_t size) {
#ifdef ENFORCE_QUEUE_SYNC
  chip_context->syncQueues(this);
#endif
  auto ev = memCopyAsyncImpl(dst, src, size);
  updateLastEvent(ev);
  return hipSuccess;
}
void CHIPQueue::memFill(void *dst, size_t size, const void *pattern,
                        size_t pattern_size) {
#ifdef ENFORCE_QUEUE_SYNC
  chip_context->syncQueues(this);
#endif
  auto ev = memFillImpl(dst, size, pattern, pattern_size);
  updateLastEvent(ev);
}
CHIPEvent *CHIPQueue::memFillImpl(void *dst, size_t size, const void *pattern,
                                  size_t pattern_size) {
  auto ev = memFillAsyncImpl(dst, size, pattern, pattern_size);
  finish();
  return ev;
}
void CHIPQueue::memFillAsync(void *dst, size_t size, const void *pattern,
                             size_t pattern_size) {
#ifdef ENFORCE_QUEUE_SYNC
  chip_context->syncQueues(this);
#endif
  auto ev = memFillAsyncImpl(dst, size, pattern, pattern_size);
  updateLastEvent(ev);
}
void CHIPQueue::memCopy2D(void *dst, size_t dpitch, const void *src,
                          size_t spitch, size_t width, size_t height) {
#ifdef ENFORCE_QUEUE_SYNC
  chip_context->syncQueues(this);
#endif
  auto ev = memCopy2DAsyncImpl(dst, dpitch, src, spitch, width, height);
  finish();
  updateLastEvent(ev);
}
CHIPEvent *CHIPQueue::memCopy2DImpl(void *dst, size_t dpitch, const void *src,
                                    size_t spitch, size_t width,
                                    size_t height) {
#ifdef ENFORCE_QUEUE_SYNC
  chip_context->syncQueues(this);
#endif
  auto ev = memCopy2DAsyncImpl(dst, dpitch, src, spitch, width, height);
  finish();
  return ev;
}
void CHIPQueue::memCopy2DAsync(void *dst, size_t dpitch, const void *src,
                               size_t spitch, size_t width, size_t height) {
#ifdef ENFORCE_QUEUE_SYNC
  chip_context->syncQueues(this);
#endif
  auto ev = memCopy2DAsyncImpl(dst, dpitch, src, spitch, width, height);
  updateLastEvent(ev);
}
void CHIPQueue::memCopy3D(void *dst, size_t dpitch, size_t dspitch,
                          const void *src, size_t spitch, size_t sspitch,
                          size_t width, size_t height, size_t depth) {
#ifdef ENFORCE_QUEUE_SYNC
  chip_context->syncQueues(this);
#endif
  auto ev = memCopy3DAsyncImpl(dst, dpitch, dspitch, src, spitch, sspitch,
                               width, height, depth);
  finish();
  updateLastEvent(ev);
}
CHIPEvent *CHIPQueue::memCopy3DImpl(void *dst, size_t dpitch, size_t dspitch,
                                    const void *src, size_t spitch,
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
void CHIPQueue::memCopy3DAsync(void *dst, size_t dpitch, size_t dspitch,
                               const void *src, size_t spitch, size_t sspitch,
                               size_t width, size_t height, size_t depth) {
#ifdef ENFORCE_QUEUE_SYNC
  chip_context->syncQueues(this);
#endif
  auto ev = memCopy3DAsyncImpl(dst, dpitch, dspitch, src, spitch, sspitch,
                               width, height, depth);
  updateLastEvent(ev);
}
void CHIPQueue::memCopyToTexture(CHIPTexture *texObj, void *src) {
#ifdef ENFORCE_QUEUE_SYNC
  chip_context->syncQueues(this);
#endif
  auto ev = memCopyToTextureImpl(texObj, src);
  updateLastEvent(ev);
}
CHIPEvent *CHIPQueue::launch(CHIPExecItem *exec_item) {
#ifdef ENFORCE_QUEUE_SYNC
  chip_context->syncQueues(this);
#endif
  auto ev = launchImpl(exec_item);
  ev->msg = "launch";
  updateLastEvent(ev);
  return ev;
}
CHIPEvent *CHIPQueue::enqueueBarrier(
    std::vector<CHIPEvent *> *eventsToWaitFor) {
  auto ev = enqueueBarrierImpl(eventsToWaitFor);
  updateLastEvent(ev);
  return ev;
}
CHIPEvent *CHIPQueue::enqueueMarker() {
#ifdef ENFORCE_QUEUE_SYNC
  chip_context->syncQueues(this);
#endif
  auto ev = enqueueMarkerImpl();
  updateLastEvent(ev);
  return ev;
}

void CHIPQueue::memPrefetch(const void *ptr, size_t count) {
#ifdef ENFORCE_QUEUE_SYNC
  chip_context->syncQueues(this);
#endif
  auto ev = memPrefetchImpl(ptr, count);
  updateLastEvent(ev);
}

void CHIPQueue::launchHostFunc(const void *hostFunction, dim3 numBlocks,
                               dim3 dimBlocks, void **args,
                               size_t sharedMemBytes) {
  CHIPExecItem e(numBlocks, dimBlocks, sharedMemBytes,
                 Backend->getActiveQueue());
  e.setArgPointer(args);
  auto ev = e.launchByHostPtr(hostFunction);
  ev->msg = "launchHostFunc";
  updateLastEvent(ev);
}

void CHIPQueue::launchWithKernelParams(dim3 grid, dim3 block,
                                       unsigned int sharedMemBytes, void **args,
                                       CHIPKernel *kernel) {
  UNIMPLEMENTED();
}

void CHIPQueue::launchWithExtraParams(dim3 grid, dim3 block,
                                      unsigned int sharedMemBytes, void **extra,
                                      CHIPKernel *kernel) {
  UNIMPLEMENTED();
}

///////// End Enqueue Operations //////////

CHIPDevice *CHIPQueue::getDevice() {
  if (chip_device == nullptr) {
    std::string msg = "chip_device is null";
    CHIPERR_LOG_AND_THROW(msg, hipErrorLaunchFailure);
  }

  return chip_device;
}

unsigned int CHIPQueue::getFlags() { return flags; }
// hipError_t CHIPQueue::memCopy(void *dst, const void *src, size_t size) {}
// hipError_t CHIPQueue::memCopyAsync(void *, void const *, unsigned long) {}

int CHIPQueue::getPriorityRange(int lower_or_upper) { UNIMPLEMENTED(0); }
int CHIPQueue::getPriority() { UNIMPLEMENTED(0); }
bool CHIPQueue::addCallback(hipStreamCallback_t callback, void *userData) {
  CHIPCallbackData *cb = Backend->createCallbackData(callback, userData, this);

  Backend->callback_stack.push(cb);

  // Setup event handling on the CPU side
  if (!event_monitor) event_monitor = Backend->createEventMonitor();
  return true;
}

bool CHIPQueue::query() { UNIMPLEMENTED(true); }
