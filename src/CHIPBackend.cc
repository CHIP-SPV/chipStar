#include "CHIPBackend.hh"
#include <utility>
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

void CHIPAllocationTracker::recordAllocation(void *ptr) {
  ptr_set.insert(ptr);
  return;
}

// CHIPEvent
// ************************************************************************
CHIPEvent::CHIPEvent(CHIPContext *ctx_in, CHIPEventType event_type_)
    : status(EVENT_STATUS_INIT), flags(event_type_), chip_context(ctx_in) {}
CHIPEvent::~CHIPEvent() {}

bool CHIPEvent::recordStream(CHIPQueue *chip_queue_) { UNIMPLEMENTED(true); };
bool CHIPEvent::wait() { UNIMPLEMENTED(true); };
bool CHIPEvent::isFinished() { UNIMPLEMENTED(true); };
float CHIPEvent::getElapsedTime(CHIPEvent *other) { UNIMPLEMENTED(true); };

// CHIPModule
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

CHIPDeviceVar *CHIPModule::getGlobalVar(std::string name) {
  auto var = std::find_if(
      chip_vars.begin(), chip_vars.end(),
      [name](CHIPDeviceVar *v) { return v->getName().compare(name) == 0; });
  if (var == chip_vars.end()) {
    std::string msg = "Failed to find global variable by name: " + name;
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

hipError_t CHIPExecItem::launchByHostPtr(const void *hostPtr) {
  if (chip_queue == nullptr) {
    std::string msg = "Tried to launch CHIPExecItem but its queue is null";
    CHIPERR_LOG_AND_THROW(msg, hipErrorLaunchFailure);
  }

  CHIPDevice *dev = chip_queue->getDevice();
  this->chip_kernel = dev->findKernelByHostPtr(hostPtr);
  return (chip_queue->launch(this));
}

dim3 CHIPExecItem::getBlock() { return block_dim; }
dim3 CHIPExecItem::getGrid() { return grid_dim; }
CHIPKernel *CHIPExecItem::getKernel() { return chip_kernel; }
size_t CHIPExecItem::getSharedMem() { return shared_mem; }
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
    logDebug("Kernel name: {} host_f_ptr: {}", kernel->getName(),
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

CHIPDeviceVar *CHIPDevice::getDynGlobalVar(const void *host_var_ptr) {
  auto found_dyn = host_var_ptr_to_chipdevicevar_dyn.find(host_var_ptr);
  if (found_dyn != host_var_ptr_to_chipdevicevar_dyn.end())
    return found_dyn->second;

  return nullptr;
}

CHIPDeviceVar *CHIPDevice::getStatGlobalVar(const void *host_var_ptr) {
  auto found_stat = host_var_ptr_to_chipdevicevar_stat.find(host_var_ptr);
  if (found_stat != host_var_ptr_to_chipdevicevar_stat.end())
    return found_stat->second;

  return nullptr;
}

CHIPDeviceVar *CHIPDevice::getGlobalVar(const void *host_var_ptr) {
  auto found_dyn = getDynGlobalVar(host_var_ptr);
  if (found_dyn) return found_dyn;

  auto found_stat = getStatGlobalVar(host_var_ptr);
  if (found_stat) return found_stat;

  return nullptr;
}

int CHIPDevice::getAttr(hipDeviceAttribute_t attr) {
  int *pi;
  hipDeviceProp_t prop = {0};
  copyDeviceProperties(&prop);

  switch (attr) {
    case hipDeviceAttributeMaxThreadsPerBlock:
      *pi = prop.maxThreadsPerBlock;
      break;
    case hipDeviceAttributeMaxBlockDimX:
      *pi = prop.maxThreadsDim[0];
      break;
    case hipDeviceAttributeMaxBlockDimY:
      *pi = prop.maxThreadsDim[1];
      break;
    case hipDeviceAttributeMaxBlockDimZ:
      *pi = prop.maxThreadsDim[2];
      break;
    case hipDeviceAttributeMaxGridDimX:
      *pi = prop.maxGridSize[0];
      break;
    case hipDeviceAttributeMaxGridDimY:
      *pi = prop.maxGridSize[1];
      break;
    case hipDeviceAttributeMaxGridDimZ:
      *pi = prop.maxGridSize[2];
      break;
    case hipDeviceAttributeMaxSharedMemoryPerBlock:
      *pi = prop.sharedMemPerBlock;
      break;
    case hipDeviceAttributeTotalConstantMemory:
      *pi = prop.totalConstMem;
      break;
    case hipDeviceAttributeWarpSize:
      *pi = prop.warpSize;
      break;
    case hipDeviceAttributeMaxRegistersPerBlock:
      *pi = prop.regsPerBlock;
      break;
    case hipDeviceAttributeClockRate:
      *pi = prop.clockRate;
      break;
    case hipDeviceAttributeMemoryClockRate:
      *pi = prop.memoryClockRate;
      break;
    case hipDeviceAttributeMemoryBusWidth:
      *pi = prop.memoryBusWidth;
      break;
    case hipDeviceAttributeMultiprocessorCount:
      *pi = prop.multiProcessorCount;
      break;
    case hipDeviceAttributeComputeMode:
      *pi = prop.computeMode;
      break;
    case hipDeviceAttributeL2CacheSize:
      *pi = prop.l2CacheSize;
      break;
    case hipDeviceAttributeMaxThreadsPerMultiProcessor:
      *pi = prop.maxThreadsPerMultiProcessor;
      break;
    case hipDeviceAttributeComputeCapabilityMajor:
      *pi = prop.major;
      break;
    case hipDeviceAttributeComputeCapabilityMinor:
      *pi = prop.minor;
      break;
    case hipDeviceAttributePciBusId:
      *pi = prop.pciBusID;
      break;
    case hipDeviceAttributeConcurrentKernels:
      *pi = prop.concurrentKernels;
      break;
    case hipDeviceAttributePciDeviceId:
      *pi = prop.pciDeviceID;
      break;
    case hipDeviceAttributeMaxSharedMemoryPerMultiprocessor:
      *pi = prop.maxSharedMemoryPerMultiProcessor;
      break;
    case hipDeviceAttributeIsMultiGpuBoard:
      *pi = prop.isMultiGpuBoard;
      break;
    case hipDeviceAttributeCooperativeLaunch:
      *pi = prop.cooperativeLaunch;
      break;
    case hipDeviceAttributeCooperativeMultiDeviceLaunch:
      *pi = prop.cooperativeMultiDeviceLaunch;
      break;
    case hipDeviceAttributeIntegrated:
      *pi = prop.integrated;
      break;
    case hipDeviceAttributeMaxTexture1DWidth:
      *pi = prop.maxTexture1D;
      break;
    case hipDeviceAttributeMaxTexture2DWidth:
      *pi = prop.maxTexture2D[0];
      break;
    case hipDeviceAttributeMaxTexture2DHeight:
      *pi = prop.maxTexture2D[1];
      break;
    case hipDeviceAttributeMaxTexture3DWidth:
      *pi = prop.maxTexture3D[0];
      break;
    case hipDeviceAttributeMaxTexture3DHeight:
      *pi = prop.maxTexture3D[1];
      break;
    case hipDeviceAttributeMaxTexture3DDepth:
      *pi = prop.maxTexture3D[2];
      break;
    case hipDeviceAttributeHdpMemFlushCntl:
      *reinterpret_cast<unsigned int **>(pi) = prop.hdpMemFlushCntl;
      break;
    case hipDeviceAttributeHdpRegFlushCntl:
      *reinterpret_cast<unsigned int **>(pi) = prop.hdpRegFlushCntl;
      break;
    case hipDeviceAttributeMaxPitch:
      *pi = prop.memPitch;
      break;
    case hipDeviceAttributeTextureAlignment:
      *pi = prop.textureAlignment;
      break;
    case hipDeviceAttributeTexturePitchAlignment:
      *pi = prop.texturePitchAlignment;
      break;
    case hipDeviceAttributeKernelExecTimeout:
      *pi = prop.kernelExecTimeoutEnabled;
      break;
    case hipDeviceAttributeCanMapHostMemory:
      *pi = prop.canMapHostMemory;
      break;
    case hipDeviceAttributeEccEnabled:
      *pi = prop.ECCEnabled;
      break;
    case hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc:
      *pi = prop.cooperativeMultiDeviceUnmatchedFunc;
      break;
    case hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim:
      *pi = prop.cooperativeMultiDeviceUnmatchedGridDim;
      break;
    case hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim:
      *pi = prop.cooperativeMultiDeviceUnmatchedBlockDim;
      break;
    case hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem:
      *pi = prop.cooperativeMultiDeviceUnmatchedSharedMem;
      break;
    case hipDeviceAttributeAsicRevision:
      *pi = prop.asicRevision;
      break;
    case hipDeviceAttributeManagedMemory:
      *pi = prop.managedMemory;
      break;
    case hipDeviceAttributeDirectManagedMemAccessFromHost:
      *pi = prop.directManagedMemAccessFromHost;
      break;
    case hipDeviceAttributeConcurrentManagedAccess:
      *pi = prop.concurrentManagedAccess;
      break;
    case hipDeviceAttributePageableMemoryAccess:
      *pi = prop.pageableMemoryAccess;
      break;
    case hipDeviceAttributePageableMemoryAccessUsesHostPageTables:
      *pi = prop.pageableMemoryAccessUsesHostPageTables;
      break;
    case hipDeviceAttributeCanUseStreamWaitValue:
      // hipStreamWaitValue64() and hipStreamWaitValue32() support
      //*pi = g_devices[device]->devices()[0]->info().aqlBarrierValue_;
      CHIPERR_LOG_AND_THROW(
          "CHIPDevice::getAttr(hipDeviceAttributeCanUseStreamWaitValue path "
          "unimplemented",
          hipErrorTbd);
      break;
    default:
      CHIPERR_LOG_AND_THROW("CHIPDevice::getAttr asked for an unkown attribute",
                            hipErrorInvalidValue);
  }
  return *pi;
}

size_t CHIPDevice::getGlobalMemSize() {
  return hip_device_props.totalGlobalMem;
}

void CHIPDevice::registerFunctionAsKernel(std::string *module_str,
                                          const void *host_f_ptr,
                                          const char *host_f_name) {
  CHIPModule *chip_module;
  auto chip_module_found = host_f_ptr_to_chipmodule_map.find(host_f_ptr);
  if (chip_module_found != host_f_ptr_to_chipmodule_map.end()) {
    chip_module = chip_module_found->second;
  } else {
    chip_module = addModule(module_str);
    chip_module->compileOnce(this);  // Compile it
    host_f_ptr_to_chipmodule_map[module_str] = chip_module;
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
  void *retval;

  CHIPDevice *chip_dev = Backend->getActiveDevice();
  assert(chip_dev->getContext() == this);

  if (!chip_dev->allocation_tracker->reserveMem(size)) return nullptr;
  retval = allocate_(size, alignment, mem_type);
  if (retval == nullptr)
    chip_dev->allocation_tracker->releaseMemReservation(size);

  return retval;
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

unsigned int CHIPContext::getFlags() { UNIMPLEMENTED(0); }

void CHIPContext::setFlags(unsigned int flags) { UNIMPLEMENTED(); }

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

void CHIPContext::recordEvent(CHIPQueue *q, CHIPEvent *event) {
  UNIMPLEMENTED();
}

CHIPTexture *CHIPContext::createImage(hipResourceDesc *resDesc,
                                      hipTextureDesc *texDesc) {
  UNIMPLEMENTED(nullptr);
}

// CHIPBackend
//*************************************************************************************

CHIPBackend::CHIPBackend() { logDebug("CHIPBackend Base Constructor"); };
CHIPBackend::~CHIPBackend(){};

void CHIPBackend::initialize(std::string platform_str,
                             std::string device_type_str,
                             std::string device_ids_str) {
  initialize_(platform_str, device_type_str, device_ids_str);
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
    CHIPERR_LOG_AND_THROW(msg, hipErrorLaunchFailure);
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
  logTrace("CHIPBackend->configureCall()");
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
}

CHIPQueue *CHIPBackend::findQueue(CHIPQueue *q) {
  auto q_found = std::find(chip_queues.begin(), chip_queues.end(), q);
  if (q_found == chip_queues.end()) return nullptr;
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
};
CHIPQueue::CHIPQueue(CHIPDevice *chip_device_, unsigned int flags_)
    : CHIPQueue(chip_device_, flags_, 0){};
CHIPQueue::CHIPQueue(CHIPDevice *chip_device_)
    : CHIPQueue(chip_device_, 0, 0){};
CHIPQueue::~CHIPQueue(){};

CHIPDevice *CHIPQueue::getDevice() {
  if (chip_device == nullptr) {
    std::string msg = "chip_device is null";
    CHIPERR_LOG_AND_THROW(msg, hipErrorLaunchFailure);
  }

  return chip_device;
}

unsigned int CHIPQueue::getFlags() { UNIMPLEMENTED(0); }
hipError_t CHIPQueue::launch(CHIPExecItem *) { UNIMPLEMENTED(hipSuccess); }
// hipError_t CHIPQueue::memCopy(void *dst, const void *src, size_t size) {}
// hipError_t CHIPQueue::memCopyAsync(void *, void const *, unsigned long) {}

hipError_t CHIPQueue::launchWithKernelParams(dim3 grid, dim3 block,
                                             unsigned int sharedMemBytes,
                                             void **args, CHIPKernel *kernel) {
  UNIMPLEMENTED(hipSuccess);
}

hipError_t CHIPQueue::launchWithExtraParams(dim3 grid, dim3 block,
                                            unsigned int sharedMemBytes,
                                            void **extra, CHIPKernel *kernel) {
  UNIMPLEMENTED(hipSuccess);
}

int CHIPQueue::getPriorityRange(int lower_or_upper) { UNIMPLEMENTED(0); }
int CHIPQueue::getPriority() { UNIMPLEMENTED(0); }
bool CHIPQueue::addCallback(hipStreamCallback_t callback, void *userData) {
  UNIMPLEMENTED(true);
}
bool CHIPQueue::launchHostFunc(const void *hostFunction, dim3 numBlocks,
                               dim3 dimBlocks, void **args,
                               size_t sharedMemBytes) {
  dim3 dimGrid = {dimBlocks.x * numBlocks.x, dimBlocks.y * numBlocks.y,
                  dimBlocks.z * numBlocks.z};
  CHIPExecItem e(numBlocks, dimBlocks, sharedMemBytes,
                 Backend->getActiveQueue());
  e.setArgPointer(args);
  e.launchByHostPtr(hostFunction);
}
bool CHIPQueue::enqueueBarrierForEvent(CHIPEvent *e) { UNIMPLEMENTED(true); }
bool CHIPQueue::query() { UNIMPLEMENTED(true); }
void CHIPQueue::memFill(void *dst, size_t size, const void *pattern,
                        size_t pattern_size) {
  UNIMPLEMENTED();
}
void CHIPQueue::memFillAsync(void *dst, size_t size, const void *pattern,
                             size_t pattern_size) {
  UNIMPLEMENTED();
}

bool CHIPQueue::memPrefetch(const void *ptr, size_t count) {
  UNIMPLEMENTED(true);
}
