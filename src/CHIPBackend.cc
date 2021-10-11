#include "CHIPBackend.hh"
#include <utility>
// CHIPDeviceVar
// ************************************************************************
size_t CHIPDeviceVar::getSize() {}
std::string CHIPDeviceVar::getName() {}

// CHIPAllocationTracker
// ************************************************************************
CHIPAllocationTracker::CHIPAllocationTracker() {}
CHIPAllocationTracker::~CHIPAllocationTracker() {}

allocation_info *CHIPAllocationTracker::getByHostPtr(const void *host_ptr) {}
allocation_info *CHIPAllocationTracker::getByDevPtr(const void *dev_ptr) {}

// CHIPEvent
// ************************************************************************
CHIPEvent::CHIPEvent(CHIPContext *ctx_in, CHIPEventType event_type_)
    : status(EVENT_STATUS_INIT), flags(event_type_), chip_context(ctx_in) {}
CHIPEvent::~CHIPEvent() {}

bool CHIPEvent::recordStream(CHIPQueue *chip_queue_){};
bool CHIPEvent::wait(){};
bool CHIPEvent::isFinished(){};
float CHIPEvent::getElapsedTime(CHIPEvent *other){};

// CHIPModule
//*************************************************************************************
CHIPModule::CHIPModule(std::string *module_str) { src = *module_str; }
CHIPModule::CHIPModule(std::string &&module_str) { src = module_str; }
CHIPModule::~CHIPModule() {}

void CHIPModule::addKernel(CHIPKernel *kernel) {
  chip_kernels.push_back(kernel);
}

void CHIPModule::compileOnce(CHIPDevice *chip_dev) {
  std::call_once(compiled, &CHIPModule::compile, this, chip_dev);
}

// void CHIPModule::compile(CHIPDevice *chip_dev) {
//   logCritical(
//       "CHIPModule::compile() base implementation should never be called");
//   std::abort();
// }

CHIPKernel *CHIPModule::getKernel(std::string name) {
  auto kernel = std::find_if(
      chip_kernels.begin(), chip_kernels.end(),
      [name](CHIPKernel *k) { return k->getName().compare(name) == 0; });
  if (kernel == chip_kernels.end()) {
    logError("Failed to find kernel {} in module {}", name.c_str(),
             (void *)this);
    return nullptr;
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
    logError("Failed to find kernel with host pointer {} in module {}",
             host_f_ptr, (void *)this);
    return nullptr;
  }

  return *kernel;
}

std::vector<CHIPKernel *> &CHIPModule::getKernels() { return chip_kernels; }

CHIPDeviceVar *CHIPModule::getGlobalVar(std::string name) {
  auto var = std::find_if(
      chip_vars.begin(), chip_vars.end(),
      [name](CHIPDeviceVar *v) { return v->getName().compare(name) == 0; });
  if (var == chip_vars.end()) {
    logError("Failed to find global variable {} in module {}", name,
             (void *)this);
    return nullptr;
  }

  return *var;
}

// CHIPKernel
//*************************************************************************************
CHIPKernel::~CHIPKernel(){};
std::string CHIPKernel::getName() { return host_f_name; }
const void *CHIPKernel::getHostPtr() { return host_f_ptr; }
const void *CHIPKernel::getDevPtr() { return dev_f_ptr; }

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

void CHIPExecItem::setArg(const void *arg, size_t size, size_t offset) {
  if ((offset + size) > arg_data.size()) arg_data.resize(offset + size + 1024);

  std::memcpy(arg_data.data() + offset, arg, size);
  logDebug("CHIPExecItem.setArg() on {} size {} offset {}\n", (void *)this,
           size, offset);
  offset_sizes.push_back(std::make_tuple(offset, size));
}

hipError_t CHIPExecItem::launch(CHIPKernel *Kernel) {
  logWarn("Calling CHIPExecItem.launch() base launch which does nothing");
  return hipSuccess;
};

hipError_t CHIPExecItem::launchByHostPtr(const void *hostPtr) {
  if (chip_queue == nullptr) {
    logCritical(
        "CHIPExecItem.launchByHostPtr() was called but queue pointer is null");
    return (hipErrorLaunchFailure);
  }

  CHIPDevice *dev = chip_queue->getDevice();
  this->chip_kernel = dev->findKernelByHostPtr(hostPtr);
  logTrace("Found kernel for host pointer {} : {}", hostPtr,
           chip_kernel->getName());

  return (chip_queue->launch(this));
  // return launch(chip_kernel);
}

dim3 CHIPExecItem::getBlock() { return block_dim; }
dim3 CHIPExecItem::getGrid() { return grid_dim; }
CHIPKernel *CHIPExecItem::getKernel() { return chip_kernel; }
// CHIPDevice
//*************************************************************************************
CHIPDevice::CHIPDevice() {
  logDebug("Device {} is {}: name \"{}\" \n", idx, (void *)this,
           hip_device_props.name);
};
CHIPDevice::~CHIPDevice(){};

std::vector<CHIPKernel *> &CHIPDevice::getKernels() { return chip_kernels; };

void CHIPDevice::copyDeviceProperties(hipDeviceProp_t *prop) {
  logTrace("CHIPDevice->copy_device_properties()");
  if (prop) std::memcpy(prop, &this->hip_device_props, sizeof(hipDeviceProp_t));
}

CHIPKernel *CHIPDevice::findKernelByHostPtr(const void *hostPtr) {
  logTrace("CHIPDevice::findKernelByHostPtr({})", hostPtr);
  std::vector<CHIPKernel *> chip_kernels = getKernels();
  logDebug("Listing Kernels for device {}", device_name);
  for (auto &kernel : chip_kernels) {
    logDebug("{}", kernel->getName());
  }

  auto found_kernel = std::find_if(chip_kernels.begin(), chip_kernels.end(),
                                   [&hostPtr](CHIPKernel *kernel) {
                                     return kernel->getHostPtr() == hostPtr;
                                   });

  if (found_kernel == chip_kernels.end()) {
    logCritical("Failed to find kernel {} on device #{}:{}", hostPtr, idx,
                device_name);
    std::abort();  // Exception
  } else {
    logDebug("Found kernel {} with host pointer {}", (*found_kernel)->getName(),
             (*found_kernel)->getHostPtr());
  }

  return *found_kernel;
}

CHIPContext *CHIPDevice::getContext() { return ctx; }
int CHIPDevice::getDeviceId() { return idx; }

// TODO CHIP Design Choice - should this be even called that?
// bool CHIPDevice::allocate(size_t bytes) {
//   logTrace("CHIPDevice->reserve_mem()");
//   std::lock_guard<std::mutex> Lock(mtx);
//   if (bytes <= (hip_device_props.totalGlobalMem - total_used_mem)) {
//     total_used_mem += bytes;
//     if (total_used_mem > max_used_mem) max_used_mem = total_used_mem;
//     logDebug("Currently used memory on dev {}: {} M\n", idx,
//              (total_used_mem >> 20));
//     return true;
//   } else {
//     logError(
//         "Can't allocate {} bytes of memory on device # {}\n. "
//         "GlobalMemSize:{} TotalUsedMem: {}",
//         bytes, idx, hip_device_props.totalGlobalMem, total_used_mem);
//     return false;
//   }
// }

// bool CHIPDevice::free(size_t bytes) {
//   std::lock_guard<std::mutex> Lock(mtx);
//   if (total_used_mem >= bytes) {
//     total_used_mem -= bytes;
//     return true;
//   } else {
//     return false;
//   }
// }

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
      // TODO - These are undefined
      // case hipDeviceAttributeCooperativeLaunch:
      //   *pi = prop.cooperativeLaunch;
      //   break;
      // case hipDeviceAttributeCooperativeMultiDeviceLaunch:
      //   *pi = prop.cooperativeMultiDeviceLaunch;
      //   break;
      // case hipDeviceAttributeIntegrated:
      //   *pi = prop.integrated;
      //   break;
      // case hipDeviceAttributeMaxTexture1DWidth:
      //   *pi = prop.maxTexture1D;
      //   break;
      // case hipDeviceAttributeMaxTexture2DWidth:
      //   *pi = prop.maxTexture2D[0];
      //   break;
      // case hipDeviceAttributeMaxTexture2DHeight:
      //   *pi = prop.maxTexture2D[1];
      //   break;
      // case hipDeviceAttributeMaxTexture3DWidth:
      //   *pi = prop.maxTexture3D[0];
      //   break;
      // case hipDeviceAttributeMaxTexture3DHeight:
      //   *pi = prop.maxTexture3D[1];
      //   break;
      // case hipDeviceAttributeMaxTexture3DDepth:
      //   *pi = prop.maxTexture3D[2];
      //   break;
      // case hipDeviceAttributeHdpMemFlushCntl:
      //   *reinterpret_cast<unsigned int **>(pi) = prop.hdpMemFlushCntl;
      //   break;
      // case hipDeviceAttributeHdpRegFlushCntl:
      //   *reinterpret_cast<unsigned int **>(pi) = prop.hdpRegFlushCntl;
      //   break;
      // case hipDeviceAttributeMaxPitch:
      //   *pi = prop.memPitch;
      //   break;
      // case hipDeviceAttributeTextureAlignment:
      //   *pi = prop.textureAlignment;
      //   break;
      // case hipDeviceAttributeTexturePitchAlignment:
      //   *pi = prop.texturePitchAlignment;
      //   break;
      // case hipDeviceAttributeKernelExecTimeout:
      //   *pi = prop.kernelExecTimeoutEnabled;
      //   break;
      // case hipDeviceAttributeCanMapHostMemory:
      //   *pi = prop.canMapHostMemory;
      //   break;
      // case hipDeviceAttributeEccEnabled:
      //   *pi = prop.ECCEnabled;
      //   break;
      // case hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc:
      //   *pi = prop.cooperativeMultiDeviceUnmatchedFunc;
      //   break;
      // case hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim:
      //   *pi = prop.cooperativeMultiDeviceUnmatchedGridDim;
      //   break;
      // case hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim:
      //   *pi = prop.cooperativeMultiDeviceUnmatchedBlockDim;
      //   break;
      // case hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem:
      //   *pi = prop.cooperativeMultiDeviceUnmatchedSharedMem;
      //   break;
      // case hipDeviceAttributeAsicRevision:
      //   *pi = prop.asicRevision;
      //   break;
      // case hipDeviceAttributeManagedMemory:
      //   *pi = prop.managedMemory;
      //   break;
      // case hipDeviceAttributeDirectManagedMemAccessFromHost:
      //   *pi = prop.directManagedMemAccessFromHost;
      //   break;
      // case hipDeviceAttributeConcurrentManagedAccess:
      //   *pi = prop.concurrentManagedAccess;
      //   break;
      // case hipDeviceAttributePageableMemoryAccess:
      //   *pi = prop.pageableMemoryAccess;
      //   break;
      // case hipDeviceAttributePageableMemoryAccessUsesHostPageTables:
      //   *pi = prop.pageableMemoryAccessUsesHostPageTables;
      //   break;
      // case hipDeviceAttributeCanUseStreamWaitValue:
      //   // hipStreamWaitValue64() and hipStreamWaitValue32() support
      //   *pi = g_devices[device]->devices()[0]->info().aqlBarrierValue_;
      break;
    default:
      // HIP_RETURN(hipErrorInvalidValue);
      return -1;
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
    // chip_module =
    //     new CHIPModule(module_str);  // Create a new module for this source
    chip_module = addModule(module_str);
    chip_module->compileOnce(this);  // Compile it
    host_f_ptr_to_chipmodule_map[module_str] = chip_module;
    // TODO Place it in the Backend cache
  }
  CHIPKernel *kernel = chip_module->getKernel(std::string(host_f_name));
  if (!kernel) {
    logCritical(
        "Device {}: tried to register host function <{}, {}> but failed to "
        "find kernel with matching name",
        getDeviceId(), host_f_ptr, host_f_name);
    std::abort();
  }

  kernel->setHostPtr(host_f_ptr);
  // assert(kernel->getDevPtr() != nullptr);

  chip_kernels.push_back(kernel);
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

bool CHIPDevice::reserveMem(size_t bytes) {
  std::lock_guard<std::mutex> Lock(mtx);
  if (bytes <= (getGlobalMemSize() - TotalUsedMem)) {
    TotalUsedMem += bytes;
    if (TotalUsedMem > MaxUsedMem) MaxUsedMem = TotalUsedMem;
    logDebug("Currently used memory on dev {}: {} M\n", getName().c_str(),
             (TotalUsedMem >> 20));
    return true;
  } else {
    logError("Can't allocate {} bytes of memory\n", bytes);
    return false;
  }
}

hipError_t CHIPDevice::setPeerAccess(CHIPDevice *peer, int flags,
                                     bool canAccessPeer) {}

int CHIPDevice::getPeerAccess(CHIPDevice *peerDevice) {}

void CHIPDevice::setCacheConfig(hipFuncCache_t cfg) {}

void CHIPDevice::setFuncCacheConfig(const void *func, hipFuncCache_t config) {}

hipFuncCache_t CHIPDevice::getCacheConfig() {}

hipSharedMemConfig CHIPDevice::getSharedMemConfig() {}

bool CHIPDevice::removeQueue(CHIPQueue *q) {}

void CHIPDevice::setSharedMemConfig(hipSharedMemConfig config) {}

size_t CHIPDevice::getUsedGlobalMem() {}

bool CHIPDevice::hasPCIBusId(int, int, int) {}

bool CHIPDevice::releaseMemReservation(unsigned long bytes) {
  std::lock_guard<std::mutex> Lock(mtx);
  if (TotalUsedMem >= bytes) {
    TotalUsedMem -= bytes;
    return true;
  } else {
    return false;
  }
}

CHIPQueue *CHIPDevice::getActiveQueue() { return chip_queues[0]; }
// CHIPContext
//*************************************************************************************
CHIPContext::CHIPContext() {}
CHIPContext::~CHIPContext() {}
bool CHIPContext::addDevice(CHIPDevice *dev) {
  logTrace("CHIPContext.add_device() {}", dev->getName());
  chip_devices.push_back(dev);
  // TODO check for success
  return true;
}

std::vector<CHIPDevice *> &CHIPContext::getDevices() {
  if (chip_devices.size() == 0)
    logWarn("CHIPContext.get_devices() was called but chip_devices is empty");
  return chip_devices;
}

std::vector<CHIPQueue *> &CHIPContext::getQueues() {
  if (chip_queues.size() == 0) {
    logCritical(
        "CHIPContext.get_queues() was called but no queues were added to "
        "this context");
    std::abort();
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

  if (!chip_dev->reserveMem(size)) return nullptr;
  retval = allocate_(size, alignment, mem_type);
  if (retval == nullptr) chip_dev->releaseMemReservation(size);

  return retval;
}

hipError_t CHIPContext::findPointerInfo(hipDeviceptr_t *pbase, size_t *psize,
                                        hipDeviceptr_t dptr) {
  allocation_info *info = Backend->AllocationTracker.getByDevPtr(dptr);
  if (!info) return hipErrorInvalidDevicePointer;
  *pbase = info->base_ptr;
  *psize = info->size;
  return hipSuccess;
}

unsigned int CHIPContext::getFlags() {}

void CHIPContext::setFlags(unsigned int flags) {}

void CHIPContext::reset() {}

CHIPContext *CHIPContext::retain() {}

hipError_t CHIPContext::free(void *ptr) {
  CHIPDevice *chip_dev = Backend->getActiveDevice();
  allocation_info *info = Backend->AllocationTracker.getByDevPtr(ptr);
  if (!info) return hipErrorInvalidDevicePointer;

  chip_dev->releaseMemReservation(info->size);
  free_(ptr);
  return hipSuccess;
}

void CHIPContext::recordEvent(CHIPQueue *q, CHIPEvent *event) {}

CHIPTexture *CHIPContext::createImage(hipResourceDesc *resDesc,
                                      hipTextureDesc *texDesc) {}

// CHIPBackend
//*************************************************************************************

CHIPBackend::CHIPBackend() { logDebug("CHIPBackend Base Constructor"); };
CHIPBackend::~CHIPBackend(){};

void CHIPBackend::initialize(std::string platform_str,
                             std::string device_type_str,
                             std::string device_ids_str) {
  initialize_(platform_str, device_type_str, device_ids_str);
  if (chip_devices.size() == 0) {
    logCritical("No CHIPDevices were initialized.");
    std::abort();
  }
  setActiveDevice(chip_devices[0]);
}

void CHIPBackend::setActiveDevice(CHIPDevice *chip_dev) {
  auto I = std::find(chip_devices.begin(), chip_devices.end(), chip_dev);
  if (I == chip_devices.end()) {
    logCritical(
        "Tried to set active device with CHIPDevice pointer that is not in "
        "CHIPBackend::chip_devices");
    std::abort();
  };
  active_dev = chip_dev;
  active_ctx = chip_dev->getContext();
  active_q = chip_dev->getActiveQueue();
}
std::vector<CHIPQueue *> &CHIPBackend::getQueues() { return chip_queues; }
CHIPQueue *CHIPBackend::getActiveQueue() {
  if (active_q == nullptr) {
    logCritical(
        "CHIPBackend.getActiveQueue() was called but no queues have "
        "been initialized;\n");
    std::abort();
  }
  return active_q;
};

CHIPContext *CHIPBackend::getActiveContext() {
  if (active_ctx == nullptr) {
    logCritical(
        "CHIPBackend.getActiveContext() was called but active_ctx is null");
    std::abort();
  }
  return active_ctx;
};

CHIPDevice *CHIPBackend::getActiveDevice() {
  if (active_dev == nullptr) {
    logCritical(
        "CHIPBackend.getActiveDevice() was called but active_ctx is null");
    std::abort();
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
    logCritical(
        "CHIPQueue.getDevice() was called but device is a null pointer");
    std::abort();  // TODO Exception?
  }

  return chip_device;
}

unsigned int CHIPQueue::getFlags() {}
hipError_t CHIPQueue::launch(CHIPExecItem *) {}
hipError_t CHIPQueue::memCopy(void *dst, const void *src, size_t size) {}
hipError_t CHIPQueue::memCopyAsync(void *, void const *, unsigned long) {}

hipError_t CHIPQueue::launchWithKernelParams(dim3 grid, dim3 block,
                                             unsigned int sharedMemBytes,
                                             void **args, CHIPKernel *kernel) {}

hipError_t CHIPQueue::launchWithExtraParams(dim3 grid, dim3 block,
                                            unsigned int sharedMemBytes,
                                            void **extra, CHIPKernel *kernel) {}

int CHIPQueue::getPriorityRange(int lower_or_upper) {}
int CHIPQueue::getPriority() {}
void CHIPQueue::finish() {}
bool CHIPQueue::addCallback(hipStreamCallback_t callback, void *userData) {}
bool CHIPQueue::launchHostFunc(const void *hostFunction, dim3 numBlocks,
                               dim3 dimBlocks, void **args,
                               size_t sharedMemBytes) {}
bool CHIPQueue::enqueueBarrierForEvent(CHIPEvent *e) {}
bool CHIPQueue::query() {}
void CHIPQueue::memFill(void *dst, size_t size, const void *pattern,
                        size_t pattern_size) {}
void CHIPQueue::memFillAsync(void *dst, size_t size, const void *pattern,
                             size_t pattern_size) {}

bool CHIPQueue::memPrefetch(const void *ptr, size_t count) {}
