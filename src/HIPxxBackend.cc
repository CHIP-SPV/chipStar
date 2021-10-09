#include "HIPxxBackend.hh"
#include <utility>
// HIPxxEvent
// ************************************************************************
HIPxxEvent::HIPxxEvent(HIPxxContext *ctx_in, HIPxxEventType event_type_)
    : status(EVENT_STATUS_INIT), flags(event_type_), hipxx_context(ctx_in) {}
HIPxxEvent::~HIPxxEvent() {}

bool HIPxxEvent::recordStream(HIPxxQueue *hipxx_queue_){};
bool HIPxxEvent::wait(){};
bool HIPxxEvent::isFinished(){};
float HIPxxEvent::getElapsedTime(HIPxxEvent *other){};

// HIPxxModule
//*************************************************************************************
HIPxxModule::HIPxxModule(std::string *module_str) { src = *module_str; }
HIPxxModule::HIPxxModule(std::string &&module_str) { src = module_str; }
HIPxxModule::~HIPxxModule() {}

void HIPxxModule::addKernel(HIPxxKernel *kernel) {
  hipxx_kernels.push_back(kernel);
}

void HIPxxModule::compileOnce(HIPxxDevice *hipxx_dev) {
  std::call_once(compiled, &HIPxxModule::compile, this, hipxx_dev);
}

void HIPxxModule::compile(HIPxxDevice *hipxx_dev) {
  logCritical(
      "HIPxxModule::compile() base implementation should never be called");
  std::abort();
}

HIPxxKernel *HIPxxModule::getKernel(std::string name) {
  auto kernel = std::find_if(
      hipxx_kernels.begin(), hipxx_kernels.end(),
      [name](HIPxxKernel *k) { return k->getName().compare(name) == 0; });
  if (kernel == hipxx_kernels.end()) {
    logError("Failed to find kernel {} in module {}", name.c_str(),
             (void *)this);
    return nullptr;
  }

  return *kernel;
}

HIPxxKernel *HIPxxModule::getKernel(const void *host_f_ptr) {
  auto kernel = std::find_if(
      hipxx_kernels.begin(), hipxx_kernels.end(),
      [host_f_ptr](HIPxxKernel *k) { return k->getHostPtr() == host_f_ptr; });
  if (kernel == hipxx_kernels.end()) {
    logError("Failed to find kernel with host pointer {} in module {}",
             host_f_ptr, (void *)this);
    return nullptr;
  }

  return *kernel;
}

std::vector<HIPxxKernel *> &HIPxxModule::getKernels() { return hipxx_kernels; }

HIPxxDeviceVar *HIPxxModule::getGlobalVar(std::string name) {
  auto var = std::find_if(
      hipxx_vars.begin(), hipxx_vars.end(),
      [name](HIPxxDeviceVar *v) { return v->getName().compare(name) == 0; });
  if (var == hipxx_vars.end()) {
    logError("Failed to find global variable {} in module {}", name,
             (void *)this);
    return nullptr;
  }

  return *var;
}

// HIPxxKernel
//*************************************************************************************
HIPxxKernel::~HIPxxKernel(){};
std::string HIPxxKernel::getName() { return host_f_name; }
const void *HIPxxKernel::getHostPtr() { return host_f_ptr; }
const void *HIPxxKernel::getDevPtr() { return dev_f_ptr; }

void HIPxxKernel::setName(std::string host_f_name_) {
  host_f_name = host_f_name_;
}
void HIPxxKernel::setHostPtr(const void *host_f_ptr_) {
  host_f_ptr = host_f_ptr_;
}
void HIPxxKernel::setDevPtr(const void *dev_f_ptr_) { dev_f_ptr = dev_f_ptr_; }

// HIPxxExecItem
//*************************************************************************************
HIPxxExecItem::HIPxxExecItem(dim3 grid_dim_, dim3 block_dim_,
                             size_t shared_mem_, hipStream_t hipxx_queue_)
    : grid_dim(grid_dim_),
      block_dim(block_dim_),
      shared_mem(shared_mem_),
      hipxx_queue(hipxx_queue_){};
HIPxxExecItem::~HIPxxExecItem(){};

void HIPxxExecItem::setArg(const void *arg, size_t size, size_t offset) {
  if ((offset + size) > arg_data.size()) arg_data.resize(offset + size + 1024);

  std::memcpy(arg_data.data() + offset, arg, size);
  logDebug("HIPxxExecItem.setArg() on {} size {} offset {}\n", (void *)this,
           size, offset);
  offset_sizes.push_back(std::make_tuple(offset, size));
}

hipError_t HIPxxExecItem::launch(HIPxxKernel *Kernel) {
  logWarn("Calling HIPxxExecItem.launch() base launch which does nothing");
  return hipSuccess;
};

hipError_t HIPxxExecItem::launchByHostPtr(const void *hostPtr) {
  if (hipxx_queue == nullptr) {
    logCritical(
        "HIPxxExecItem.launchByHostPtr() was called but queue pointer is null");
    return (hipErrorLaunchFailure);
  }

  HIPxxDevice *dev = hipxx_queue->getDevice();
  this->hipxx_kernel = dev->findKernelByHostPtr(hostPtr);
  logTrace("Found kernel for host pointer {} : {}", hostPtr,
           hipxx_kernel->getName());
  return launch(hipxx_kernel);
}

// HIPxxDevice
//*************************************************************************************
HIPxxDevice::HIPxxDevice() {
  logDebug("Device {} is {}: name \"{}\" \n", idx, (void *)this,
           hip_device_props.name);
};
HIPxxDevice::~HIPxxDevice(){};

std::vector<HIPxxKernel *> &HIPxxDevice::getKernels() { return hipxx_kernels; };

void HIPxxDevice::copyDeviceProperties(hipDeviceProp_t *prop) {
  logTrace("HIPxxDevice->copy_device_properties()");
  if (prop) std::memcpy(prop, &this->hip_device_props, sizeof(hipDeviceProp_t));
}

HIPxxKernel *HIPxxDevice::findKernelByHostPtr(const void *hostPtr) {
  logTrace("HIPxxDevice::findKernelByHostPtr({})", hostPtr);
  std::vector<HIPxxKernel *> hipxx_kernels = getKernels();
  logDebug("Listing Kernels for device {}", device_name);
  for (auto &kernel : hipxx_kernels) {
    logDebug("{}", kernel->getName());
  }

  auto found_kernel = std::find_if(hipxx_kernels.begin(), hipxx_kernels.end(),
                                   [&hostPtr](HIPxxKernel *kernel) {
                                     return kernel->getHostPtr() == hostPtr;
                                   });

  if (found_kernel == hipxx_kernels.end()) {
    logCritical("Failed to find kernel {} on device #{}:{}", hostPtr, idx,
                device_name);
    std::abort();  // Exception
  } else {
    logDebug("Found kernel {} with host pointer {}", (*found_kernel)->getName(),
             (*found_kernel)->getHostPtr());
  }

  return *found_kernel;
}

HIPxxContext *HIPxxDevice::getContext() { return ctx; }
int HIPxxDevice::getDeviceId() { return idx; }

// TODO HIPxx Design Choice - should this be even called that?
// bool HIPxxDevice::allocate(size_t bytes) {
//   logTrace("HIPxxDevice->reserve_mem()");
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

// bool HIPxxDevice::free(size_t bytes) {
//   std::lock_guard<std::mutex> Lock(mtx);
//   if (total_used_mem >= bytes) {
//     total_used_mem -= bytes;
//     return true;
//   } else {
//     return false;
//   }
// }

int HIPxxDevice::getAttr(hipDeviceAttribute_t attr) {
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

void HIPxxDevice::registerFunctionAsKernel(std::string *module_str,
                                           const void *host_f_ptr,
                                           const char *host_f_name) {
  HIPxxModule *hipxx_module;
  auto hipxx_module_found = host_f_ptr_to_hipxxmodule_map.find(host_f_ptr);
  if (hipxx_module_found != host_f_ptr_to_hipxxmodule_map.end()) {
    hipxx_module = hipxx_module_found->second;
  } else {
    hipxx_module =
        new HIPxxModule(module_str);  // Create a new module for this source
    hipxx_module->compileOnce(this);  // Compile it
    host_f_ptr_to_hipxxmodule_map[module_str] = hipxx_module;
    // TODO Place it in the Backend cache
  }
  HIPxxKernel *kernel = hipxx_module->getKernel(std::string(host_f_name));
  if (!kernel) {
    logCritical(
        "Device {}: tried to register host function <{}, {}> but failed to "
        "find kernel with matching name",
        getDeviceId(), host_f_ptr, host_f_name);
    std::abort();
  }

  kernel->setHostPtr(host_f_ptr);
  assert(kernel->getDevPtr() != nullptr);

  hipxx_kernels.push_back(kernel);
  logDebug("Device {}: successfully registered function as kernel.", getName());
  return;
}

void HIPxxDevice::addQueue(HIPxxQueue *hipxx_queue_) {
  auto queue_found =
      std::find(hipxx_queues.begin(), hipxx_queues.end(), hipxx_queue_);
  if (queue_found == hipxx_queues.end()) hipxx_queues.push_back(hipxx_queue_);
  return;
}
// HIPxxContext
//*************************************************************************************
HIPxxContext::HIPxxContext() {}
HIPxxContext::~HIPxxContext() {}
bool HIPxxContext::addDevice(HIPxxDevice *dev) {
  logTrace("HIPxxContext.add_device() {}", dev->getName());
  hipxx_devices.push_back(dev);
  // TODO check for success
  return true;
}

std::vector<HIPxxDevice *> &HIPxxContext::getDevices() {
  if (hipxx_devices.size() == 0)
    logWarn("HIPxxContext.get_devices() was called but hipxx_devices is empty");
  return hipxx_devices;
}

std::vector<HIPxxQueue *> &HIPxxContext::getQueues() {
  if (hipxx_queues.size() == 0) {
    logCritical(
        "HIPxxContext.get_queues() was called but no queues were added to "
        "this context");
    std::abort();
  }
  return hipxx_queues;
}
void HIPxxContext::addQueue(HIPxxQueue *q) {
  logTrace("HIPxxContext.add_queue()");
  hipxx_queues.push_back(q);
}
hipStream_t HIPxxContext::findQueue(hipStream_t stream) {
  std::vector<HIPxxQueue *> Queues = getQueues();
  if (stream == nullptr) return Backend->getActiveQueue();

  auto I = std::find(Queues.begin(), Queues.end(), stream);
  if (I == Queues.end()) return nullptr;
  return *I;
}

void HIPxxContext::finishAll() {
  for (HIPxxQueue *q : hipxx_queues) q->finish();
}

void *HIPxxContext::allocate(size_t size) {
  return allocate(size, 0, HIPxxMemoryType::Shared);
}

void *HIPxxContext::allocate(size_t size, HIPxxMemoryType mem_type) {
  return allocate(size, 0, mem_type);
}
void *HIPxxContext::allocate(size_t size, size_t alignment,
                             HIPxxMemoryType mem_type) {
  std::lock_guard<std::mutex> Lock(mtx);
  void *retval;

  HIPxxDevice *hipxx_dev = Backend->getActiveDevice();
  assert(hipxx_dev->getContext() == this);

  if (!hipxx_dev->reserveMem(size)) return nullptr;
  retval = allocate_(size, alignment, mem_type);
  if (retval == nullptr) hipxx_dev->releaseMemReservation(size);

  return retval;
}

// HIPxxBackend
//*************************************************************************************

HIPxxBackend::HIPxxBackend() { logDebug("HIPxxBackend Base Constructor"); };
HIPxxBackend::~HIPxxBackend(){};

void HIPxxBackend::initialize(std::string platform_str,
                              std::string device_type_str,
                              std::string device_ids_str){};

void HIPxxBackend::setActiveDevice(HIPxxDevice *hipxx_dev) {
  auto I = std::find(hipxx_devices.begin(), hipxx_devices.end(), hipxx_dev);
  if (I == hipxx_devices.end()) {
    logCritical(
        "Tried to set active device with HIPxxDevice pointer that is not in "
        "HIPxxBackend::hipxx_devices");
    std::abort();
  };
  active_dev = hipxx_dev;
  active_ctx = hipxx_dev->getContext();
  active_q = hipxx_dev->getActiveQueue();
}
std::vector<HIPxxQueue *> &HIPxxBackend::getQueues() { return hipxx_queues; }
HIPxxQueue *HIPxxBackend::getActiveQueue() {
  if (active_q == nullptr) {
    logCritical(
        "HIPxxBackend.getActiveQueue() was called but no queues have "
        "been initialized;\n");
    std::abort();
  }
  return active_q;
};

HIPxxContext *HIPxxBackend::getActiveContext() {
  if (active_ctx == nullptr) {
    logCritical(
        "HIPxxBackend.getActiveContext() was called but active_ctx is null");
    std::abort();
  }
  return active_ctx;
};

HIPxxDevice *HIPxxBackend::getActiveDevice() {
  if (active_dev == nullptr) {
    logCritical(
        "HIPxxBackend.getActiveDevice() was called but active_ctx is null");
    std::abort();
  }
  return active_dev;
};

std::vector<HIPxxDevice *> &HIPxxBackend::getDevices() { return hipxx_devices; }

size_t HIPxxBackend::getNumDevices() { return hipxx_devices.size(); }
std::vector<std::string *> &HIPxxBackend::getModulesStr() {
  return modules_str;
}

void HIPxxBackend::addContext(HIPxxContext *ctx_in) {
  hipxx_contexts.push_back(ctx_in);
}
void HIPxxBackend::addQueue(HIPxxQueue *q_in) {
  logDebug("HIPxxBackend.add_queue()");
  hipxx_queues.push_back(q_in);
}
void HIPxxBackend::addDevice(HIPxxDevice *dev_in) {
  logTrace("HIPxxDevice.add_device() {}", dev_in->getName());
  hipxx_devices.push_back(dev_in);
}

void HIPxxBackend::registerModuleStr(std::string *mod_str) {
  logTrace("HIPxxBackend->register_module()");
  std::lock_guard<std::mutex> Lock(mtx);
  getModulesStr().push_back(mod_str);
}

void HIPxxBackend::unregisterModuleStr(std::string *mod_str) {
  logTrace("HIPxxBackend->unregister_module()");
  auto found_mod = std::find(modules_str.begin(), modules_str.end(), mod_str);
  if (found_mod != modules_str.end()) {
    getModulesStr().erase(found_mod);
  } else {
    logWarn(
        "Module {} not found in HIPxxBackend.modules_str while trying to "
        "unregister",
        (void *)mod_str);
  }
}

hipError_t HIPxxBackend::configureCall(dim3 grid, dim3 block, size_t shared,
                                       hipStream_t q) {
  std::lock_guard<std::mutex> Lock(mtx);
  logTrace("HIPxxBackend->configureCall()");
  if (q == nullptr) q = getActiveQueue();
  HIPxxExecItem *ex = new HIPxxExecItem(grid, block, shared, q);
  hipxx_execstack.push(ex);

  return hipSuccess;
}

hipError_t HIPxxBackend::setArg(const void *arg, size_t size, size_t offset) {
  logTrace("HIPxxBackend->set_arg()");
  std::lock_guard<std::mutex> Lock(mtx);
  HIPxxExecItem *ex = hipxx_execstack.top();
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

bool HIPxxBackend::registerFunctionAsKernel(std::string *module_str,
                                            const void *host_f_ptr,
                                            const char *host_f_name) {
  logTrace("HIPxxBackend.registerFunctionAsKernel()");
  for (auto &ctx : hipxx_contexts)
    for (auto &dev : ctx->getDevices())
      dev->registerFunctionAsKernel(module_str, host_f_ptr, host_f_name);
  return true;
}

HIPxxDevice *HIPxxBackend::findDeviceMatchingProps(
    const hipDeviceProp_t *properties) {
  HIPxxDevice *matched_device;
  int maxMatchedCount = 0;
  for (auto &dev : hipxx_devices) {
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

// HIPxxQueue
//*************************************************************************************
HIPxxQueue::HIPxxQueue(HIPxxDevice *hipxx_device_, unsigned int flags_,
                       int priority_)
    : hipxx_device(hipxx_device_), flags(flags_), priority(priority_) {
  hipxx_context = hipxx_device_->getContext();
};
HIPxxQueue::HIPxxQueue(HIPxxDevice *hipxx_device_, unsigned int flags_)
    : HIPxxQueue(hipxx_device_, flags_, 0){};
HIPxxQueue::HIPxxQueue(HIPxxDevice *hipxx_device_)
    : HIPxxQueue(hipxx_device_, 0, 0){};
HIPxxQueue::~HIPxxQueue(){};

HIPxxDevice *HIPxxQueue::getDevice() {
  if (hipxx_device == nullptr) {
    logCritical(
        "HIPxxQueue.getDevice() was called but device is a null pointer");
    std::abort();  // TODO Exception?
  }

  return hipxx_device;
}
