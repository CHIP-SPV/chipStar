#include "HIPxxBackend.hh"
// HIPxxEvent
// ************************************************************************
HIPxxEvent::HIPxxEvent(HIPxxContext *ctx_in, unsigned flags_in)
    : status(EVENT_STATUS_INIT), flags(flags_in), hipxx_context(ctx_in) {}
HIPxxEvent::HIPxxEvent() {}
HIPxxEvent::~HIPxxEvent() {}

bool HIPxxEvent::recordStream(HIPxxQueue *hipxx_queue_){};
bool HIPxxEvent::wait(){};
bool HIPxxEvent::isFinished(){};
float HIPxxEvent::getElapsedTime(HIPxxEvent *other){};

// HIPxxModule
//*************************************************************************************
void HIPxxModule::addKernel(void *HostFunctionPtr,
                            std::string HostFunctionName) {
  // TODO
  HIPxxKernel *kernel = new HIPxxKernel();
  hipxx_kernels.push_back(kernel);
}

// HIPxxKernel
//*************************************************************************************
HIPxxKernel::HIPxxKernel(){};
HIPxxKernel::~HIPxxKernel(){};
std::string HIPxxKernel::getName() { return host_f_name; }
const void *HIPxxKernel::getHostPtr() { return host_f_ptr; }
const void *HIPxxKernel::getDevPtr() { return dev_f_ptr; }

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
  logDebug("HIPxxExecItem.set_arg() on {} size {} offset {}\n", (void *)this,
           size, offset);
  offset_sizes.push_back(std::make_tuple(offset, size));
}
hipError_t HIPxxExecItem::launch(HIPxxKernel *Kernel) {
  logWarn("Calling HIPxxExecItem.launch() base launch which does nothing");
  return hipSuccess;
};

hipError_t HIPxxExecItem::launchByHostPtr(const void *hostPtr) {
  if (hipxx_queue == nullptr) {
    logCritical("HIPxxExecItem.launch() was called but queue pointer is null");
    // TODO better errors
    std::abort();
  }

  HIPxxDevice *dev = hipxx_queue->getDevice();
  this->hipxx_kernel = dev->findKernelByHostPtr(hostPtr);
  logTrace("Found kernel for host pointer {} : {}", hostPtr,
           hipxx_kernel->getName());
  // TODO verify that all is in place either here or in HIPxxQueue
  return hipxx_queue->launch(this);
}

// HIPxxDevice
//*************************************************************************************
HIPxxDevice::HIPxxDevice() {
  logDebug("Device {} is {}: name \"{}\" \n", idx, (void *)this,
           hip_device_props.name);
};
HIPxxDevice::~HIPxxDevice(){};

void HIPxxDevice::addKernel(HIPxxKernel *kernel) {
  logTrace("Adding kernel {} to device # {} {}", kernel->getName(), idx,
           device_name);
  hipxx_kernels.push_back(kernel);
}
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
HIPxxQueue *HIPxxDevice::getQueue() { return q; }
int HIPxxDevice::getDeviceId() { return idx; }
bool HIPxxDevice::getModuleAndFName(const void *host_f_ptr,
                                    std::string &host_f_name,
                                    HIPxxModule *hipxx_module) {
  logTrace("HIPxxDevice.getModuleAndFName");
  std::lock_guard<std::mutex> Lock(mtx);

  auto it1 = host_ptr_to_hipxxmodule_map.find(host_f_ptr);
  auto it2 = host_ptr_to_name_map.find(host_f_ptr);

  if ((it1 == host_ptr_to_hipxxmodule_map.end()) ||
      (it2 == host_ptr_to_name_map.end()))
    return false;

  host_f_name.assign(it2->second);
  hipxx_module = it1->second;
  return true;
}

bool HIPxxDevice::allocate(size_t bytes) {
  logTrace("HIPxxDevice->reserve_mem()");
  std::lock_guard<std::mutex> Lock(mtx);
  if (bytes <= (hip_device_props.totalGlobalMem - total_used_mem)) {
    total_used_mem += bytes;
    if (total_used_mem > max_used_mem) max_used_mem = total_used_mem;
    logDebug("Currently used memory on dev {}: {} M\n", idx,
             (total_used_mem >> 20));
    return true;
  } else {
    logError(
        "Can't allocate {} bytes of memory on device # {}\n. "
        "GlobalMemSize:{} TotalUsedMem: {}",
        bytes, idx, hip_device_props.totalGlobalMem, total_used_mem);
    return false;
  }
}
bool HIPxxDevice::free(size_t bytes) {
  std::lock_guard<std::mutex> Lock(mtx);
  if (total_used_mem >= bytes) {
    total_used_mem -= bytes;
    return true;
  } else {
    return false;
  }
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
HIPxxQueue *HIPxxContext::getDefaultQueue() {
  if (hipxx_queues.size() == 0) {
    logCritical(
        "HIPxxContext.get_default_queue() was called but hipxx_queues is "
        "empty");
    std::abort();
  }
  return hipxx_queues[0];
}

hipStream_t HIPxxContext::findQueue(hipStream_t stream) {
  std::vector<HIPxxQueue *> Queues = getQueues();
  HIPxxQueue *DefaultQueue = Queues.at(0);
  if (stream == nullptr || stream == DefaultQueue) return DefaultQueue;

  auto I = std::find(Queues.begin(), Queues.end(), stream);
  if (I == Queues.end()) return nullptr;
  return *I;
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
  active_q = hipxx_dev->getQueue();
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
  logTrace("HIPxxBackend->configure_call()");
  std::lock_guard<std::mutex> Lock(mtx);
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
  for (auto &ctx : hipxx_contexts) {
    ctx->registerFunctionAsKernel(module_str, host_f_ptr, host_f_name);
  }
  return true;
}

// HIPxxQueue
//*************************************************************************************

HIPxxQueue::HIPxxQueue(){};
HIPxxQueue::~HIPxxQueue(){};

std::string HIPxxQueue::getInfo() {
  // TODO review this
  std::string info;
  info = hipxx_device->getName();
  return info;
}

HIPxxDevice *HIPxxQueue::getDevice() {
  if (hipxx_device == nullptr) {
    logCritical(
        "HIPxxQueue.getDevice() was called but device is a null pointer");
    std::abort();  // TODO Exception?
  }

  return hipxx_device;
}
