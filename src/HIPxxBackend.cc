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

  HIPxxDevice *dev = hipxx_queue->get_device();
  this->hipxx_kernel = dev->findKernelByHostPtr(hostPtr);
  logTrace("Found kernel for host pointer {} : {}", hostPtr,
           hipxx_kernel->getName());
  // TODO verify that all is in place either here or in HIPxxQueue
  return hipxx_queue->launch(this);
}

// HIPxxDevice
//*************************************************************************************
HIPxxDevice::HIPxxDevice() {
  logDebug("Device {} is {}: name \"{}\" \n", global_id, (void *)this,
           hip_device_props.name);
};
HIPxxDevice::~HIPxxDevice(){};

void HIPxxDevice::addKernel(HIPxxKernel *kernel) {
  logTrace("Adding kernel {} to device # {} {}", kernel->getName(), global_id,
           device_name);
  hipxx_kernels.push_back(kernel);
}
std::vector<HIPxxKernel *> &HIPxxDevice::getKernels() { return hipxx_kernels; };

void HIPxxDevice::copyDeviceProperties(hipDeviceProp_t *prop) {
  logTrace("HIPxxDevice->copy_device_properties()");
  if (prop) std::memcpy(prop, &this->hip_device_props, sizeof(hipDeviceProp_t));
}

bool HIPxxDevice::addContext(HIPxxContext *ctx) {
  logTrace("HIPxxDevice.add_context() {}");
  hipxx_contexts.push_back(ctx);
  // TODO check for success
  return true;
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
    logCritical("Failed to find kernel {} on device #{}:{}", hostPtr, global_id,
                device_name);
    std::abort();  // Exception
  } else {
    logDebug("Found kernel {} with host pointer {}", (*found_kernel)->getName(),
             (*found_kernel)->getHostPtr());
  }

  return *found_kernel;
}
HIPxxContext *HIPxxDevice::getDefaultContext() {
  // TODO Check for initialization
  // if (hipxx_contexts.size() == 0)
  return hipxx_contexts.at(0);
}
bool HIPxxDevice::getModuleAndFName(const void *HostFunction,
                                    std::string &FunctionName,
                                    HIPxxModule *hipxx_module) {
  logTrace("HIPxxDevice.getModuleAndFName");
  std::lock_guard<std::mutex> Lock(mtx);

  auto it1 = host_ptr_to_hipxxmodule_map.find(HostFunction);
  auto it2 = host_ptr_to_name_map.find(HostFunction);

  if ((it1 == host_ptr_to_hipxxmodule_map.end()) ||
      (it2 == host_ptr_to_name_map.end()))
    return false;

  FunctionName.assign(it2->second);
  hipxx_module = it1->second;
  return true;
}

bool HIPxxDevice::allocate(size_t bytes) {
  logTrace("HIPxxDevice->reserve_mem()");
  std::lock_guard<std::mutex> Lock(mtx);
  if (bytes <= (hip_device_props.totalGlobalMem - total_used_mem)) {
    total_used_mem += bytes;
    if (total_used_mem > max_used_mem) max_used_mem = total_used_mem;
    logDebug("Currently used memory on dev {}: {} M\n", global_id,
             (total_used_mem >> 20));
    return true;
  } else {
    logError(
        "Can't allocate {} bytes of memory on device # {}\n. "
        "GlobalMemSize:{} TotalUsedMem: {}",
        bytes, global_id, hip_device_props.totalGlobalMem, total_used_mem);
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

//*************************************************************************************

std::string HIPxxQueue::get_info() {
  // TODO review this
  std::string info;
  info = hipxx_device->getName();
  return info;
}

HIPxxDevice *HIPxxQueue::get_device() {
  if (hipxx_device == nullptr) {
    logCritical(
        "HIPxxQueue.get_device() was called but device is a null pointer");
    std::abort();  // TODO Exception?
  }

  return hipxx_device;
}
