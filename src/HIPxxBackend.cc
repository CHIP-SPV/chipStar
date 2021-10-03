#include "HIPxxBackend.hh"
//*************************************************************************************
HIPxxEvent::HIPxxEvent(HIPxxContext *ctx_in, unsigned flags_in)
    : status(EVENT_STATUS_INIT), flags(flags_in), hipxx_context(ctx_in) {}
HIPxxEvent::HIPxxEvent() {}
HIPxxEvent::~HIPxxEvent() {}

bool HIPxxEvent::recordStream(HIPxxQueue *hipxx_queue_){};
bool HIPxxEvent::wait(){};
bool HIPxxEvent::isFinished(){};
float HIPxxEvent::getElapsedTime(HIPxxEvent *other){};
//*************************************************************************************
void HIPxxModule::addKernel(void *HostFunctionPtr,
                            std::string HostFunctionName) {
  // TODO
  HIPxxKernel *kernel = new HIPxxKernel();
  hipxx_kernels.push_back(kernel);
}
//*************************************************************************************
HIPxxKernel::HIPxxKernel(){};
HIPxxKernel::~HIPxxKernel(){};
std::string HIPxxKernel::getName() { return host_f_name; }
const void *HIPxxKernel::getHostPtr() { return host_f_ptr; }
const void *HIPxxKernel::getDevPtr() { return dev_f_ptr; }
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
//*************************************************************************************
HIPxxKernel *HIPxxDevice::findKernelByHostPtr(const void *hostPtr) {
  logTrace("HIPxxDevice::findKernelByHostPtr({})", hostPtr);
  std::vector<HIPxxKernel *> hipxx_kernels = get_kernels();
  logDebug("Listing Kernels for device {}", device_name);
  for (auto &kernel : hipxx_kernels) {
    logDebug("{}", kernel->getName());
  }

  auto found_kernel = std::find_if(hipxx_kernels.begin(), hipxx_kernels.end(),
                                   [&hostPtr](HIPxxKernel *kernel) {
                                     return kernel->getHostPtr() == hostPtr;
                                   });

  if (found_kernel == hipxx_kernels.end()) {
    logCritical("Failed to find kernel {} on device #{}:{}", hostPtr, pcie_idx,
                device_name);
    std::abort();  // Exception
  } else {
    logDebug("Found kernel {} with host pointer {}", (*found_kernel)->getName(),
             (*found_kernel)->getHostPtr());
  }

  return *found_kernel;
}

bool HIPxxContext::add_device(HIPxxDevice *dev) {
  logTrace("HIPxxContext.add_device() {}", dev->get_name());
  hipxx_devices.push_back(dev);
  // TODO check for success
  return true;
}
//*************************************************************************************
HIPxxDevice::HIPxxDevice() {
  logDebug("Device {} is {}: name \"{}\" \n", pcie_idx, (void *)this,
           hip_device_props.name);
};
bool HIPxxDevice::add_context(HIPxxContext *ctx) {
  logTrace("HIPxxDevice.add_context() {}");
  hipxx_contexts.push_back(ctx);
  // TODO check for success
  return true;
}

bool HIPxxDevice::getModuleAndFName(const void *HostFunction,
                                    std::string &FunctionName,
                                    HIPxxModule *hipxx_module) {
  logTrace("HIPxxDevice.getModuleAndFName");
  std::lock_guard<std::mutex> Lock(mtx);

  auto it1 = HostPtrToModuleMap.find(HostFunction);
  auto it2 = HostPtrToNameMap.find(HostFunction);

  if ((it1 == HostPtrToModuleMap.end()) || (it2 == HostPtrToNameMap.end()))
    return false;

  FunctionName.assign(it2->second);
  hipxx_module = it1->second;
  return true;
}

void HIPxxDevice::copy_device_properties(hipDeviceProp_t *prop) {
  logTrace("HIPxxDevice->copy_device_properties()");
  if (prop) std::memcpy(prop, &this->hip_device_props, sizeof(hipDeviceProp_t));
}

std::vector<HIPxxKernel *> &HIPxxDevice::get_kernels() {
  return hipxx_kernels;
};

void HIPxxDevice::add_kernel(HIPxxKernel *kernel) {
  logTrace("Adding kernel {} to device # {} {}", kernel->getName(), pcie_idx,
           device_name);
  hipxx_kernels.push_back(kernel);
}

HIPxxContext *HIPxxDevice::get_default_context() {
  // TODO Check for initialization
  // if (hipxx_contexts.size() == 0)
  return hipxx_contexts.at(0);
}

bool HIPxxDevice::reserve_mem(size_t bytes) {
  logTrace("HIPxxDevice->reserve_mem()");
  std::lock_guard<std::mutex> Lock(mtx);
  if (bytes <= (hip_device_props.totalGlobalMem - TotalUsedMem)) {
    TotalUsedMem += bytes;
    if (TotalUsedMem > MaxUsedMem) MaxUsedMem = TotalUsedMem;
    logDebug("Currently used memory on dev {}: {} M\n", pcie_idx,
             (TotalUsedMem >> 20));
    return true;
  } else {
    logError(
        "Can't allocate {} bytes of memory on device # {}\n. "
        "GlobalMemSize:{} TotalUsedMem: {}",
        bytes, pcie_idx, hip_device_props.totalGlobalMem, TotalUsedMem);
    return false;
  }
}

bool HIPxxDevice::release_mem(size_t bytes) {
  std::lock_guard<std::mutex> Lock(mtx);
  if (TotalUsedMem >= bytes) {
    TotalUsedMem -= bytes;
    return true;
  } else {
    return false;
  }
}

//*************************************************************************************

//*************************************************************************************

std::string HIPxxQueue::get_info() {
  // TODO review this
  std::string info;
  info = hipxx_device->get_name();
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
