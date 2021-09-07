#include "HIPxxBackend.hh"

bool HIPxxDevice::register_function_as_kernel(std::string* module_str,
                                              const void* HostFunctionPtr,
                                              const char* FunctionName) {
  std::lock_guard<std::mutex> Lock(mtx);
  logTrace("HIPxxDevice::registerFunction");

  // Get modules in binary representation
  // These are extracted from the fat binary
  std::vector<std::string*> modules_str = Backend->get_modules_str();
  if (modules_str.size() == 0) {
    logCritical(
        "HIPxxDevice tried to register function but modules_str was empty");
    std::abort();
  }

  auto it = std::find(modules_str.begin(), modules_str.end(), module_str);
  if (it == modules_str.end()) {
    logError("Module PTR not FOUND: {}\n", (void*)module_str);
    return false;
  }

  // Parse out HIPxxModule from binary representation
  HIPxxModule* module = new HIPxxModule(module_str);

  HostPtrToModuleStrMap.emplace(std::make_pair(HostFunctionPtr, module_str));
  HostPtrToModuleMap.emplace(std::make_pair(HostFunctionPtr, module));
  HostPtrToNameMap.emplace(std::make_pair(HostFunctionPtr, FunctionName));

  // TODO Create & compile/createProgram a kernel
  // Maybe this should be done in the Module constructor?
  // HIPxxKernel kernel(HostFunctionPtr, FunctionName);
  // module->add_kernel(kernel);

  hipxx_modules.push_back(module);

  // return (PrimaryContext->createProgramBuiltin(module, HostFunctionPtr, temp)
  return true;
}

HIPxxModule::HIPxxModule(std::string* module_str) {
  // TODO
  logDebug("Initializing HIPxxModule from binary string\n", "");
}

void HIPxxModule::add_kernel(void* HostFunctionPtr,
                             std::string HostFunctionName) {
  // TODO
  HIPxxKernel* kernel = new HIPxxKernel();
  Kernels.push_back(kernel);
}

bool HIPxxContext::add_device(HIPxxDevice* dev) {
  logTrace("HIPxxContext.add_device() {}", dev->get_name());
  hipxx_devices.push_back(dev);
  // TODO check for success
  return true;
}

bool HIPxxDevice::add_context(HIPxxContext* ctx) {
  logTrace("HIPxxDevice.add_context() {}");
  hipxx_contexts.push_back(ctx);
  // TODO check for success
  return true;
}

HIPxxContext* HIPxxDevice::get_default_context() {
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
