#include "HIPxxBackend.hh"

bool HIPxxDevice::registerFunction(std::string* module_str,
                                   const void* HostFunctionPtr,
                                   const char* FunctionName) {
  std::lock_guard<std::mutex> Lock(DeviceMutex);

  // Get modules in binary representation
  // These are extracted from the fat binary
  std::vector<std::string*> modules_str = Backend->get_modules_str();

  auto it = std::find(modules_str.begin(), modules_str.end(), module_str);
  if (it == modules_str.end()) {
    logError("Module PTR not FOUND: {%p}\n", (void*)module_str);
    return false;
  }

  // Parse out HIPxxModule from binary representation
  HIPxxModule* module = new HIPxxModule(module_str);

  HostPtrToModuleStrMap.emplace(std::make_pair(HostFunctionPtr, module_str));
  HostPtrToModuleStrMap.emplace(std::make_pair(HostFunctionPtr, module));
  HostPtrToNameMap.emplace(std::make_pair(HostFunctionPtr, FunctionName));

  // TODO Create & compile/createProgram a kernel
  // Maybe this should be done in the Module constructor?
  // HIPxxKernel kernel(HostFunctionPtr, FunctionName);
  // module->add_kernel(kernel);

  Modules.push_back(module);

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
  Devices.push_back(dev);
  // TODO check for success
  return true;
}

bool HIPxxDevice::add_context(HIPxxContext* ctx) {
  xxContexts.push_back(ctx);
  // TODO check for success
  return true;
}

HIPxxContext* HIPxxDevice::get_default_context() {
  // TODO Check for initialization
  // if (xxContexts.size() == 0)
  return xxContexts.at(0);
