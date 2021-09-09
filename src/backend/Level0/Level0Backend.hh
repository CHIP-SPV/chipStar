#ifndef HIPXX_BACKEND_LEVEL0_H
#define HIPXX_BACKEND_LEVEL0_H

#include "../../HIPxxBackend.hh"
#include "../include/ze_api.h"

class HIPxxBackendLevel0 : public HIPxxBackend {
 public:
  virtual void initialize(std::string HIPxxPlatformStr,
                          std::string HIPxxDeviceTypeStr,
                          std::string HIPxxDeviceStr) override {
    logDebug("HIPxxBackendLevel0 Initialize");
    // Initialize the driver
    ze_result_t status = zeInit(0);
    // LZ_PROCESS_ERROR(status);
    logDebug("INITIALIZE LEVEL-0 (via calling zeInit) {}\n", status);

    // Initialize HipLZ device drivers and relevant devices
    // LZDriver::InitDrivers(HipLZDrivers, ZE_DEVICE_TYPE_GPU);

    // // Register fat binary modules
    // for (std::string* module : LZDriver::FatBinModules) {
    //   for (size_t driverId = 0; driverId < NumLZDrivers; ++driverId) {
    //     LZDriver::HipLZDriverById(driverId).registerModule(module);
    //   }
    // }

    // // Register functions
    // for (auto fi : LZDriver::RegFunctions) {
    //   std::string* module = std::get<0>(fi);
    //   const void* hostFunction = std::get<1>(fi);
    //   const char* deviceName = std::get<2>(fi);
    //   for (size_t driverId = 0; driverId < NumLZDrivers; ++driverId) {
    //     if (LZDriver::HipLZDriverById(driverId).registerFunction(
    //             module, hostFunction, deviceName)) {
    //       logDebug("__hipRegisterFunction: HipLZ kernel {} found\n",
    //                deviceName);
    //     } else {
    //       logCritical("__hipRegisterFunction can NOT find HipLZ kernel: {}
    //       \n",
    //                   deviceName);
    //       std::abort();
    //     }
    //   }
    // }

    // // Register globale variables
    // for (auto vi : LZDriver::GlobalVars) {
    //   std::string* module = std::get<0>(vi);
    //   char* hostVar = std::get<1>(vi);
    //   const char* deviceName = std::get<2>(vi);
    //   int size = std::get<3>(vi);
    //   std::string devName = deviceName;
    //   for (size_t driverId = 0; driverId < NumLZDrivers; ++driverId) {
    //     if (LZDriver::HipLZDriverById(driverId).registerVar(module, hostVar,
    //                                                         deviceName,
    //                                                         size)) {
    //       logDebug("__hipRegisterVar: variable {} found\n", deviceName);
    //     } else {
    //       logError("__hipRegisterVar could not find: {}\n", deviceName);
    //     }
    //   }
    // }
  }

  virtual void initialize() override {
    std::string empty;
    initialize(empty, empty, empty);
  }

  void uninitialize() override {
    logTrace("HIPxxBackendLevel0 uninitializing");
    logWarn("HIPxxBackendLevel0->uninitialize() not implemented");
  }
};

class HIPxxContextLevel0 : public HIPxxContext {
 public:
  virtual bool register_function_as_kernel(std::string* module_str,
                                           const void* HostFunctionPtr,
                                           const char* FunctionName) {
    logDebug(
        "HIPxxContextLevel0.register_function_as_kernel not yet implemented");
    return true;
  }
};

#endif