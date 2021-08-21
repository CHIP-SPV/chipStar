/**
 * @file HIPxxBindings.hh
 * @author Paulius Velesko (pvelesko@gmail.com)
 * @brief Implementations of the HIP API functions using the HIPxx interface
 * providing basic functionality such hipMemcpy, host and device function
 * registration, hipLaunchByPtr, etc.
 * These functions operate on base HIPxx class pointers allowing for backend
 * selection at runtime and backend-specific implementations are done by
 * inheriting from base HIPxx classes and overriding virtual member functions.
 * @version 0.1
 * @date 2021-08-19
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef HIPXX_BINDINGS_H
#define HIPXX_BINDINGS_H

#include "HIPxxBackend.hh"
#include "HIPxxDriver.hh"
#include "temporary.hh"

extern "C" void __hipRegisterFunction(void **data, const void *hostFunction,
                                      char *deviceFunction,
                                      const char *deviceName,
                                      unsigned int threadLimit, void *tid,
                                      void *bid, dim3 *blockDim, dim3 *gridDim,
                                      int *wSize) {
  Backend->initialize();
  std::string *module_str = reinterpret_cast<std::string *>(data);

  std::string devFunc = deviceFunction;
  logDebug("RegisterFunction on module {}\n", (void *)module_str);

  for (HIPxxDevice *dev : Backend->get_devices()) {
    if (dev->registerFunction(module_str, hostFunction, deviceName)) {
      logDebug("__hipRegisterFunction: kernel {} found\n", deviceName);
    } else {
      logCritical("__hipRegisterFunction can NOT find kernel: {} \n",
                  deviceName);
      std::abort();
    }
  }
  // Put the function information into a temproary storage
  // LZDriver::RegFunctions.push_back(
  //    std::make_tuple(module, hostFunction, deviceName));
}

#endif