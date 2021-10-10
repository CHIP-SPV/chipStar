/**
 * @file CHIPDriver.cc
 * @author Paulius Velesko (pvelesko@gmail.com)
 * @brief Definitions of extern declared functions and objects in CHIPDriver.hh
 * Initializing the CHIP runtime with backend selection through HIPXX_BE
 * environment variable.
 * @version 0.1
 * @date 2021-08-19
 *
 * @copyright Copyright (c) 2021
 *
 */
#include "CHIPDriver.hh"

#include <string>

#include "backend/backends.hh"

std::once_flag initialized;
std::once_flag uninitialized;
CHIPBackend* Backend;

std::string read_env_var(std::string ENV_VAR) {
  const char* ENV_VAR_IN = std::getenv(ENV_VAR.c_str());
  if (ENV_VAR_IN == nullptr) {
    return std::string();
  }

  return std::string(ENV_VAR_IN);
};

std::string read_backend_selection();

void read_env_vars(std::string& CHIPPlatformStr, std::string& CHIPDeviceTypeStr,
                   std::string& CHIPDeviceStr) {
  CHIPPlatformStr = read_env_var("HIPXX_PLATFORM");
  if (CHIPPlatformStr.size() == 0) CHIPPlatformStr = "0";

  CHIPDeviceTypeStr = read_env_var("HIPXX_DEVICE_TYPE");
  if (CHIPDeviceTypeStr.size() == 0) CHIPDeviceTypeStr = "default";

  CHIPDeviceStr = read_env_var("HIPXX_DEVICE");
  if (CHIPDeviceStr.size() == 0) CHIPDeviceStr = "0";

  std::cout << "\n";
  std::cout << "HIPXX_PLATFORM=" << CHIPPlatformStr << std::endl;
  std::cout << "HIPXX_DEVICE_TYPE=" << CHIPDeviceTypeStr << std::endl;
  std::cout << "HIPXX_DEVICE=" << CHIPDeviceStr << std::endl;
  std::cout << "\n";
};

void CHIPInitializeCallOnce(std::string BE) {
  std::string CHIPPlatformStr, CHIPDeviceTypeStr, CHIPDeviceStr;
  read_env_vars(CHIPPlatformStr, CHIPDeviceTypeStr, CHIPDeviceStr);
  logDebug("CHIPDriver Initialize");
  // Get the current Backend Env Var

  // If no BE is passed to init explicitly, read env var
  std::string HIPXX_BE;
  if (BE.size() == 0) {
    HIPXX_BE = read_env_var("HIPXX_BE");
  } else {
    HIPXX_BE = BE;
  }

  // TODO Check configuration for what backends are configured
  if (!HIPXX_BE.compare("OPENCL")) {
    logTrace("HIPXXBE=OPENCL... Initializing OpenCL Backend");
    Backend = new CHIPBackendOpenCL();
  } else if (!HIPXX_BE.compare("LEVEL0")) {
    logTrace("HIPXXBE=LEVEL0... Initializing Level0 Backend");
    Backend = new CHIPBackendLevel0();
  } else if (!HIPXX_BE.compare("")) {
    logWarn("HIPXX_BE was not set. Defaulting to OPENCL");
    Backend = new CHIPBackendOpenCL();
  } else {
    logCritical("Invalid Backend Selection");
    std::abort();
  }
  Backend->initialize(CHIPPlatformStr, CHIPDeviceTypeStr, CHIPDeviceStr);
}

extern void CHIPInitialize(std::string BE) {
  std::call_once(initialized, &CHIPInitializeCallOnce, BE);
};

void CHIPUninitializeCallOnce() {
  logTrace("Uninitializing CHIP...");
  Backend->uninitialize();
}

extern void CHIPUninitialize() {
  std::call_once(uninitialized, &CHIPUninitializeCallOnce);
}
