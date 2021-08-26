/**
 * @file HIPxxDriver.cc
 * @author Paulius Velesko (pvelesko@gmail.com)
 * @brief Definitions of extern declared functions and objects in HIPxxDriver.hh
 * Initializing the HIPxx runtime with backend selection through HIPXX_BE
 * environment variable.
 * @version 0.1
 * @date 2021-08-19
 *
 * @copyright Copyright (c) 2021
 *
 */
#include "HIPxxDriver.hh"

#include <string>

#include "backend/backends.hh"

std::once_flag initialized;
HIPxxBackend* Backend;

std::string read_env_var(std::string ENV_VAR) {
  const char* ENV_VAR_IN = std::getenv(ENV_VAR.c_str());
  if (ENV_VAR_IN == nullptr) {
    return std::string();
  }

  return std::string(ENV_VAR_IN);
};

std::string read_backend_selection();

void read_env_vars(std::string& HIPxxPlatformStr,
                   std::string& HIPxxDeviceTypeStr,
                   std::string& HIPxxDeviceStr) {
  HIPxxPlatformStr = read_env_var("HIPXX_PLATFORM");
  if (HIPxxPlatformStr.size() == 0) HIPxxPlatformStr = "0";

  HIPxxDeviceTypeStr = read_env_var("HIPXX_DEVICE_TYPE");
  if (HIPxxDeviceTypeStr.size() == 0) HIPxxDeviceTypeStr = "default";

  HIPxxDeviceStr = read_env_var("HIPXX_DEVICE");
  if (HIPxxDeviceStr.size() == 0) HIPxxDeviceStr = "0";

  std::cout << "\n";
  std::cout << "HIPXX_PLATFORM=" << HIPxxPlatformStr << std::endl;
  std::cout << "HIPXX_DEVICE_TYPE=" << HIPxxDeviceTypeStr << std::endl;
  std::cout << "HIPXX_DEVICE=" << HIPxxDeviceStr << std::endl;
  std::cout << "\n";
};

void _initialize(std::string BE) {
  std::string HIPxxPlatformStr, HIPxxDeviceTypeStr, HIPxxDeviceStr;
  read_env_vars(HIPxxPlatformStr, HIPxxDeviceTypeStr, HIPxxDeviceStr);
  std::cout << "HIPxxDriver Initialize\n";
  // Get the current Backend Env Var
  std::string HIPXX_BE = read_env_var("HIPXX_BE");
  if (!HIPXX_BE.compare("OPENCL")) {
    Backend = new HIPxxBackendOpenCL();
  } else if (!HIPXX_BE.compare("LEVEL0")) {
    std::cout << "LEVEL0 Backend not yet implemented\n";
    std::abort();
  } else if (!HIPXX_BE.compare("")) {
    std::cout << "HIPXX_BE was not set. Defaulting to OPENCL\n";
    Backend = new HIPxxBackendOpenCL();
  } else {
    std::cout << "Invalid Backend Selection\n";
    std::abort();
  }
  Backend->initialize(HIPxxPlatformStr, HIPxxDeviceTypeStr, HIPxxDeviceStr);
}

void initialize(std::string BE) {
  std::call_once(initialized, &_initialize, BE);
};
