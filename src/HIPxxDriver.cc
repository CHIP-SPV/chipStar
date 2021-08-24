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

void read_env_vars(std::string& HIPxxPlatformStr,
                   std::string& HIPxxDeviceTypeStr,
                   std::string& HIPxxDeviceStr) {
  const char* HIPxxPlatformStr_in = std::getenv("HIPXX_PLATFORM");
  const char* HIPxxDeviceTypeStr_in = std::getenv("HIPXX_DEVICE_TYPE");
  const char* HIPxxDeviceStr_in = std::getenv("HIPXX_DEVICE");

  std::cout << "\n";
  if (HIPxxPlatformStr_in == nullptr) {
    std::cout << "HIPXX_PLATFORM unset. Using default.\n";
    HIPxxPlatformStr = "0";
  } else {
    HIPxxPlatformStr = HIPxxPlatformStr_in;
  }

  if (HIPxxDeviceTypeStr_in == nullptr) {
    std::cout << "HIPXX_DEVICE_TYPE unset. Using default.\n";
    HIPxxDeviceTypeStr = "default";
  } else {
    HIPxxDeviceTypeStr = HIPxxDeviceTypeStr_in;
  }

  if (HIPxxDeviceStr_in == nullptr) {
    std::cout << "HIPXX_DEVICE unset. Using default.\n";
    HIPxxDeviceStr = "0";
  } else {
    HIPxxDeviceStr = HIPxxDeviceStr_in;
  }

  std::cout << "\n";
  std::cout << "HIPXX_PLATFORM=" << HIPxxPlatformStr << std::endl;
  std::cout << "HIPXX_DEVICE_TYPE=" << HIPxxDeviceTypeStr << std::endl;
  std::cout << "HIPXX_DEVICE=" << HIPxxDeviceStr << std::endl;
  std::cout << "\n";
};

void _initialize() {
  std::string HIPxxPlatformStr, HIPxxDeviceTypeStr, HIPxxDeviceStr;
  read_env_vars(HIPxxPlatformStr, HIPxxDeviceTypeStr, HIPxxDeviceStr);
  std::cout << "HIPxxDriver Initialize\n";
  // Get the current Backend Env Var
  Backend = new HIPxxBackendOpenCL();
  Backend->initialize(HIPxxPlatformStr, HIPxxDeviceTypeStr, HIPxxDeviceStr);
};

void initialize() { std::call_once(initialized, &_initialize); };