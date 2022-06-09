/**
 * @file CHIPDriver.cc
 * @author Paulius Velesko (pvelesko@gmail.com)
 * @brief Definitions of extern declared functions and objects in CHIPDriver.hh
 * Initializing the CHIP runtime with backend selection through CHIP_BE
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

std::once_flag Initialized;
std::once_flag EnvInitialized;
std::once_flag Uninitialized;
bool UsingDefaultBackend;
CHIPBackend *Backend;
std::string CHIPPlatformStr, CHIPDeviceTypeStr, CHIPDeviceStr, CHIPBackendType;

std::string read_env_var(std::string EnvVar, bool Lower = true) {
  logDebug("Reading {} from env", EnvVar);
  const char *EnvVarIn = std::getenv(EnvVar.c_str());
  if (EnvVarIn == nullptr) {
    return std::string();
  }
  std::string Var = std::string(EnvVarIn);
  if (Lower)
    std::transform(Var.begin(), Var.end(), Var.begin(),
                   [](unsigned char Ch) { return std::tolower(Ch); });

  return Var;
}

void CHIPReadEnvVarsCallOnce() {
  CHIPPlatformStr = read_env_var("CHIP_PLATFORM");
  if (CHIPPlatformStr.size() == 0)
    CHIPPlatformStr = "0";

  CHIPDeviceTypeStr = read_env_var("CHIP_DEVICE_TYPE");
  if (CHIPDeviceTypeStr.size() == 0)
    CHIPDeviceTypeStr = "gpu";

  CHIPDeviceStr = read_env_var("CHIP_DEVICE");
  if (CHIPDeviceStr.size() == 0)
    CHIPDeviceStr = "0";

  UsingDefaultBackend = false;
  CHIPBackendType = read_env_var("CHIP_BE");
  if (CHIPBackendType.size() == 0) {
    CHIPBackendType = "opencl";
    UsingDefaultBackend = true;
  }

  logDebug("CHIP_PLATFORM={}", CHIPPlatformStr.c_str());
  logDebug("CHIP_DEVICE_TYPE={}", CHIPDeviceTypeStr.c_str());
  logDebug("CHIP_DEVICE={}", CHIPDeviceStr.c_str());
  logDebug("CHIP_BE={}", CHIPBackendType.c_str());
}

void CHIPReadEnvVars() {
  std::call_once(EnvInitialized, &CHIPReadEnvVarsCallOnce);
}

void CHIPInitializeCallOnce(std::string BackendStr) {
  CHIPReadEnvVars();
  logDebug("CHIPDriver Initialize");

  // Read JIT options from the env

  // Get the current Backend Env Var

  // If no BE is passed to init explicitly, read env var
  std::string ChipBe;
  if (BackendStr.size() == 0) {
    ChipBe = CHIPBackendType;
  } else {
    ChipBe = BackendStr;
  }

  // TODO Check configuration for what backends are configured
  if (!ChipBe.compare("opencl")) {
    if (UsingDefaultBackend)
      logWarn("CHIP_BE was not set. Defaulting to OPENCL");
    logDebug("CHIPBE=OPENCL... Initializing OpenCL Backend");
    Backend = new CHIPBackendOpenCL();
  } else if (!ChipBe.compare("level0")) {
    logDebug("CHIPBE=LEVEL0... Initializing Level0 Backend");
    Backend = new CHIPBackendLevel0();
  } else {
    CHIPERR_LOG_AND_THROW(
        "Invalid CHIP-SPV Backend Selected. Accepted values : level0, opencl.",
        hipErrorInitializationError);
  }
  Backend->initialize(CHIPPlatformStr, CHIPDeviceTypeStr, CHIPDeviceStr);
}

extern void CHIPInitialize(std::string BE) {
  std::call_once(Initialized, &CHIPInitializeCallOnce, BE);
}

void CHIPUninitializeCallOnce() {
  logDebug("Uninitializing CHIP...");
  Backend->uninitialize();
}

extern void CHIPUninitialize() {
  std::call_once(Uninitialized, &CHIPUninitializeCallOnce);
}

extern hipError_t CHIPReinitialize(const uintptr_t *NativeHandles, int NumHandles) {
    CHIPReadEnvVars();
    logDebug("CHIPDriver REInitialize");

    if (Backend)
    {
      logDebug("uninitializing existing Backend object.");
      Backend->uninitialize();
      delete Backend;
    }

    // TODO Check configuration for what backends are configured
    if (!CHIPBackendType.compare("opencl")) {
      if (UsingDefaultBackend)
        logWarn("CHIP_BE was not set. Defaulting to OPENCL");
      logDebug("CHIPBE=OPENCL... Initializing OpenCL Backend");
      Backend = new CHIPBackendOpenCL();
    } else if (!CHIPBackendType.compare("level0")) {
      logDebug("CHIPBE=LEVEL0... Initializing Level0 Backend");
      Backend = new CHIPBackendLevel0();
    } else {
      CHIPERR_LOG_AND_THROW(
          "Invalid CHIP-SPV Backend Selected. Accepted values : level0, opencl.",
          hipErrorInitializationError);
    }

    int RequiredHandles = Backend->ReqNumHandles();
    if (RequiredHandles != NumHandles)  {
        delete Backend;
        CHIPERR_LOG_AND_THROW(
            "Invalid number of native handles",
            hipErrorInitializationError);
    }

    Backend->initializeFromNative(NativeHandles, NumHandles);
    return hipSuccess;
}
