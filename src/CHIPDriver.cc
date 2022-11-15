/*
 * Copyright (c) 2021-22 CHIP-SPV developers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/**
 * @file CHIPDriver.cc
 * @author Paulius Velesko (pvelesko@pglc.io)
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
#include <memory>

#include "backend/backends.hh"

std::once_flag Initialized;
std::once_flag EnvInitialized;
std::once_flag Uninitialized;
bool UsingDefaultBackend;
CHIPBackend *Backend = nullptr;
std::string CHIPPlatformStr, CHIPDeviceTypeStr, CHIPDeviceStr, CHIPBackendType;

// Uninitializes the backend when the application exits.
void __attribute__((destructor)) uninitializeBackend() {
  // Generally, __hipUnregisterFatBinary would uninitialize the
  // backend when it finishes unregistration of all modules. However,
  // there won't be hip(Un)registerFatBinary() calls if the HIP
  // program does not have embedded kernels. This makes sure we
  // uninitialize the backend at exit.
  if (Backend && Backend->getNumRegisteredModules() == 0) {
    CHIPUninitialize();
    delete Backend;
    Backend = nullptr;
  }
}

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

  CHIPBackendType = read_env_var("CHIP_BE");
  if (CHIPBackendType.size() == 0) {
    CHIPBackendType = "default";
  }

  logDebug("CHIP_PLATFORM={}", CHIPPlatformStr.c_str());
  logDebug("CHIP_DEVICE_TYPE={}", CHIPDeviceTypeStr.c_str());
  logDebug("CHIP_DEVICE={}", CHIPDeviceStr.c_str());
  logDebug("CHIP_BE={}", CHIPBackendType.c_str());
}

void CHIPReadEnvVars() {
  std::call_once(EnvInitialized, &CHIPReadEnvVarsCallOnce);
}

static void createBackendObject() {
  assert(Backend == nullptr);
  const std::string ChipBe = CHIPBackendType;

  if (!ChipBe.compare("opencl")) {
#ifdef HAVE_OPENCL
    logDebug("CHIPBE=OPENCL... Initializing OpenCL Backend");
    Backend = new CHIPBackendOpenCL();
#else
    CHIPERR_LOG_AND_THROW("Invalid CHIP-SPV Backend Selected. This CHIP-SPV "
                          "was not compiled with OpenCL backend",
                          hipErrorInitializationError);
#endif
  } else if (!ChipBe.compare("level0")) {
#ifdef HAVE_LEVEL0
    logDebug("CHIPBE=LEVEL0... Initializing Level0 Backend");
    Backend = new CHIPBackendLevel0();
#else
    CHIPERR_LOG_AND_THROW("Invalid CHIP-SPV Backend Selected. This CHIP-SPV "
                          "was not compiled with Level0 backend",
                          hipErrorInitializationError);
#endif
  } else if (!ChipBe.compare("default")) {
#ifdef HAVE_LEVEL0
    if (!Backend) {
      logDebug("CHIPBE=default... trying Level0 Backend");
      Backend = new CHIPBackendLevel0();
    }
#endif
#ifdef HAVE_OPENCL
    if (!Backend) {
      logDebug("CHIPBE=default... trying OpenCL Backend");
      Backend = new CHIPBackendOpenCL();
    }
#endif
    if (!Backend) {
      CHIPERR_LOG_AND_THROW("Could not initialize any backend.",
                            hipErrorInitializationError);
    }
  } else {
    CHIPERR_LOG_AND_THROW(
        "Invalid CHIP-SPV Backend Selected. Accepted values : level0, opencl.",
        hipErrorInitializationError);
  }
}

void CHIPInitializeCallOnce() {
  CHIPReadEnvVars();
  logDebug("CHIPDriver Initialize");

  createBackendObject();

  Backend->initialize(CHIPPlatformStr, CHIPDeviceTypeStr, CHIPDeviceStr);
}

extern void CHIPInitialize() {
  std::call_once(Initialized, &CHIPInitializeCallOnce);
}

void CHIPUninitializeCallOnce() {
  logDebug("Uninitializing CHIP...");
  if (Backend) {
    Backend->uninitialize();
    delete Backend;
    Backend = nullptr;
  }
}

extern void CHIPUninitialize() {
  std::call_once(Uninitialized, &CHIPUninitializeCallOnce);
}

extern hipError_t CHIPReinitialize(const uintptr_t *NativeHandles,
                                   int NumHandles) {
  CHIPReadEnvVars();
  logDebug("CHIPDriver REInitialize");

  // Kernel compilation already took place so we need save these modules and
  // pass them to re-initialization function
  auto Modules = Backend->getActiveDevice()->getModules();

  if (Backend) {
    logDebug("uninitializing existing Backend object.");
    Backend->uninitialize();
    delete Backend;
    Backend = nullptr;
  }

  createBackendObject();

  int RequiredHandles = Backend->ReqNumHandles();
  if (RequiredHandles != NumHandles) {
    delete Backend;
    Backend = nullptr;
    CHIPERR_LOG_AND_THROW("Invalid number of native handles",
                          hipErrorInitializationError);
  }

  Backend->initializeFromNative(NativeHandles, NumHandles);
  for (auto ModulePair : Modules) {
    Backend->getActiveDevice()->addModule(ModulePair.first, ModulePair.second);
  }

  return hipSuccess;
}
