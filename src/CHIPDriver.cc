/*
 * Copyright (c) 2021-22 chipStar developers
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
#include "Utils.hh"

std::once_flag Initialized;
std::once_flag EnvInitialized;
std::once_flag Uninitialized;
bool UsingDefaultBackend;
chipstar::Backend *Backend = nullptr;
std::atomic_ulong CHIPNumRegisteredFatBinaries;

EnvVars ChipEnvVars;

// CUDA Driver API: "If cuInit() has not been called, any function
// from the driver API will return CUDA_ERROR_NOT_INITIALIZED".
//
// CUDA Programming manual v12.1: "The runtime maintains an error
// variable for each host thread that is initialized to cudaSuccess
// ..."
//
// HIP API is mostly based on the CUDA runtime API so follow the
// programming manual.
thread_local hipError_t CHIPTlsLastError = hipSuccess;

// Uninitializes the backend when the application exits.
void __attribute__((destructor)) uninitializeBackend() {
  // Generally, __hipUnregisterFatBinary would uninitialize the
  // backend when it finishes unregistration of all modules. However,
  // there won't be hip(Un)registerFatBinary() calls if the HIP
  // program does not have embedded kernels. This makes sure we
  // uninitialize the backend at exit.
  if (Backend && CHIPNumRegisteredFatBinaries == 0)
    CHIPUninitialize();
}

static void createBackendObject() {
  assert(Backend == nullptr);

  if (ChipEnvVars.Backend.getType() == BackendType::OPENCL) {
#ifdef HAVE_OPENCL
    logDebug("CHIPBE=OPENCL... Initializing OpenCL Backend");
    Backend = new CHIPBackendOpenCL();
#else
    CHIPERR_LOG_AND_THROW("Invalid chipStar Backend Selected. This chipStar "
                          "was not compiled with OpenCL backend",
                          hipErrorInitializationError);
#endif
  } else if (ChipEnvVars.Backend.getType() == BackendType::LEVEL0) {
#ifdef HAVE_LEVEL0
    logDebug("CHIPBE=LEVEL0... Initializing Level0 Backend");
    Backend = new CHIPBackendLevel0();
#else
    CHIPERR_LOG_AND_THROW("Invalid chipStar Backend Selected. This chipStar "
                          "was not compiled with Level0 backend",
                          hipErrorInitializationError);
#endif
  } else if (ChipEnvVars.Backend.getType() == BackendType::DEFAULT) {
#ifdef HAVE_OPENCL
    if (!Backend) {
      logDebug("CHIPBE=default... trying OpenCL Backend");
      Backend = new CHIPBackendOpenCL();
    }
#endif
#ifdef HAVE_LEVEL0
    if (!Backend) {
      logDebug("CHIPBE=default... trying Level0 Backend");
      Backend = new CHIPBackendLevel0();
    }
#endif
    if (!Backend) {
      CHIPERR_LOG_AND_THROW("Could not initialize any backend.",
                            hipErrorInitializationError);
    }
  }
}

void CHIPInitializeCallOnce() {
  logDebug("CHIPDriver Initialize");

  createBackendObject();

  Backend->initialize();
}

extern void CHIPInitialize() {
  std::call_once(Initialized, &CHIPInitializeCallOnce);
}

void CHIPUninitializeCallOnce() {
  logDebug("Uninitializing CHIP...");
  if (Backend) {
    if (getSPVRegister().getNumSources()) {
      logWarn("Program still has unloaded HIP modules at program exit.");
      logInfo("Unloaded module count: {}", getSPVRegister().getNumSources());
    }
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
  logDebug("CHIPDriver REInitialize");

  // chipstar::Kernel compilation may have already taken place so we need save
  // these modules and pass them to re-initialization function
  auto ModuleState = Backend->getActiveDevice()->getModuleState();

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
  Backend->getActiveDevice()->addFromModuleState(ModuleState);

  return hipSuccess;
}

const char *CHIPGetBackendName() { return ChipEnvVars.Backend.str(); }
