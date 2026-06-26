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

/// Prevent more than one HIP API call from executing at the same time
std::mutex ApiMtx;

#include <string>
#include <memory>
#include <cstdlib>

#include "backend/backends.hh"
#include "Utils.hh"

std::once_flag Initialized;
std::once_flag EnvInitialized;
std::once_flag Uninitialized;
bool UsingDefaultBackend;
chipstar::Backend *Backend = nullptr;
std::atomic_ulong CHIPNumRegisteredFatBinaries;

EnvVars ChipEnvVars;

// Opt-in: allow a single allocation larger than the device's reported
// maxMemAllocSize (e.g. the 4 GiB cap on Intel Arc / Data Center GPUs) to
// succeed when sufficient device memory is available. Controlled by
// CHIP_UNRESTRICTED_ALLOC_SIZE=1; OFF by default.
//
// This applies to both backends equally:
//   - chipStar's own getMaxMallocSize() guard is bypassed (CHIPBackend.cc).
//   - OpenCL: NEO requires AllowUnrestrictedSize=1, which is only honored when
//     NEOReadDebugKeys=1 (set here, before any driver call).
//   - Level Zero: the device allocation is made with the
//     ze_relaxed_allocation_limits extension (CHIPBackendLevel0.cc).
//
// It is OFF by default because the OpenCL path is not free: NEOReadDebugKeys=1
// switches the NEO driver into debug-key mode globally, which makes high
// register-pressure / high-scratch kernels (e.g. unoptimized -O0 kernels) fail
// at execution with CL_OUT_OF_RESOURCES even when they run fine without it.
// That regression is far more common than the need for >4 GiB single
// allocations, so we do not enable it unless explicitly requested.
bool chipUnrestrictedAllocSize() {
  static const bool Enabled = [] {
    const char *Opt = std::getenv("CHIP_UNRESTRICTED_ALLOC_SIZE");
    return Opt && Opt[0] == '1';
  }();
  return Enabled;
}

// NEO debug keys must be set before the OpenCL driver is loaded, so this runs
// in a priority-101 constructor. setenv(..., overwrite=0) preserves any
// user-provided override.
__attribute__((constructor(101))) static void chipEnableNeoLargeAlloc() {
  if (chipUnrestrictedAllocSize()) {
    setenv("NEOReadDebugKeys", "1", /*overwrite=*/0);
    setenv("AllowUnrestrictedSize", "1", /*overwrite=*/0);
  }
}

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

  if (ChipEnvVars.getBackend().getType() == BackendType::OpenCL) {
#ifdef HAVE_OPENCL
    logDebug("CHIPBE=OPENCL... Initializing OpenCL Backend");
    Backend = new CHIPBackendOpenCL();
#else
    CHIPERR_LOG_AND_THROW("Invalid chipStar Backend Selected. This chipStar "
                          "was not compiled with OpenCL backend",
                          hipErrorInitializationError);
#endif
  } else if (ChipEnvVars.getBackend().getType() == BackendType::Level0) {
#ifdef HAVE_LEVEL0
    logDebug("CHIPBE=LEVEL0... Initializing Level0 Backend");
    Backend = new CHIPBackendLevel0();
#else
    CHIPERR_LOG_AND_THROW("Invalid chipStar Backend Selected. This chipStar "
                          "was not compiled with Level0 backend",
                          hipErrorInitializationError);
#endif
  } else if (ChipEnvVars.getBackend().getType() == BackendType::Default) {
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

  if (ChipEnvVars.getBackend().getType() == BackendType::Default) {
    // Default mode: try each compiled backend until one initializes.
#ifdef HAVE_OPENCL
    try {
      logDebug("CHIPBE=default... trying OpenCL Backend");
      Backend = new CHIPBackendOpenCL();
      Backend->initialize();
      return;
    } catch (...) {
      logDebug("OpenCL backend failed to initialize");
      if (Backend) delete Backend, Backend = nullptr;
    }
#endif
#ifdef HAVE_LEVEL0
    try {
      logDebug("CHIPBE=default... trying Level0 Backend");
      Backend = new CHIPBackendLevel0();
      Backend->initialize();
      return;
    } catch (...) {
      logDebug("Level0 backend failed to initialize");
      if (Backend) delete Backend, Backend = nullptr;
    }
#endif
    CHIPERR_LOG_AND_THROW(
        "No backend could be initialized. Tried all compiled backends.",
        hipErrorInitializationError);
  }

  try {
    createBackendObject();
    Backend->initialize();
  } catch (...) {
    if (Backend) delete Backend, Backend = nullptr;
    // Throw (don't abort) so callers that tolerate a missing device can recover.
    // __hipRegisterFatBinary() deliberately swallows this to allow test
    // discovery (e.g. Catch2 --list-tests) on machines without a GPU; an abort
    // here escapes that catch and kills discovery. Real HIP API calls still
    // surface the failure as hipErrorInitializationError on the next call.
    CHIPERR_LOG_AND_THROW("Backend initialization failed. No device available.",
                          hipErrorInitializationError);
  }
}

extern void CHIPInitialize() {
  std::call_once(Initialized, &CHIPInitializeCallOnce);
}

void CHIPUninitializeCallOnce() {
  logDebug("Uninitializing CHIP...");
  if (ChipEnvVars.getSkipUninit()) {
    logWarn("Uninitialization skipped");
    return;
  }
  if (Backend) {
    if (getSPVRegister().getNumSources()) {
      logWarn("Program still has unloaded HIP modules at program exit.");
      logInfo("Unloaded module count: {}", getSPVRegister().getNumSources());
    }

    // Synchronize all devices before cleanup to ensure any pending operations
    // (especially JIT compilation) complete before tearing down resources
    for (auto Dev : Backend->getDevices()) {
      // Sync all user queues
      {
        std::lock_guard<std::mutex> Lock(Dev->QueueAddRemoveMtx);
        for (auto Q : Dev->getQueuesNoLock()) {
          Q->finish();
        }
      }
      // Sync default queues
      auto LegacyQueue = Dev->getLegacyDefaultQueue();
      if (LegacyQueue) {
        LegacyQueue->finish();
      }
      if (Dev->isPerThreadStreamUsed()) {
        auto PerThreadQueue = Dev->getPerThreadDefaultQueue();
        if (PerThreadQueue) {
          PerThreadQueue->finish();
        }
      }
    }

    // call deallocateDeviceVariables on all devices.

    for (auto Dev : Backend->getDevices()) {
      Dev->deallocateDeviceVariables();
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
  Backend->setReinitializeFlag(true);

  return hipSuccess;
}
