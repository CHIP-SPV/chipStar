/*
 * Copyright (c) 2021-23 chipStar developers
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
 * @file CHIPBindings.hh
 * @author Paulius Velesko (pvelesko@pglc.io)
 * @brief Implementations of the HIP API functions using the CHIP interface
 * providing basic functionality such hipMemcpy, host and device function
 * registration, hipLaunchByPtr, etc.
 * These functions operate on base CHIP class pointers allowing for backend
 * selection at runtime and backend-specific implementations are done by
 * inheriting from base CHIP classes and overriding virtual member functions.
 * @version 0.1
 * @date 2023-02-02
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef CHIP_HIPCTX_H
#define CHIP_HIPCTX_H

#include "hip/hip_runtime_api.h"

hipError_t hipCtxCreate(hipCtx_t *ctx, unsigned int flags, hipDevice_t device) {
  CHIP_TRY
  CHIPInitialize();

  ERROR_CHECK_DEVNUM(device);

  auto ChipCtx = Backend->getDevices()[device]->getContext();
  ChipCtx->retain();
  *ctx = ChipCtx;

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipCtxDestroy(hipCtx_t ctx) {
  CHIP_TRY
  CHIPInitialize();
  auto ChipCtx = static_cast<chipstar::Context *>(ctx);
  if (ChipCtx == nullptr) {
    RETURN(hipErrorInvalidValue);
  }

  // Need to remove the ctx of calling thread if its the top one
  if (!Backend->ChipCtxStack.empty() &&
      Backend->ChipCtxStack.top() == ChipCtx) {
    Backend->ChipCtxStack.pop();
  }

  // decrease refcount and delete if 0
  ChipCtx->release();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipCtxPopCurrent(hipCtx_t *ctx) {
  CHIP_TRY
  CHIPInitialize();

  if (Backend->ChipCtxStack.empty()) {
    *ctx = nullptr;
  } else {
    *ctx = Backend->ChipCtxStack.top();
    Backend->ChipCtxStack.pop();
  }

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipCtxPushCurrent(hipCtx_t ctx) {
  CHIP_TRY
  CHIPInitialize();

  auto ChipCtx = static_cast<chipstar::Context *>(ctx);
  if (!ChipCtx) {
    if (!Backend->ChipCtxStack.empty()) {
      Backend->ChipCtxStack.pop();
    }
  } else {
    Backend->ChipCtxStack.push(ChipCtx);
  }

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipCtxSetCurrent(hipCtx_t ctx) {
  CHIP_TRY
  CHIPInitialize();
  Backend->setActiveContext(static_cast<chipstar::Context *>(ctx));
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipCtxGetCurrent(hipCtx_t *ctx) {
  CHIP_TRY
  CHIPInitialize();
  *ctx = Backend->getActiveContext();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipCtxGetDevice(hipDevice_t *device) {
  CHIP_TRY
  CHIPInitialize();
  *device = Backend->getActiveContext()->getDevice()->getDeviceId();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipCtxGetApiVersion(hipCtx_t ctx, int *apiVersion) {
  // Unimplemented in hipamd
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t hipCtxGetCacheConfig(hipFuncCache_t *cacheConfig) {
  // Unimplemented in hipamd
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig) {
  // Unimplemented in hipamd
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config) {
  // Unimplemented in hipamd
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig *pConfig) {
  // Unimplemented in hipamd
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t hipCtxSynchronize(void) {
  CHIP_TRY
  CHIPInitialize();
  auto Dev = Backend->getActiveDevice();
  LOCK(Dev->QueueAddRemoveMtx);
  for (auto &Q : Dev->getQueuesNoLock()) {
    Q->finish();
  }
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipCtxGetFlags(unsigned int *flags) {
  // Unimplemented in hipamd
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags) {
  // hipamd implementation
  RETURN(hipSuccess);
}

hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx) {
  // hipamd implementation
  RETURN(hipSuccess);
}

hipError_t hipDevicePrimaryCtxRelease(hipDevice_t Device) {

  // hipamd implementation
  // HIP_INIT_API(hipDevicePrimaryCtxRelease, dev);

  // if (static_cast<unsigned int>(dev) >= g_devices.size()) {
  //   HIP_RETURN(hipErrorInvalidDevice);
  // }

  // HIP_RETURN(hipSuccess);

  CHIP_TRY
  CHIPInitialize();

  ERROR_CHECK_DEVNUM(Device);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDevicePrimaryCtxRetain(hipCtx_t *Context, hipDevice_t Device) {
  // hipamd implementation
  // HIP_INIT_API(hipDevicePrimaryCtxRetain, pctx, dev);

  // if (static_cast<unsigned int>(dev) >= g_devices.size()) {
  //   HIP_RETURN(hipErrorInvalidDevice);
  // }
  // if (pctx == nullptr) {
  //   HIP_RETURN(hipErrorInvalidValue);
  // }

  // *pctx = reinterpret_cast<hipCtx_t>(g_devices[dev]);

  // HIP_RETURN(hipSuccess);

  CHIP_TRY
  CHIPInitialize();

  NULLCHECK(Context);
  ERROR_CHECK_DEVNUM(Device);

  *Context = Backend->getDevices()[Device]->getContext();
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipDevicePrimaryCtxReset(hipDevice_t Device) {
  CHIP_TRY
  CHIPInitialize();

  ERROR_CHECK_DEVNUM(Device);

  Backend->getDevices()[Device]->getContext()->reset();

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t Device, unsigned int Flags) {
  CHIP_TRY
  CHIPInitialize();

  if (static_cast<unsigned int>(Device) >= Backend->getDevices().size()) {
    RETURN(hipErrorInvalidDevice);
  } else {
    RETURN(hipErrorContextAlreadyInUse);
  }

  CHIP_CATCH
}

hipError_t hipDevicePrimaryCtxGetState(hipDevice_t Device, unsigned int *Flags,
                                       int *Active) {
  CHIP_TRY
  CHIPInitialize();

  if (static_cast<unsigned int>(Device) >= Backend->getDevices().size()) {
    RETURN(hipErrorInvalidDevice);
  }

  if (Flags != nullptr) {
    *Flags = 0;
  }

  if (Active != nullptr) {
    auto ActiveDev = Backend->getActiveDevice();
    auto TestDev = Backend->getDevices()[Device];
    *Active = ActiveDev == TestDev;
  }

  RETURN(hipSuccess);
  CHIP_CATCH
}

#endif // CHIP_HIPCTX_H
