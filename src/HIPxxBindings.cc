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
#include "hip/hip_fatbin.h"
#include "hip/hip.hh"
#include "temporary.hh"

#define SPIR_TRIPLE "hip-spir64-unknown-unknown"

static unsigned binaries_loaded = 0;

hipError_t hipEventCreate(hipEvent_t *event) {
  return hipEventCreateWithFlags(event, 0);
}

class ClEvent;
hipError_t hipEventCreate(ClEvent **event) {
  // logWarn("hipEventCreate not implemented");
  return hipSuccess;
  // return hipEventCreateWithFlags(event, 0);
}

hipError_t hipFree(void *ptr) {
  logWarn("hipFree not yet implemented");
  return hipSuccess;
  // LZ_TRY
  // ERROR_IF((ptr == nullptr), hipSuccess);

  // LZContext *cont = getTlsDefaultLzCtx();
  // ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  // if (cont->free(ptr))
  //   RETURN(hipSuccess);
  // else
  //   RETURN(hipErrorInvalidDevicePointer);
  // LZ_CATCH
  RETURN(hipSuccess);
};

hipError_t hipLaunchByPtr(const void *hostFunction) {
  HIPxxInitialize();
  logTrace("hipLaunchByPtr");
  HIPxxExecItem *exec_item = Backend->hipxx_execstack.top();
  Backend->hipxx_execstack.pop();

  if (exec_item->launchByHostPtr(hostFunction))
    RETURN(hipSuccess);
  else
    RETURN(hipErrorLaunchFailure);
}

hipError_t hipGetDeviceProperties(hipDeviceProp_t *prop, int deviceId) {
  logTrace("hipGetDeviceProperties");
  HIPxxInitialize();
  std::vector<HIPxxDevice *> devices =
      Backend->get_default_context()->get_devices();
  if (deviceId > devices.size() - 1) {
    logCritical(
        "hipGetDeviceProperties requested a deviceId {} greater than number of "
        "devices {}",
        deviceId, devices.size());
    std::abort();
  }
  // TODO WHYY??!!?!?
  // devices[deviceId]->populate_device_properties();
  devices[deviceId]->copy_device_properties(prop);
  logTrace("done hipGetDeviceProperties");

  return (hipSuccess);
}

hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes,
                     hipMemcpyKind kind) {
  logTrace("hipMemcpy");
  HIPxxInitialize();

  HIPxxContext *ctx = Backend->get_default_context();
  ERROR_IF((ctx == nullptr), hipErrorInvalidDevice);

  if (kind == hipMemcpyHostToHost) {
    memcpy(dst, src, sizeBytes);
    RETURN(hipSuccess);
  } else {
    RETURN(ctx->memCopy(dst, src, sizeBytes, nullptr));
  }
  // LZ_CATCH
  // ze_result_t status = zeCommandQueueSynchronize(cont->hQueue, UINT64_MAX);
  // if (status != ZE_RESULT_SUCCESS) {
  // 	  throw InvalidLevel0Initialization("HipLZ zeCommandQueueSynchronize
  // FAILED with return code " + std::to_string(status));
  // }
  RETURN(hipSuccess);
}

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  // logWarn("hipEventRecord not yet implemented");
  // HIPLZ_INIT();

  // LZ_TRY
  // ERROR_IF((event == nullptr), hipErrorInvalidValue);

  // LZContext* cont = getTlsDefaultLzCtx();
  // ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  // RETURN(cont->recordEvent(stream, event));

  // LZ_CATCH
  RETURN(hipSuccess);
}

class ClQueue;
hipError_t hipEventRecord(ClEvent *, ClQueue *) {
  // logWarn("hipEventRecord not yet implemented");
  RETURN(hipSuccess);
}

hipError_t hipConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                            hipStream_t stream) {
  HIPxxInitialize();
  logTrace("hipConfigureCall()");
  RETURN(Backend->configure_call(gridDim, blockDim, sharedMem, stream));
  RETURN(hipSuccess);
}

hipError_t hipDeviceSynchronize(void) {
  // logWarn("hipDeviceSynchronize not yet implemented");
  // HIPLZ_INIT();

  // LZ_TRY

  // LZContext *cont = getTlsDefaultLzCtx();
  // ERROR_IF((cont == nullptr), hipErrorInvalidDevice);
  // // Synchronize among HipLZ queues
  // cont->finishAll();
  // RETURN(hipSuccess);

  // LZ_CATCH
  RETURN(hipSuccess);
}

hipError_t hipEventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop) {
  // logWarn("hipEventElapsedTime not yet implemented");
  // HIPLZ_INIT();

  // LZ_TRY
  // ERROR_IF((start == nullptr), hipErrorInvalidValue);
  // ERROR_IF((stop == nullptr), hipErrorInvalidValue);

  // LZContext* cont = getTlsDefaultLzCtx();
  // ERROR_IF((cont == nullptr), hipErrorInvalidDevice);

  // RETURN(cont->eventElapsedTime(ms, start, stop));
  // LZ_CATCH
  RETURN(hipSuccess);
}

hipError_t hipEventElapsedTime(float *, ClEvent *, ClEvent *) {
  // logWarn("hipEventElapsedTime not yet implemented");
  RETURN(hipSuccess);
}

hipError_t hipEventDestroy(ClEvent *) {
  // logWarn("hipEventDestroy not yet implemented");
  RETURN(hipSuccess);
}

hipError_t hipEventDestroy(hipEvent_t event) {
  // logWarn("hipEventDestroy not yet implemented");
  // HIPLZ_INIT();

  // ERROR_IF((event == nullptr), hipErrorInvalidValue);

  // delete event;
  RETURN(hipSuccess);
}

hipError_t hipGetLastError(void) {
  // logWarn("hipGetLastError not yet implemented");
  // HIPLZ_INIT();

  // hipError_t temp = tls_LastError;
  // tls_LastError = hipSuccess;
  // return temp;
  RETURN(hipSuccess);
}

hipError_t hipEventCreateWithFlags(hipEvent_t *event, unsigned flags) {
  // logWarn("hipEventCreateWithFlags not yet implemented");
  HIPxxInitialize();

  ERROR_IF((event == nullptr), hipErrorInvalidValue);

  // hipEvent_t EventPtr = cont->createEvent(flags);
  // TODO
  // hipEvent_t EventPtr;
  // if (EventPtr) {
  //  *event = EventPtr;
  //  RETURN(hipSuccess);
  //} else {
  //  RETURN(hipErrorOutOfMemory);
  //}
  return (hipSuccess);
}

extern "C" void **__hipRegisterFatBinary(const void *data) {
  HIPxxInitialize();
  logTrace("__hipRegisterFatBinary");

  const __CudaFatBinaryWrapper *fbwrapper =
      reinterpret_cast<const __CudaFatBinaryWrapper *>(data);
  if (fbwrapper->magic != __hipFatMAGIC2 || fbwrapper->version != 1) {
    logCritical("The given object is not hipFatBinary !\n");
    std::abort();
  }

  const __ClangOffloadBundleHeader *header = fbwrapper->binary;
  std::string magic(reinterpret_cast<const char *>(header),
                    sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1);
  if (magic.compare(CLANG_OFFLOAD_BUNDLER_MAGIC)) {
    logCritical(
        "The bundled binaries are not Clang bundled "
        "(CLANG_OFFLOAD_BUNDLER_MAGIC is missing)\n");
    std::abort();
  }

  std::string *module = new std::string;
  if (!module) {
    logCritical("Failed to allocate memory\n");
    std::abort();
  }

  const __ClangOffloadBundleDesc *desc = &header->desc[0];
  bool found = false;

  for (uint64_t i = 0; i < header->numBundles;
       ++i, desc = reinterpret_cast<const __ClangOffloadBundleDesc *>(
                reinterpret_cast<uintptr_t>(&desc->triple[0]) +
                desc->tripleSize)) {
    std::string triple{&desc->triple[0], sizeof(SPIR_TRIPLE) - 1};
    logDebug("Triple of bundle {} is: {}\n", i, triple);

    if (triple.compare(SPIR_TRIPLE) == 0) {
      found = true;
      break;
    } else {
      logDebug("not a SPIR triple, ignoring\n");
      continue;
    }
  }

  if (!found) {
    logDebug("Didn't find any suitable compiled binary!\n");
    std::abort();
  }

  const char *string_data = reinterpret_cast<const char *>(
      reinterpret_cast<uintptr_t>(header) + (uintptr_t)desc->offset);
  size_t string_size = desc->size;
  module->assign(string_data, string_size);

  logDebug("Register module: {} \n", (void *)module);

  Backend->register_module(module);

  ++binaries_loaded;

  return (void **)module;
}

extern "C" void __hipUnregisterFatBinary(void *data) {
  std::string *module = reinterpret_cast<std::string *>(data);

  logDebug("Unregister module: {} \n", (void *)module);
  Backend->unregister_module(module);

  --binaries_loaded;
  logDebug("__hipUnRegisterFatBinary {}\n", binaries_loaded);

  if (binaries_loaded == 0) {
    HIPxxUninitialize();
  }

  delete module;
}

extern "C" void __hipRegisterFunction(void **data, const void *hostFunction,
                                      char *deviceFunction,
                                      const char *deviceName,
                                      unsigned int threadLimit, void *tid,
                                      void *bid, dim3 *blockDim, dim3 *gridDim,
                                      int *wSize) {
  HIPxxInitialize();
  std::string *module_str = reinterpret_cast<std::string *>(data);

  std::string devFunc = deviceFunction;
  logDebug("RegisterFunction on module {}\n", (void *)module_str);

  logDebug("RegisterFunction on {} devices", Backend->get_num_devices());
  Backend->register_function_as_kernel(module_str, hostFunction, deviceName);
}

hipError_t hipSetupArgument(const void *arg, size_t size, size_t offset) {
  logTrace("hipSetupArgument");
  HIPxxInitialize();
  RETURN(Backend->set_arg(arg, size, offset));
  return hipSuccess;
}

hipError_t hipMalloc(void **ptr, size_t size) {
  logTrace("hipMalloc");
  HIPxxInitialize();

  ERROR_IF((ptr == nullptr), hipErrorInvalidValue);

  if (size == 0) {
    *ptr = nullptr;
    RETURN(hipSuccess);
  }
  // TODO Can we have multiple contexts on one backend? Should allocations take
  // place on all existing contexts?
  void *retval = Backend->get_default_context()->allocate(size);
  ERROR_IF((retval == nullptr), hipErrorMemoryAllocation);

  *ptr = retval;
  return hipSuccess;
}

#endif