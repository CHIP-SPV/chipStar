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
#include "macros.hh"

#define SPIR_TRIPLE "hip-spir64-unknown-unknown"

static unsigned binaries_loaded = 0;

//*****************************************************************************
//*****************************************************************************
//*****************************************************************************
hipError_t hipGetDevice(int *deviceId) {
  HIPxxInitialize();

  ERROR_IF((deviceId == nullptr),
           hipErrorInvalidValue);  // Check API compliance

  HIPxxDevice *dev = Backend->getActiveDevice();
  *deviceId = dev->getDeviceId();

  RETURN(hipSuccess);
}

hipError_t hipGetDeviceCount(int *count) {
  HIPxxInitialize();
  ERROR_IF((count == nullptr), hipErrorInvalidValue);  // Check API compliance
  *count = Backend->getNumDevices();

  RETURN(hipSuccess);
}

hipError_t hipSetDevice(int deviceId) {
  HIPxxInitialize();

  ERROR_CHECK_DEVNUM(deviceId);

  HIPxxDevice *selected_device = Backend->getDevices()[deviceId];
  Backend->setActiveDevice(selected_device);

  RETURN(hipSuccess);
}

hipError_t hipDeviceSynchronize(void) {
  HIPxxInitialize();

  HIPxxContext *ctx = Backend->getActiveContext();
  ERROR_IF((ctx == nullptr), hipErrorInvalidDevice);
  // Synchronize among HipLZ queues
  ctx->finishAll();

  RETURN(hipSuccess);
}

hipError_t hipDeviceReset(void) {
  HIPxxInitialize();

  HIPxxDevice *dev = Backend->getActiveDevice();
  ERROR_IF((dev == nullptr), hipErrorInvalidDevice);

  dev->reset();
  RETURN(hipSuccess);
}

hipError_t hipDeviceGet(hipDevice_t *device, int ordinal) {
  HIPxxInitialize();

  ERROR_IF((device == nullptr), hipErrorInvalidDevice);
  ERROR_CHECK_DEVNUM(ordinal);

  **device = Backend->getDevices()[ordinal];
  RETURN(hipSuccess);
}

hipError_t hipDeviceComputeCapability(int *major, int *minor,
                                      hipDevice_t device) {
  HIPxxInitialize();

  ERROR_CHECK_DEVHANDLE(*device);

  hipDeviceProp_t props;
  (*device)->copyDeviceProperties(&props);

  if (major) *major = props.major;
  if (minor) *minor = props.minor;

  RETURN(hipSuccess);
}

hipError_t hipDeviceGetAttribute(int *pi, hipDeviceAttribute_t attr,
                                 int deviceId) {
  HIPxxInitialize();
  ERROR_CHECK_DEVNUM(deviceId);

  if (Backend->getDevices()[deviceId]->getAttr(pi, attr))
    RETURN(hipErrorInvalidValue);
  else
    RETURN(hipSuccess);
}

hipError_t hipGetDeviceProperties(hipDeviceProp_t *prop, int deviceId) {
  HIPxxInitialize();
  ERROR_CHECK_DEVNUM(deviceId);
  Backend->getDevices()[deviceId]->copyDeviceProperties(prop);

  RETURN(hipSuccess);
}

hipError_t hipDeviceGetLimit(size_t *pValue, enum hipLimit_t limit) {
  HIPxxInitialize();
  ERROR_IF((pValue == nullptr), hipErrorInvalidValue);
  switch (limit) {
    case hipLimitMallocHeapSize:
      *pValue = 0;  // TODO Get this from properties
      break;
    default:
      RETURN(hipErrorUnsupportedLimit);
  }
  RETURN(hipSuccess);
}

hipError_t hipDeviceGetName(char *name, int len, hipDevice_t device) {
  HIPxxInitialize();
  ERROR_CHECK_DEVHANDLE(*device);
  std::string dev_name = (*device)->getName();

  size_t namelen = dev_name.size();
  namelen = (namelen < (size_t)len ? namelen : len - 1);
  memcpy(name, dev_name.data(), namelen);
  name[namelen] = 0;
  RETURN(hipSuccess);
}

hipError_t hipDeviceTotalMem(size_t *bytes, hipDevice_t device) {
  HIPxxInitialize();
  ERROR_CHECK_DEVHANDLE(*device);
  // TODO why did this not throw error if passed in should I do this check: ?
  // if (bytes == nullptr) {
  //  logCritical(
  //      "hipDeviceTotalMem was passed a null pointer for returning size");
  //  std::abort();
  //}

  if (bytes) *bytes = (*device)->getGlobalMemSize();
  RETURN(hipSuccess);
}

hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig) {
  HIPxxInitialize();
  Backend->getActiveDevice()->setCacheConfig(cacheConfig);

  RETURN(hipSuccess);
}

hipError_t hipDeviceGetCacheConfig(hipFuncCache_t *cacheConfig) {
  HIPxxInitialize();

  if (cacheConfig) *cacheConfig = Backend->getActiveDevice()->getCacheConfig();
  RETURN(hipSuccess);
}

hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig *pConfig) {
  HIPxxInitialize();
  if (pConfig) *pConfig = Backend->getActiveDevice()->getSharedMemConfig();
  RETURN(hipSuccess);
}

hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig pConfig) {
  HIPxxInitialize();
  Backend->getActiveDevice()->setSharedMemConfig(pConfig);
  RETURN(hipSuccess);
}

hipError_t hipFuncSetCacheConfig(const void *func, hipFuncCache_t config) {
  HIPxxInitialize();
  RETURN(hipSuccess);
}

hipError_t hipDeviceGetPCIBusId(char *pciBusId, int len, int deviceId) {
  HIPxxInitialize();

  ERROR_CHECK_DEVNUM(deviceId);
  HIPxxDevice *dev = Backend->getDevices()[deviceId];

  hipDeviceProp_t prop;
  dev->copyDeviceProperties(&prop);
  snprintf(pciBusId, len, "%04x:%04x:%04x", prop.pciDomainID, prop.pciBusID,
           prop.pciDeviceID);
  RETURN(hipSuccess);
}

hipError_t hipDeviceGetByPCIBusId(int *deviceId, const char *pciBusId) {
  HIPxxInitialize();

  int pciDomainID, pciBusID, pciDeviceID;
  int err =
      sscanf(pciBusId, "%4x:%4x:%4x", &pciDomainID, &pciBusID, &pciDeviceID);
  if (err == EOF || err < 3) RETURN(hipErrorInvalidValue);
  for (size_t i = 0; i < Backend->getNumDevices(); i++) {
    HIPxxDevice *dev = Backend->getDevices()[i];
    if (dev->hasPCIBusId(pciDomainID, pciBusID, pciDeviceID)) {
      *deviceId = i;
      RETURN(hipSuccess);
    }
  }

  RETURN(hipErrorInvalidDevice);
}

hipError_t hipSetDeviceFlags(unsigned flags) {
  // TODO
  HIPxxInitialize();
  logCritical("hipSetDeviceFlags not yet implemented");
  RETURN(hipSuccess);
}

hipError_t hipDeviceCanAccessPeer(int *canAccessPeer, int deviceId,
                                  int peerDeviceId) {
  HIPxxInitialize();

  ERROR_CHECK_DEVNUM(deviceId);
  ERROR_CHECK_DEVNUM(peerDeviceId);
  if (deviceId == peerDeviceId) {
    *canAccessPeer = 0;
    RETURN(hipSuccess);
  }

  HIPxxDevice *dev = Backend->getDevices()[deviceId];
  HIPxxDevice *peer = Backend->getDevices()[peerDeviceId];

  *canAccessPeer = dev->getPeerAccess(peer);

  RETURN(hipSuccess);
}

hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags) {
  HIPxxInitialize();

  HIPxxDevice *dev = Backend->getActiveDevice();
  HIPxxDevice *peer = Backend->getDevices()[peerDeviceId];

  RETURN(dev->setPeerAccess(peer, flags, true));
}

hipError_t hipDeviceDisablePeerAccess(int peerDeviceId) {
  HIPxxInitialize();

  HIPxxDevice *dev = Backend->getActiveDevice();
  HIPxxDevice *peer = Backend->getDevices()[peerDeviceId];

  RETURN(dev->setPeerAccess(peer, 0, false));
}

hipError_t hipChooseDevice(int *deviceId, const hipDeviceProp_t *prop) {
  HIPxxInitialize();

  HIPxxDevice *dev = Backend->findDeviceMatchingProps(prop);
  if (!dev) RETURN(hipErrorInvalidValue);

  *deviceId = dev->getDeviceId();

  RETURN(hipSuccess);
}

hipError_t hipDriverGetVersion(int *driverVersion) {
  HIPxxInitialize();

  if (driverVersion) {
    *driverVersion = 4;
    RETURN(hipSuccess);
  } else
    RETURN(hipErrorInvalidValue);
}

hipError_t hipRuntimeGetVersion(int *runtimeVersion) {
  HIPxxInitialize();

  if (runtimeVersion) {
    *runtimeVersion = 1;
    RETURN(hipSuccess);
  } else
    RETURN(hipErrorInvalidValue);
}

hipError_t hipGetLastError(void) {
  HIPxxInitialize();

  hipError_t temp = Backend->tls_last_error;
  Backend->tls_last_error = hipSuccess;
  return temp;
}

hipError_t hipPeekAtLastError(void) {
  HIPxxInitialize();

  return Backend->tls_last_error;
}

const char *hipGetErrorName(hipError_t hip_error) {
  switch (hip_error) {
    case hipSuccess:
      return "hipSuccess";
    case hipErrorOutOfMemory:
      return "hipErrorOutOfMemory";
    case hipErrorNotInitialized:
      return "hipErrorNotInitialized";
    case hipErrorDeinitialized:
      return "hipErrorDeinitialized";
    case hipErrorProfilerDisabled:
      return "hipErrorProfilerDisabled";
    case hipErrorProfilerNotInitialized:
      return "hipErrorProfilerNotInitialized";
    case hipErrorProfilerAlreadyStarted:
      return "hipErrorProfilerAlreadyStarted";
    case hipErrorProfilerAlreadyStopped:
      return "hipErrorProfilerAlreadyStopped";
    case hipErrorInvalidImage:
      return "hipErrorInvalidImage";
    case hipErrorInvalidContext:
      return "hipErrorInvalidContext";
    case hipErrorContextAlreadyCurrent:
      return "hipErrorContextAlreadyCurrent";
    case hipErrorMapFailed:
      return "hipErrorMapFailed";
    case hipErrorUnmapFailed:
      return "hipErrorUnmapFailed";
    case hipErrorArrayIsMapped:
      return "hipErrorArrayIsMapped";
    case hipErrorAlreadyMapped:
      return "hipErrorAlreadyMapped";
    case hipErrorNoBinaryForGpu:
      return "hipErrorNoBinaryForGpu";
    case hipErrorAlreadyAcquired:
      return "hipErrorAlreadyAcquired";
    case hipErrorNotMapped:
      return "hipErrorNotMapped";
    case hipErrorNotMappedAsArray:
      return "hipErrorNotMappedAsArray";
    case hipErrorNotMappedAsPointer:
      return "hipErrorNotMappedAsPointer";
    case hipErrorECCNotCorrectable:
      return "hipErrorECCNotCorrectable";
    case hipErrorUnsupportedLimit:
      return "hipErrorUnsupportedLimit";
    case hipErrorContextAlreadyInUse:
      return "hipErrorContextAlreadyInUse";
    case hipErrorPeerAccessUnsupported:
      return "hipErrorPeerAccessUnsupported";
    case hipErrorInvalidKernelFile:
      return "hipErrorInvalidKernelFile";
    case hipErrorInvalidGraphicsContext:
      return "hipErrorInvalidGraphicsContext";
    case hipErrorInvalidSource:
      return "hipErrorInvalidSource";
    case hipErrorFileNotFound:
      return "hipErrorFileNotFound";
    case hipErrorSharedObjectSymbolNotFound:
      return "hipErrorSharedObjectSymbolNotFound";
    case hipErrorSharedObjectInitFailed:
      return "hipErrorSharedObjectInitFailed";
    case hipErrorOperatingSystem:
      return "hipErrorOperatingSystem";
    case hipErrorSetOnActiveProcess:
      return "hipErrorSetOnActiveProcess";
    case hipErrorInvalidHandle:
      return "hipErrorInvalidHandle";
    case hipErrorNotFound:
      return "hipErrorNotFound";
    case hipErrorIllegalAddress:
      return "hipErrorIllegalAddress";
    case hipErrorInvalidSymbol:
      return "hipErrorInvalidSymbol";
    case hipErrorMissingConfiguration:
      return "hipErrorMissingConfiguration";
    case hipErrorMemoryAllocation:
      return "hipErrorMemoryAllocation";
    case hipErrorInitializationError:
      return "hipErrorInitializationError";
    case hipErrorLaunchFailure:
      return "hipErrorLaunchFailure";
    case hipErrorPriorLaunchFailure:
      return "hipErrorPriorLaunchFailure";
    case hipErrorLaunchTimeOut:
      return "hipErrorLaunchTimeOut";
    case hipErrorLaunchOutOfResources:
      return "hipErrorLaunchOutOfResources";
    case hipErrorInvalidDeviceFunction:
      return "hipErrorInvalidDeviceFunction";
    case hipErrorInvalidConfiguration:
      return "hipErrorInvalidConfiguration";
    case hipErrorInvalidDevice:
      return "hipErrorInvalidDevice";
    case hipErrorInvalidValue:
      return "hipErrorInvalidValue";
    case hipErrorInvalidDevicePointer:
      return "hipErrorInvalidDevicePointer";
    case hipErrorInvalidMemcpyDirection:
      return "hipErrorInvalidMemcpyDirection";
    case hipErrorUnknown:
      return "hipErrorUnknown";
    case hipErrorInvalidResourceHandle:
      return "hipErrorInvalidResourceHandle";
    case hipErrorNotReady:
      return "hipErrorNotReady";
    case hipErrorNoDevice:
      return "hipErrorNoDevice";
    case hipErrorPeerAccessAlreadyEnabled:
      return "hipErrorPeerAccessAlreadyEnabled";
    case hipErrorNotSupported:
      return "hipErrorNotSupported";
    case hipErrorPeerAccessNotEnabled:
      return "hipErrorPeerAccessNotEnabled";
    case hipErrorRuntimeMemory:
      return "hipErrorRuntimeMemory";
    case hipErrorRuntimeOther:
      return "hipErrorRuntimeOther";
    case hipErrorHostMemoryAlreadyRegistered:
      return "hipErrorHostMemoryAlreadyRegistered";
    case hipErrorHostMemoryNotRegistered:
      return "hipErrorHostMemoryNotRegistered";
    case hipErrorTbd:
      return "hipErrorTbd";
    default:
      return "hipErrorUnknown";
  }
}

const char *hipGetErrorString(hipError_t hipError) {
  return hipGetErrorName(hipError);
}

hipError_t hipStreamCreate(hipStream_t *stream) {
  return hipStreamCreateWithFlags(stream, 0);
}

hipError_t hipStreamCreateWithFlags(hipStream_t *stream, unsigned int flags) {
  return hipStreamCreateWithPriority(stream, flags, 0);
}

hipError_t hipStreamCreateWithPriority(hipStream_t *stream, unsigned int flags,
                                       int priority) {
  HIPxxInitialize();

  ERROR_IF((stream == nullptr), hipErrorInvalidResourceHandle);

  HIPxxDevice *dev = Backend->getActiveDevice();
  ERROR_IF((dev == nullptr), hipErrorInvalidDevice);

  *stream = dev->addQueue(flags, priority);

  if (stream)
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);
}

hipError_t hipDeviceGetStreamPriorityRange(int *leastPriority,
                                           int *greatestPriority) {
  HIPxxQueue *q = Backend->getActiveQueue();

  if (leastPriority) *leastPriority = q->getPriorityRange(0);
  if (greatestPriority) *greatestPriority = q->getPriorityRange(1);
  RETURN(hipSuccess);
}

hipError_t hipStreamDestroy(hipStream_t stream) {
  HIPxxInitialize();
  ERROR_IF((stream == nullptr), hipErrorInvalidResourceHandle);

  HIPxxDevice *dev = Backend->getActiveDevice();
  ERROR_IF((dev == nullptr), hipErrorInvalidDevice);

  if (dev->removeQueue(stream))
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);
}

hipError_t hipStreamQuery(hipStream_t stream) {
  HIPxxInitialize();
  ERROR_IF((stream == nullptr), hipErrorInvalidResourceHandle);

  if (stream->query()) {
    RETURN(hipSuccess);
  } else {
    RETURN(hipErrorNotReady);
  }
}

hipError_t hipStreamSynchronize(hipStream_t stream) {
  HIPxxInitialize();
  ERROR_IF((stream == nullptr), hipErrorInvalidResourceHandle);
  stream->finish();
  RETURN(hipSuccess);
}

hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event,
                              unsigned int flags) {
  HIPxxInitialize();

  ERROR_IF((stream == nullptr), hipErrorInvalidResourceHandle);
  ERROR_IF((event == nullptr), hipErrorInvalidResourceHandle);

  if (stream->enqueueBarrierForEvent(event))
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);
}

hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int *flags) {
  HIPxxInitialize();
  ERROR_IF((stream == nullptr), hipErrorInvalidResourceHandle);
  ERROR_IF((flags == nullptr), hipErrorInvalidResourceHandle);

  *flags = stream->getFlags();
  RETURN(hipSuccess);
}

hipError_t hipStreamGetPriority(hipStream_t stream, int *priority) {
  HIPxxInitialize();
  ERROR_IF((stream == nullptr), hipErrorInvalidResourceHandle);
  ERROR_IF((priority == nullptr), hipErrorInvalidResourceHandle);

  *priority = stream->getPriority();
  RETURN(hipSuccess);
}

hipError_t hipStreamAddCallback(hipStream_t stream,
                                hipStreamCallback_t callback, void *userData,
                                unsigned int flags) {
  HIPxxInitialize();
  ERROR_IF((stream == nullptr), hipErrorInvalidResourceHandle);
  ERROR_IF((callback == nullptr), hipErrorInvalidResourceHandle);

  if (stream->addCallback(callback, userData))
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);
}

hipError_t hipMemGetAddressRange(hipDeviceptr_t *pbase, size_t *psize,
                                 hipDeviceptr_t dptr) {
  HIPxxInitialize();
  HIPxxContext *ctx = Backend->getActiveContext();

  if (ctx->findPointerInfo(pbase, psize, dptr))
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);
}

hipError_t hipDevicePrimaryCtxGetState(hipDevice_t device, unsigned int *flags,
                                       int *active) {
  HIPxxInitialize();
  ERROR_CHECK_DEVHANDLE(*device);

  ERROR_IF((flags == nullptr || active == nullptr), hipErrorInvalidValue);

  HIPxxContext *currentCtx = Backend->getActiveContext();

  // Currently device only has 1 context
  HIPxxContext *primaryCtx = (*device)->getContext();

  *active = (primaryCtx == currentCtx) ? 1 : 0;
  *flags = primaryCtx->getFlags();

  RETURN(hipSuccess);
}

hipError_t hipDevicePrimaryCtxRelease(hipDevice_t device) {
  HIPxxInitialize();
  ERROR_CHECK_DEVHANDLE(*device);
  RETURN(hipSuccess);
}

hipError_t hipDevicePrimaryCtxRetain(hipCtx_t *pctx, hipDevice_t device) {
  HIPxxInitialize();
  ERROR_CHECK_DEVHANDLE(*device);
  *pctx = (*device)->getContext()->retain();
  RETURN(hipSuccess);
}

hipError_t hipDevicePrimaryCtxReset(hipDevice_t device) {
  HIPxxInitialize();
  ERROR_CHECK_DEVHANDLE(*device);
  (*device)->getContext()->reset();

  RETURN(hipSuccess);
}

hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t device, unsigned int flags) {
  HIPxxInitialize();
  ERROR_CHECK_DEVHANDLE(*device);

  if ((*device)->getContext()->setFlags(flags)) {
    RETURN(hipSuccess);
  } else {
    RETURN(hipErrorContextAlreadyInUse);
  }
}

hipError_t hipEventCreate(hipEvent_t *event) {
  return hipEventCreateWithFlags(event, 0);
}

hipError_t hipEventCreateWithFlags(hipEvent_t *event, unsigned flags) {
  HIPxxInitialize();

  ERROR_IF((event == nullptr), hipErrorInvalidValue);

  HIPxxEvent *event_ptr = new HIPxxEvent(Backend->getActiveContext(), flags);
  if (event_ptr) {
    *event = event_ptr;
    RETURN(hipSuccess);
  } else {
    RETURN(hipErrorOutOfMemory);
  }
}

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  HIPxxInitialize();
  ERROR_IF((stream == nullptr), hipErrorInvalidValue);
  ERROR_IF((event == nullptr), hipErrorInvalidValue);

  if (Backend->getActiveContext()->recordEvent(stream, event)) {
    RETURN(hipSuccess);
  } else {
    RETURN(hipErrorLaunchFailure);
  }
}

hipError_t hipEventDestroy(hipEvent_t event) {
  HIPxxInitialize();
  ERROR_IF((event == nullptr), hipErrorInvalidValue);

  delete event;
  RETURN(hipSuccess);
}

hipError_t hipEventSynchronize(hipEvent_t event) {
  HIPxxInitialize();
  ERROR_IF((event == nullptr), hipErrorInvalidValue);

  if (event->wait())
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);
}

hipError_t hipEventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop) {
  HIPxxInitialize();
  ERROR_IF((start == nullptr), hipErrorInvalidValue);
  ERROR_IF((stop == nullptr), hipErrorInvalidValue);

  *ms = start->getElapsedTime(stop);
  RETURN(hipSuccess);
}

hipError_t hipEventQuery(hipEvent_t event) {
  HIPxxInitialize();
  ERROR_IF((event == nullptr), hipErrorInvalidValue);

  if (event->isFinished())
    RETURN(hipSuccess);
  else
    RETURN(hipErrorNotReady);
}

//*****************************************************************************
//*****************************************************************************
//*****************************************************************************

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

hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes,
                     hipMemcpyKind kind) {
  logTrace("hipMemcpy");
  HIPxxInitialize();

  HIPxxContext *ctx = Backend->getActiveContext();
  ERROR_IF((ctx == nullptr), hipErrorInvalidDevice);

  if (kind == hipMemcpyHostToHost) {
    memcpy(dst, src, sizeBytes);
    RETURN(hipSuccess);
  } else {
    RETURN(ctx->memCopy(dst, src, sizeBytes, nullptr));
  }
  RETURN(hipSuccess);
}

hipError_t hipConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                            hipStream_t stream) {
  HIPxxInitialize();
  logTrace("hipConfigureCall()");
  RETURN(Backend->configureCall(gridDim, blockDim, sharedMem, stream));
  RETURN(hipSuccess);
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

  Backend->registerModuleStr(module);

  ++binaries_loaded;

  return (void **)module;
}

extern "C" void __hipUnregisterFatBinary(void *data) {
  std::string *module = reinterpret_cast<std::string *>(data);

  logDebug("Unregister module: {} \n", (void *)module);
  Backend->unregisterModuleStr(module);

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

  logDebug("RegisterFunction on {} devices", Backend->getNumDevices());
  Backend->registerFunctionAsKernel(module_str, hostFunction, deviceName);
}

hipError_t hipSetupArgument(const void *arg, size_t size, size_t offset) {
  logTrace("hipSetupArgument");
  HIPxxInitialize();
  RETURN(Backend->setArg(arg, size, offset));
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
  void *retval = Backend->getActiveContext()->allocate(size);
  ERROR_IF((retval == nullptr), hipErrorMemoryAllocation);

  *ptr = retval;
  return hipSuccess;
}

hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags) {
  logCritical("hipHostMalloc not yet implemented");
  std::abort();
  // HIPLZ_INIT();

  // LZ_TRY
  // LZContext *cont = getTlsDefaultLzCtx();
  // ERROR_IF((cont == nullptr), hipErrorInvalidDevice);
  // *ptr = cont->allocate(size, 0x1000, ClMemoryType::Shared);
  // LZ_CATCH
  // RETURN(hipSuccess);
}

#endif