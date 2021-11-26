/**
 * @file CHIPBindings.hh
 * @author Paulius Velesko (pvelesko@gmail.com)
 * @brief Implementations of the HIP API functions using the CHIP interface
 * providing basic functionality such hipMemcpy, host and device function
 * registration, hipLaunchByPtr, etc.
 * These functions operate on base CHIP class pointers allowing for backend
 * selection at runtime and backend-specific implementations are done by
 * inheriting from base CHIP classes and overriding virtual member functions.
 * @version 0.1
 * @date 2021-08-19
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef CHIP_BINDINGS_H
#define CHIP_BINDINGS_H

#include <fstream>

#include "CHIPBackend.hh"
#include "CHIPDriver.hh"
#include "hip/hip_fatbin.h"
#include "hip/hip_runtime_api.h"
#include "macros.hh"
#include "CHIPException.hh"

#include "backend/backends.hh"

#define SPIR_TRIPLE "hip-spir64-unknown-unknown"

static unsigned binaries_loaded = 0;

#define SVM_ALIGNMENT 128  // TODO Pass as CMAKE Define?

hipError_t hipMemcpy2DFromArray(void *dst, size_t dpitch, hipArray_const_t src,
                                size_t wOffset, size_t hOffset, size_t width,
                                size_t height, hipMemcpyKind kind) {
  UNIMPLEMENTED(hipErrorUnknown);
}
hipError_t hipMemsetD16Async(hipDeviceptr_t dest, unsigned short value,
                             size_t count, hipStream_t stream) {
  UNIMPLEMENTED(hipErrorUnknown);
}
hipError_t hipMemcpy2DToArrayAsync(hipArray *dst, size_t wOffset,
                                   size_t hOffset, const void *src,
                                   size_t spitch, size_t width, size_t height,
                                   hipMemcpyKind kind, hipStream_t stream) {
  UNIMPLEMENTED(hipErrorUnknown);
};
hipError_t hipMemcpy3DAsync(const struct hipMemcpy3DParms *p,
                            hipStream_t stream) {
  UNIMPLEMENTED(hipErrorUnknown);
}
hipError_t hipMemcpyWithStream(void *dst, const void *src, size_t sizeBytes,
                               hipMemcpyKind kind, hipStream_t stream) {
  UNIMPLEMENTED(hipErrorUnknown);
};
hipError_t hipMemsetD16(hipDeviceptr_t dest, unsigned short value,
                        size_t count) {
  UNIMPLEMENTED(hipErrorUnknown);
};
hipError_t hipMemcpyPeer(void *dst, int dstDeviceId, const void *src,
                         int srcDeviceId, size_t sizeBytes) {
  UNIMPLEMENTED(hipErrorUnknown);
};
hipError_t hipMemRangeGetAttribute(void *data, size_t data_size,
                                   hipMemRangeAttribute attribute,
                                   const void *dev_ptr, size_t count) {
  UNIMPLEMENTED(hipErrorUnknown);
};
hipError_t hipMemcpy2DFromArrayAsync(void *dst, size_t dpitch,
                                     hipArray_const_t src, size_t wOffset,
                                     size_t hOffset, size_t width,
                                     size_t height, hipMemcpyKind kind,
                                     hipStream_t stream) {
  UNIMPLEMENTED(hipErrorUnknown);
};
hipError_t hipMallocManaged(void **dev_ptr, size_t size, unsigned int flags) {
  UNIMPLEMENTED(hipErrorUnknown);
};
hipError_t hipMalloc3DArray(hipArray **array,
                            const struct hipChannelFormatDesc *desc,
                            struct hipExtent extent, unsigned int flags) {
  UNIMPLEMENTED(hipErrorUnknown);
};
hipError_t hipMemsetD8Async(hipDeviceptr_t dest, unsigned char value,
                            size_t count, hipStream_t stream) {
  UNIMPLEMENTED(hipErrorUnknown);
};
hipError_t hipMemcpyPeerAsync(void *dst, int dstDeviceId, const void *src,
                              int srcDevice, size_t sizeBytes,
                              hipStream_t stream) {
  UNIMPLEMENTED(hipErrorUnknown);
};
hipError_t hipMemcpyParam2DAsync(const hip_Memcpy2D *pCopy,
                                 hipStream_t stream) {
  UNIMPLEMENTED(hipErrorUnknown);
}
hipError_t __hipPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                      size_t sharedMem, hipStream_t stream) {
  CHIP_TRY
  CHIPInitialize();

  if (!stream) stream = Backend->getActiveQueue();
  logTrace("__hipPushCallConfiguration()");
  RETURN(Backend->configureCall(gridDim, blockDim, sharedMem, stream));
  CHIP_CATCH
  RETURN(hipSuccess);
}

hipError_t __hipPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                     size_t *sharedMem, hipStream_t *stream) {
  logTrace("__hipPopCallConfiguration()");
  CHIP_TRY
  CHIPInitialize();
  // if (!stream) *stream = Backend->getActiveQueue();

  auto *ei = Backend->chip_execstack.top();
  *gridDim = ei->getGrid();
  *blockDim = ei->getBlock();
  *sharedMem = ei->getSharedMem();
  *stream = ei->getQueue();
  Backend->chip_execstack.pop();
  RETURN(hipSuccess);
  CHIP_CATCH
}

//*****************************************************************************
//*****************************************************************************
//*****************************************************************************
hipError_t hipGetDevice(int *deviceId) {
  CHIP_TRY
  CHIPInitialize();

  ERROR_IF((deviceId == nullptr),
           hipErrorInvalidValue);  // Check API compliance

  CHIPDevice *dev = Backend->getActiveDevice();
  *deviceId = dev->getDeviceId();

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGetDeviceCount(int *count) {
  CHIP_TRY
  CHIPInitialize();
  ERROR_IF((count == nullptr), hipErrorInvalidValue);  // Check API compliance
  *count = Backend->getNumDevices();

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipSetDevice(int deviceId) {
  CHIP_TRY
  CHIPInitialize();

  ERROR_CHECK_DEVNUM(deviceId);

  CHIPDevice *selected_device = Backend->getDevices()[deviceId];
  Backend->setActiveDevice(selected_device);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceSynchronize(void) {
  CHIP_TRY
  CHIPInitialize();

  CHIPContext *ctx = Backend->getActiveContext();
  ERROR_IF((ctx == nullptr), hipErrorInvalidDevice);
  // Synchronize among HipLZ queues
  ctx->finishAll();

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceReset(void) {
  CHIP_TRY
  CHIPInitialize();

  CHIPDevice *dev = Backend->getActiveDevice();
  ERROR_IF((dev == nullptr), hipErrorInvalidDevice);

  dev->reset();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceGet(hipDevice_t *device, int ordinal) {
  CHIP_TRY
  CHIPInitialize();

  ERROR_IF((device == nullptr), hipErrorInvalidDevice);
  ERROR_CHECK_DEVNUM(ordinal);

  //**device = Backend->getDevices()[ordinal];
  *device = ordinal;
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceComputeCapability(int *major, int *minor,
                                      hipDevice_t device) {
  CHIP_TRY
  CHIPInitialize();

  ERROR_CHECK_DEVNUM(device);

  hipDeviceProp_t props;
  Backend->getDevices()[device]->copyDeviceProperties(&props);

  if (major) *major = props.major;
  if (minor) *minor = props.minor;

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceGetAttribute(int *pi, hipDeviceAttribute_t attr,
                                 int deviceId) {
  CHIP_TRY
  CHIPInitialize();
  ERROR_CHECK_DEVNUM(deviceId);

  *pi = Backend->getDevices()[deviceId]->getAttr(attr);
  if (*pi == -1)
    RETURN(hipErrorInvalidValue);
  else
    RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGetDeviceProperties(hipDeviceProp_t *prop, int deviceId) {
  CHIP_TRY
  CHIPInitialize();
  ERROR_CHECK_DEVNUM(deviceId);
  Backend->getDevices()[deviceId]->copyDeviceProperties(prop);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceGetLimit(size_t *pValue, enum hipLimit_t limit) {
  UNIMPLEMENTED(hipSuccess);
  //  CHIP_TRY
  //  CHIPInitialize();
  //  ERROR_IF((pValue == nullptr), hipErrorInvalidValue);
  //  switch (limit) {
  //    case hipLimitMallocHeapSize:
  //      *pValue = 0;  // TODO Get this from properties
  //      /* zeinfo reports this as
  //      Maximum memory allocation size 4294959104
  //      */
  //      break;
  //    case hipLimitPrintfFifoSize:
  //      break;
  //    default:
  //      RETURN(hipErrorUnsupportedLimit);
  //  }
  //  RETURN(hipSuccess);
  //  CHIP_CATCH
}

hipError_t hipDeviceGetName(char *name, int len, hipDevice_t device) {
  CHIP_TRY
  CHIPInitialize();
  ERROR_CHECK_DEVNUM(device);
  std::string dev_name = (Backend->getDevices()[device])->getName();

  size_t namelen = dev_name.size();
  namelen = (namelen < (size_t)len ? namelen : len - 1);
  memcpy(name, dev_name.data(), namelen);
  name[namelen] = 0;
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceTotalMem(size_t *bytes, hipDevice_t device) {
  CHIP_TRY
  CHIPInitialize();
  ERROR_CHECK_DEVNUM(device);
  if (bytes == nullptr) {
    CHIPERR_LOG_AND_THROW(
        "hipDeviceTotalMem was passed a null pointer for returning size",
        hipErrorInvalidValue);
  }

  if (bytes) *bytes = (Backend->getDevices()[device])->getGlobalMemSize();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig) {
  CHIP_TRY
  CHIPInitialize();
  Backend->getActiveDevice()->setCacheConfig(cacheConfig);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceGetCacheConfig(hipFuncCache_t *cacheConfig) {
  CHIP_TRY
  CHIPInitialize();

  if (cacheConfig) *cacheConfig = Backend->getActiveDevice()->getCacheConfig();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig *pConfig) {
  CHIP_TRY
  CHIPInitialize();
  if (pConfig) *pConfig = Backend->getActiveDevice()->getSharedMemConfig();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig pConfig) {
  CHIP_TRY
  CHIPInitialize();
  Backend->getActiveDevice()->setSharedMemConfig(pConfig);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipFuncSetCacheConfig(const void *func, hipFuncCache_t config) {
  CHIP_TRY
  CHIPInitialize();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceGetPCIBusId(char *pciBusId, int len, int deviceId) {
  CHIP_TRY
  CHIPInitialize();

  ERROR_CHECK_DEVNUM(deviceId);
  CHIPDevice *dev = Backend->getDevices()[deviceId];

  hipDeviceProp_t prop;
  dev->copyDeviceProperties(&prop);
  snprintf(pciBusId, len, "%04x:%04x:%04x", prop.pciDomainID, prop.pciBusID,
           prop.pciDeviceID);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceGetByPCIBusId(int *deviceId, const char *pciBusId) {
  CHIP_TRY
  CHIPInitialize();

  int pciDomainID, pciBusID, pciDeviceID;
  int err =
      sscanf(pciBusId, "%4x:%4x:%4x", &pciDomainID, &pciBusID, &pciDeviceID);
  if (err == EOF || err < 3) RETURN(hipErrorInvalidValue);
  for (size_t i = 0; i < Backend->getNumDevices(); i++) {
    CHIPDevice *dev = Backend->getDevices()[i];
    if (dev->hasPCIBusId(pciDomainID, pciBusID, pciDeviceID)) {
      *deviceId = i;
      RETURN(hipSuccess);
    }
  }

  RETURN(hipErrorInvalidDevice);
  CHIP_CATCH
}

hipError_t hipSetDeviceFlags(unsigned flags) {
  CHIP_TRY
  CHIPInitialize();
  UNIMPLEMENTED(hipErrorUnknown);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceCanAccessPeer(int *canAccessPeer, int deviceId,
                                  int peerDeviceId) {
  CHIP_TRY
  CHIPInitialize();

  ERROR_CHECK_DEVNUM(deviceId);
  ERROR_CHECK_DEVNUM(peerDeviceId);
  if (deviceId == peerDeviceId) {
    *canAccessPeer = 0;
    RETURN(hipSuccess);
  }

  CHIPDevice *dev = Backend->getDevices()[deviceId];
  CHIPDevice *peer = Backend->getDevices()[peerDeviceId];

  *canAccessPeer = dev->getPeerAccess(peer);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags) {
  CHIP_TRY
  CHIPInitialize();

  CHIPDevice *dev = Backend->getActiveDevice();
  CHIPDevice *peer = Backend->getDevices()[peerDeviceId];

  RETURN(dev->setPeerAccess(peer, flags, true));
  CHIP_CATCH
}

hipError_t hipDeviceDisablePeerAccess(int peerDeviceId) {
  CHIP_TRY
  CHIPInitialize();

  CHIPDevice *dev = Backend->getActiveDevice();
  CHIPDevice *peer = Backend->getDevices()[peerDeviceId];

  RETURN(dev->setPeerAccess(peer, 0, false));
  CHIP_CATCH
}

hipError_t hipChooseDevice(int *deviceId, const hipDeviceProp_t *prop) {
  CHIP_TRY
  CHIPInitialize();

  CHIPDevice *dev = Backend->findDeviceMatchingProps(prop);
  if (!dev) RETURN(hipErrorInvalidValue);

  *deviceId = dev->getDeviceId();

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDriverGetVersion(int *driverVersion) {
  CHIP_TRY
  CHIPInitialize();

  if (driverVersion) {
    *driverVersion = 4;
    RETURN(hipSuccess);
  } else
    RETURN(hipErrorInvalidValue);
  CHIP_CATCH
}

hipError_t hipRuntimeGetVersion(int *runtimeVersion) {
  CHIP_TRY
  CHIPInitialize();

  if (runtimeVersion) {
    *runtimeVersion = 1;
    RETURN(hipSuccess);
  } else
    RETURN(hipErrorInvalidValue);

  CHIP_CATCH
}

hipError_t hipGetLastError(void) {
  CHIPInitialize();

  hipError_t temp = Backend->tls_last_error;
  Backend->tls_last_error = hipSuccess;
  return temp;
}

hipError_t hipPeekAtLastError(void) {
  CHIPInitialize();

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
    // case hipErrorMemoryAllocation:
    //   return "hipErrorMemoryAllocation";
    // case hipErrorInitializationError:
    //   return "hipErrorInitializationError";
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
    // case hipErrorInvalidResourceHandle:
    //   return "hipErrorInvalidResourceHandle";
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
  RETURN(hipStreamCreateWithFlags(stream, 0));
}

hipError_t hipStreamCreateWithFlags(hipStream_t *stream, unsigned int flags) {
  RETURN(hipStreamCreateWithPriority(stream, flags, 0));
}

hipError_t hipStreamCreateWithPriority(hipStream_t *stream, unsigned int flags,
                                       int priority) {
  CHIP_TRY
  CHIPInitialize();

  ERROR_IF((stream == nullptr), hipErrorInvalidResourceHandle);

  CHIPDevice *dev = Backend->getActiveDevice();
  ERROR_IF((dev == nullptr), hipErrorInvalidDevice);

  dev->addQueue(flags, priority);

  if (stream)
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);
  CHIP_CATCH
}

hipError_t hipDeviceGetStreamPriorityRange(int *leastPriority,
                                           int *greatestPriority) {
  CHIP_TRY
  CHIPInitialize();

  CHIPQueue *q = Backend->getActiveQueue();

  if (leastPriority) *leastPriority = q->getPriorityRange(0);
  if (greatestPriority) *greatestPriority = q->getPriorityRange(1);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipStreamDestroy(hipStream_t stream) {
  CHIP_TRY
  CHIPInitialize();
  ERROR_IF((stream == nullptr), hipErrorInvalidResourceHandle);

  CHIPDevice *dev = Backend->getActiveDevice();
  ERROR_IF((dev == nullptr), hipErrorInvalidDevice);

  if (dev->removeQueue(stream))
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);
  CHIP_CATCH
}

hipError_t hipStreamQuery(hipStream_t stream) {
  CHIP_TRY
  CHIPInitialize();
  ERROR_IF((stream == nullptr), hipErrorInvalidResourceHandle);

  if (stream->query()) {
    RETURN(hipSuccess);
  } else
    RETURN(hipErrorNotReady);
  CHIP_CATCH
}

hipError_t hipStreamSynchronize(hipStream_t stream) {
  CHIP_TRY
  CHIPInitialize();
  // ERROR_IF((stream == nullptr), hipErrorInvalidResourceHandle);
  ERROR_IF((Backend->findQueue(stream) == nullptr),
           hipErrorInvalidResourceHandle);
  Backend->findQueue(stream)->finish();
  // TODO
  // stream->finish();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event,
                              unsigned int flags) {
  CHIP_TRY
  CHIPInitialize();

  ERROR_IF((stream == nullptr), hipErrorInvalidResourceHandle);
  ERROR_IF((event == nullptr), hipErrorInvalidResourceHandle);

  if (stream->enqueueBarrierForEvent(event))
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);
  CHIP_CATCH
}

hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int *flags) {
  CHIP_TRY
  CHIPInitialize();
  ERROR_IF((stream == nullptr), hipErrorInvalidResourceHandle);
  ERROR_IF((flags == nullptr), hipErrorInvalidResourceHandle);

  *flags = stream->getFlags();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipStreamGetPriority(hipStream_t stream, int *priority) {
  CHIP_TRY
  CHIPInitialize();
  ERROR_IF((stream == nullptr), hipErrorInvalidResourceHandle);
  ERROR_IF((priority == nullptr), hipErrorInvalidResourceHandle);

  *priority = stream->getPriority();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipStreamAddCallback(hipStream_t stream,
                                hipStreamCallback_t callback, void *userData,
                                unsigned int flags) {
  CHIP_TRY
  CHIPInitialize();
  ERROR_IF((stream == nullptr), hipErrorInvalidResourceHandle);
  ERROR_IF((callback == nullptr), hipErrorInvalidResourceHandle);

  if (stream->addCallback(callback, userData))
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);
  CHIP_CATCH
}

hipError_t hipMemGetAddressRange(hipDeviceptr_t *pbase, size_t *psize,
                                 hipDeviceptr_t dptr) {
  CHIP_TRY
  CHIPInitialize();
  CHIPContext *ctx = Backend->getActiveContext();

  if (ctx->findPointerInfo(pbase, psize, dptr))
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);
  CHIP_CATCH
}

hipError_t hipDevicePrimaryCtxGetState(hipDevice_t device, unsigned int *flags,
                                       int *active) {
  CHIP_TRY
  CHIPInitialize();
  ERROR_CHECK_DEVNUM(device);

  ERROR_IF((flags == nullptr || active == nullptr), hipErrorInvalidValue);

  CHIPContext *currentCtx = Backend->getActiveContext();

  // Currently device only has 1 context
  CHIPContext *primaryCtx = (Backend->getDevices()[device])->getContext();

  *active = (primaryCtx == currentCtx) ? 1 : 0;
  *flags = primaryCtx->getFlags();

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDevicePrimaryCtxRelease(hipDevice_t device) {
  CHIP_TRY
  CHIPInitialize();
  ERROR_CHECK_DEVNUM(device);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDevicePrimaryCtxRetain(hipCtx_t *pctx, hipDevice_t device) {
  CHIP_TRY
  CHIPInitialize();
  ERROR_CHECK_DEVNUM(device);
  *pctx = (Backend->getDevices()[device])->getContext()->retain();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDevicePrimaryCtxReset(hipDevice_t device) {
  CHIP_TRY
  CHIPInitialize();
  ERROR_CHECK_DEVNUM(device);
  (Backend->getDevices()[device])->getContext()->reset();

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t device, unsigned int flags) {
  CHIP_TRY
  CHIPInitialize();
  ERROR_CHECK_DEVNUM(device);

  (Backend->getDevices()[device])->getContext()->setFlags(flags);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipEventCreate(hipEvent_t *event) {
  RETURN(hipEventCreateWithFlags(event, 0));
}

hipError_t hipEventCreateWithFlags(hipEvent_t *event, unsigned flags) {
  CHIP_TRY
  CHIPInitialize();
  ERROR_IF((event == nullptr), hipErrorInvalidValue);

  *event = Backend->getActiveContext()->createEvent(flags);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  CHIP_TRY
  CHIPInitialize();
  if (!stream) stream = Backend->getActiveQueue();
  // TODO: Why does this check fail for OpenCL but not for Level0
  // ERROR_IF((event == nullptr), hipErrorInvalidValue);

  event->recordStream(stream);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipEventDestroy(hipEvent_t event) {
  CHIP_TRY
  CHIPInitialize();
  ERROR_IF((event == nullptr), hipErrorInvalidValue);

  delete event;
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipEventSynchronize(hipEvent_t event) {
  CHIP_TRY
  CHIPInitialize();
  ERROR_IF((event == nullptr), hipErrorInvalidValue);

  if (event->wait())
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);
  CHIP_CATCH
}

hipError_t hipEventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop) {
  CHIP_TRY
  CHIPInitialize();
  ERROR_IF((start == nullptr), hipErrorInvalidValue);
  ERROR_IF((stop == nullptr), hipErrorInvalidValue);

  *ms = start->getElapsedTime(stop);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipEventQuery(hipEvent_t event) {
  CHIP_TRY
  CHIPInitialize();
  ERROR_IF((event == nullptr), hipErrorInvalidValue);

  if (event->isFinished())
    RETURN(hipSuccess);
  else
    RETURN(hipErrorNotReady);
  CHIP_CATCH
}

hipError_t hipMalloc(void **ptr, size_t size) {
  CHIP_TRY
  CHIPInitialize();

  ERROR_IF((ptr == nullptr), hipErrorInvalidValue);

  if (size == 0) {
    *ptr = nullptr;
    RETURN(hipSuccess);
  }
  void *retval =
      Backend->getActiveContext()->allocate(size, CHIPMemoryType::Device);
  ERROR_IF((retval == nullptr), hipErrorMemoryAllocation);

  *ptr = retval;
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMallocManaged(void **ptr, size_t size) {
  CHIP_TRY
  CHIPInitialize();
  ERROR_IF((ptr == nullptr), hipErrorInvalidValue);

  if (size == 0) {
    *ptr = nullptr;
    RETURN(hipSuccess);
  }

  void *retval =
      Backend->getActiveContext()->allocate(size, CHIPMemoryType::Shared);
  ERROR_IF((retval == nullptr), hipErrorMemoryAllocation);

  *ptr = retval;
  RETURN(hipSuccess);
  CHIP_CATCH
}

DEPRECATED("use hipHostMalloc instead")
hipError_t hipMallocHost(void **ptr, size_t size) {
  RETURN(hipMalloc(ptr, size));
}

hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags) {
  CHIP_TRY
  CHIPInitialize();

  void *retval =
      Backend->getActiveContext()->allocate(size, 0x1000, CHIPMemoryType::Host);
  ERROR_IF((retval == nullptr), hipErrorMemoryAllocation);

  *ptr = retval;
  RETURN(hipSuccess);
  CHIP_CATCH
}

DEPRECATED("use hipHostMalloc instead")
hipError_t hipHostAlloc(void **ptr, size_t size, unsigned int flags) {
  RETURN(hipMalloc(ptr, size));
}

hipError_t hipFree(void *ptr) {
  CHIP_TRY
  CHIPInitialize();
  ERROR_IF((ptr == nullptr), hipSuccess);

  RETURN(Backend->getActiveContext()->free(ptr));
  CHIP_CATCH
}

hipError_t hipHostFree(void *ptr) {
  CHIP_TRY
  CHIPInitialize();
  RETURN(hipFree(ptr));
  CHIP_CATCH
}

DEPRECATED("use hipHostFree instead")
hipError_t hipFreeHost(void *ptr) { RETURN(hipHostFree(ptr)); }

hipError_t hipMemPrefetchAsync(const void *ptr, size_t count, int dstDevId,
                               hipStream_t stream) {
  CHIP_TRY
  CHIPInitialize();
  ERROR_IF((ptr == nullptr), hipErrorInvalidValue);

  ERROR_CHECK_DEVNUM(dstDevId);
  CHIPDevice *dev = Backend->getDevices()[dstDevId];
  CHIPContext *ctx = dev->getContext();

  // Check if given stream belongs to the requested device
  if (stream != nullptr)
    ERROR_IF(stream->getDevice() != dev, hipErrorInvalidDevice);

  bool retval = stream->memPrefetch(ptr, count);  // TODO Error Check

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemAdvise(const void *ptr, size_t count, hipMemoryAdvise advice,
                        int dstDevId) {
  CHIP_TRY
  CHIPInitialize();

  ERROR_IF((ptr == nullptr), hipErrorInvalidValue);

  if (ptr == 0 || count == 0) {
    RETURN(hipSuccess);
  }

  UNIMPLEMENTED(hipErrorUnknown);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipHostGetDevicePointer(void **devPtr, void *hstPtr,
                                   unsigned int flags) {
  CHIP_TRY
  CHIPInitialize();

  ERROR_IF(((hstPtr == nullptr) || (devPtr == nullptr)), hipErrorInvalidValue);

  UNIMPLEMENTED(hipErrorUnknown);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipHostGetFlags(unsigned int *flagsPtr, void *hostPtr) {
  CHIP_TRY
  CHIPInitialize();

  UNIMPLEMENTED(hipErrorUnknown);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipHostRegister(void *hostPtr, size_t sizeBytes,
                           unsigned int flags) {
  CHIP_TRY
  CHIPInitialize();
  UNIMPLEMENTED(hipErrorUnknown);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipHostUnregister(void *hostPtr) {
  CHIP_TRY
  CHIPInitialize();
  UNIMPLEMENTED(hipErrorUnknown);
  RETURN(hipSuccess);
  CHIP_CATCH
}

static hipError_t hipMallocPitch3D(void **ptr, size_t *pitch, size_t width,
                                   size_t height, size_t depth) {
  CHIP_TRY
  CHIPInitialize();

  ERROR_IF((ptr == nullptr), hipErrorInvalidValue);

  *pitch = ((((int)width - 1) / SVM_ALIGNMENT) + 1) * SVM_ALIGNMENT;
  const size_t sizeBytes = (*pitch) * height * ((depth == 0) ? 1 : depth);

  void *retval = Backend->getActiveContext()->allocate(sizeBytes);
  ERROR_IF((retval == nullptr), hipErrorMemoryAllocation);

  *ptr = retval;
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMallocPitch(void **ptr, size_t *pitch, size_t width,
                          size_t height) {
  CHIP_TRY
  CHIPInitialize();

  RETURN(hipMallocPitch3D(ptr, pitch, width, height, 0));
  CHIP_CATCH
}

hipError_t hipMallocArray(hipArray **array, const hipChannelFormatDesc *desc,
                          size_t width, size_t height, unsigned int flags) {
  CHIP_TRY
  CHIPInitialize();

  ERROR_IF((width == 0), hipErrorInvalidValue);

  *array = new hipArray;
  ERROR_IF((*array == nullptr), hipErrorOutOfMemory);

  array[0]->type = flags;
  array[0]->width = width;
  array[0]->height = height;
  array[0]->depth = 1;
  array[0]->desc = *desc;
  array[0]->isDrv = false;
  array[0]->textureType = hipTextureType2D;
  void **ptr = &array[0]->data;

  size_t size = width;
  if (height > 0) {
    size = size * height;
  }
  const size_t allocSize = size * ((desc->x + desc->y + desc->z + desc->w) / 8);

  void *retval = Backend->getActiveContext()->allocate(allocSize);
  ERROR_IF((retval == nullptr), hipErrorMemoryAllocation);

  *ptr = retval;
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipArrayCreate(hipArray **array,
                          const HIP_ARRAY_DESCRIPTOR *pAllocateArray) {
  CHIP_TRY
  CHIPInitialize();

  ERROR_IF((pAllocateArray->Width == 0), hipErrorInvalidValue);

  *array = new hipArray;
  ERROR_IF((*array == nullptr), hipErrorOutOfMemory);

  // TODO no longer there
  // array[0]->drvDesc = *pAllocateArray;
  array[0]->width = pAllocateArray->Width;
  array[0]->height = pAllocateArray->Height;
  array[0]->isDrv = true;
  array[0]->textureType = hipTextureType2D;
  void **ptr = &array[0]->data;

  size_t size = pAllocateArray->Width;
  if (pAllocateArray->Height > 0) {
    size = size * pAllocateArray->Height;
  }
  size_t allocSize = 0;
  switch (pAllocateArray->Format) {
    case HIP_AD_FORMAT_UNSIGNED_INT8:
      allocSize = size * sizeof(uint8_t);
      break;
    case HIP_AD_FORMAT_UNSIGNED_INT16:
      allocSize = size * sizeof(uint16_t);
      break;
    case HIP_AD_FORMAT_UNSIGNED_INT32:
      allocSize = size * sizeof(uint32_t);
      break;
    case HIP_AD_FORMAT_SIGNED_INT8:
      allocSize = size * sizeof(int8_t);
      break;
    case HIP_AD_FORMAT_SIGNED_INT16:
      allocSize = size * sizeof(int16_t);
      break;
    case HIP_AD_FORMAT_SIGNED_INT32:
      allocSize = size * sizeof(int32_t);
      break;
    case HIP_AD_FORMAT_HALF:
      allocSize = size * sizeof(int16_t);
      break;
    case HIP_AD_FORMAT_FLOAT:
      allocSize = size * sizeof(float);
      break;
    default:
      allocSize = size;
      break;
  }

  void *retval = Backend->getActiveContext()->allocate(allocSize);
  ERROR_IF((retval == nullptr), hipErrorMemoryAllocation);

  *ptr = retval;
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipFreeArray(hipArray *array) {
  CHIP_TRY
  CHIPInitialize();

  ERROR_IF((array == nullptr), hipErrorInvalidValue);

  assert(array->data != nullptr);

  hipError_t e = hipFree(array->data);

  delete array;

  return e;
  CHIP_CATCH
}

hipError_t hipMalloc3D(hipPitchedPtr *pitchedDevPtr, hipExtent extent) {
  CHIP_TRY
  CHIPInitialize();

  ERROR_IF((extent.width == 0 || extent.height == 0), hipErrorInvalidValue);
  ERROR_IF((pitchedDevPtr == nullptr), hipErrorInvalidValue);

  size_t pitch;

  hipError_t hip_status = hipMallocPitch3D(
      &pitchedDevPtr->ptr, &pitch, extent.width, extent.height, extent.depth);

  if (hip_status == hipSuccess) {
    pitchedDevPtr->pitch = pitch;
    pitchedDevPtr->xsize = extent.width;
    pitchedDevPtr->ysize = extent.height;
  }
  RETURN(hip_status);
  CHIP_CATCH
}

hipError_t hipMemGetInfo(size_t *free, size_t *total) {
  CHIP_TRY
  CHIPInitialize();

  ERROR_IF((total == nullptr || free == nullptr), hipErrorInvalidValue);

  auto device = Backend->getActiveDevice();
  *total = device->getGlobalMemSize();
  assert(device->getGlobalMemSize() > device->getUsedGlobalMem());
  *free = device->getGlobalMemSize() - device->getUsedGlobalMem();

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemPtrGetInfo(void *ptr, size_t *size) {
  CHIP_TRY
  CHIPInitialize();

  ERROR_IF((ptr == nullptr || size == nullptr), hipErrorInvalidValue);

  allocation_info *info =
      Backend->getActiveDevice()->allocation_tracker->getByDevPtr(ptr);
  // allocation_info *info = Backend->AllocationTracker.getByDevPtr(ptr);
  if (!info) RETURN(hipErrorInvalidDevicePointer);
  *size = info->size;
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemcpyAsync(void *dst, const void *src, size_t sizeBytes,
                          hipMemcpyKind kind, hipStream_t stream) {
  CHIP_TRY
  CHIPInitialize();
  auto q = Backend->findQueue(stream);
  if (!q) q = Backend->getActiveQueue();

  /*
  if ((kind == hipMemcpyDeviceToDevice) || (kind == hipMemcpyDeviceToHost)) {
    if (!cont->hasPointer(src))
      RETURN(hipErrorInvalidDevicePointer);
  }

  if ((kind == hipMemcpyDeviceToDevice) || (kind == hipMemcpyHostToDevice)) {
    if (!cont->hasPointer(dst))
      RETURN(hipErrorInvalidDevicePointer);
  }*/

  if (kind == hipMemcpyHostToHost) {
    memcpy(dst, src, sizeBytes);
    RETURN(hipSuccess);
  } else {
    // TODO
    // if (Backend->findQueue(stream) == nullptr) RETURN(hipErrorInvalidHandle);
    RETURN(q->memCopyAsync(dst, src, sizeBytes));
  }
  CHIP_CATCH
}

hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes,
                     hipMemcpyKind kind) {
  CHIP_TRY
  CHIPInitialize();
  // TODO
  if (dst == nullptr || src == nullptr) RETURN(hipErrorIllegalAddress);

  if (kind == hipMemcpyHostToHost) {
    memcpy(dst, src, sizeBytes);
    RETURN(hipSuccess);
  } else
    RETURN(Backend->getActiveQueue()->memCopy(dst, src, sizeBytes));
  CHIP_CATCH
}

hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src,
                              size_t sizeBytes, hipStream_t stream) {
  RETURN(hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDeviceToDevice, stream));
}

hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src,
                         size_t sizeBytes) {
  RETURN(hipMemcpy(dst, src, sizeBytes, hipMemcpyDeviceToDevice));
}

hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void *src, size_t sizeBytes,
                              hipStream_t stream) {
  RETURN(hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyHostToDevice, stream));
}

hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void *src, size_t sizeBytes) {
  RETURN(hipMemcpy(dst, src, sizeBytes, hipMemcpyHostToDevice));
}

hipError_t hipMemcpyDtoHAsync(void *dst, hipDeviceptr_t src, size_t sizeBytes,
                              hipStream_t stream) {
  RETURN(hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDeviceToHost, stream));
}

hipError_t hipMemcpyDtoH(void *dst, hipDeviceptr_t src, size_t sizeBytes) {
  RETURN(hipMemcpy(dst, src, sizeBytes, hipMemcpyDeviceToHost));
}

hipError_t hipMemsetD32Async(hipDeviceptr_t dst, int value, size_t count,
                             hipStream_t stream) {
  CHIP_TRY
  CHIPInitialize();
  if (!stream) stream = Backend->getActiveQueue();

  stream->memFillAsync(dst, 4 * count, &value, 4);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemsetD32(hipDeviceptr_t dst, int value, size_t count) {
  CHIP_TRY
  CHIPInitialize();

  Backend->getActiveQueue()->memFill(dst, 4 * count, &value, 4);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemset2DAsync(void *dst, size_t pitch, int value, size_t width,
                            size_t height, hipStream_t stream) {
  size_t sizeBytes = pitch * height;
  RETURN(hipMemsetAsync(dst, value, sizeBytes, stream));
}

hipError_t hipMemset2D(void *dst, size_t pitch, int value, size_t width,
                       size_t height) {
  size_t sizeBytes = pitch * height;
  RETURN(hipMemset(dst, value, sizeBytes));
}

hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int value,
                            hipExtent extent, hipStream_t stream) {
  size_t sizeBytes = pitchedDevPtr.pitch * extent.height * extent.depth;
  RETURN(hipMemsetAsync(pitchedDevPtr.ptr, value, sizeBytes, stream));
}

hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int value,
                       hipExtent extent) {
  size_t sizeBytes = pitchedDevPtr.pitch * extent.height * extent.depth;
  RETURN(hipMemset(pitchedDevPtr.ptr, value, sizeBytes));
}

hipError_t hipMemsetAsync(void *dst, int value, size_t sizeBytes,
                          hipStream_t stream) {
  CHIP_TRY
  CHIPInitialize();
  if (!stream) stream = Backend->getActiveQueue();

  char c_value = value;
  stream->memFillAsync(dst, sizeBytes, &c_value, 1);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemset(void *dst, int value, size_t sizeBytes) {
  CHIP_TRY
  CHIPInitialize();

  char c_value = value;
  Backend->getActiveQueue()->memFill(dst, sizeBytes, &c_value, 1);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemsetD8(hipDeviceptr_t dest, unsigned char value,
                       size_t sizeBytes) {
  RETURN(hipMemset(dest, value, sizeBytes));
}

hipError_t hipMemcpyParam2D(const hip_Memcpy2D *pCopy) {
  ERROR_IF((pCopy == nullptr), hipErrorInvalidValue);
  ERROR_IF((pCopy->dstArray == nullptr), hipErrorInvalidValue);
  return hipMemcpy2D(pCopy->dstArray->data, pCopy->WidthInBytes, pCopy->srcHost,
                     pCopy->srcPitch, pCopy->WidthInBytes, pCopy->Height,
                     hipMemcpyDefault);
}

hipError_t hipMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
                            size_t spitch, size_t width, size_t height,
                            hipMemcpyKind kind, hipStream_t stream) {
  CHIP_TRY
  CHIPInitialize();
  if (!stream) stream = Backend->getActiveQueue();

  if (spitch == 0) spitch = width;
  if (dpitch == 0) dpitch = width;

  if (spitch == 0 || dpitch == 0) RETURN(hipErrorInvalidValue);

  for (size_t i = 0; i < height; ++i) {
    if (hipMemcpyAsync(dst, src, width, kind, stream) != hipSuccess)
      RETURN(hipErrorLaunchFailure);
    src = (char *)src + spitch;
    dst = (char *)dst + dpitch;
  }
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
                       size_t width, size_t height, hipMemcpyKind kind) {
  CHIP_TRY
  CHIPInitialize();

  hipError_t e = hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind,
                                  Backend->getActiveQueue());
  if (e != hipSuccess) return e;

  Backend->getActiveQueue()->finish();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemcpy2DToArray(hipArray *dst, size_t wOffset, size_t hOffset,
                              const void *src, size_t spitch, size_t width,
                              size_t height, hipMemcpyKind kind) {
  CHIP_TRY
  CHIPInitialize();

  size_t byteSize;
  if (dst) {
    switch (dst[0].desc.f) {
      case hipChannelFormatKindSigned:
        byteSize = sizeof(int);
        break;
      case hipChannelFormatKindUnsigned:
        byteSize = sizeof(unsigned int);
        break;
      case hipChannelFormatKindFloat:
        byteSize = sizeof(float);
        break;
      case hipChannelFormatKindNone:
        byteSize = sizeof(size_t);
        break;
    }
  } else {
    RETURN(hipErrorUnknown);
  }

  if ((wOffset + width > (dst->width * byteSize)) || width > spitch) {
    RETURN(hipErrorInvalidValue);
  }

  size_t src_w = spitch;
  size_t dst_w = (dst->width) * byteSize;

  for (size_t i = 0; i < height; ++i) {
    void *dst_p = ((unsigned char *)dst->data + i * dst_w);
    void *src_p = ((unsigned char *)src + i * src_w);
    if (hipMemcpyAsync(dst_p, src_p, width, kind, Backend->getActiveQueue()) !=
        hipSuccess)
      RETURN(hipErrorLaunchFailure);
  }

  Backend->getActiveQueue()->finish();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemcpyToArray(hipArray *dst, size_t wOffset, size_t hOffset,
                            const void *src, size_t count, hipMemcpyKind kind) {
  void *dst_p = (unsigned char *)dst->data + wOffset;
  RETURN(hipMemcpy(dst_p, src, count, kind));
}

hipError_t hipMemcpyFromArray(void *dst, hipArray_const_t srcArray,
                              size_t wOffset, size_t hOffset, size_t count,
                              hipMemcpyKind kind) {
  void *src_p = (unsigned char *)srcArray->data + wOffset;
  RETURN(hipMemcpy(dst, src_p, count, kind));
}

hipError_t hipMemcpyAtoH(void *dst, hipArray *srcArray, size_t srcOffset,
                         size_t count) {
  return hipMemcpy((char *)dst, (char *)srcArray->data + srcOffset, count,
                   hipMemcpyDeviceToHost);
}

hipError_t hipMemcpyHtoA(hipArray *dstArray, size_t dstOffset,
                         const void *srcHost, size_t count) {
  return hipMemcpy((char *)dstArray->data + dstOffset, srcHost, count,
                   hipMemcpyHostToDevice);
}

hipError_t hipMemcpy3D(const struct hipMemcpy3DParms *p) {
  CHIP_TRY
  CHIPInitialize();
  // TODO: make this conversion
  // const HIP_MEMCPY3D desc = hip::getDrvMemcpy3DDesc(*p);
  // ERROR_IF((p == nullptr), hipErrorInvalidValue);

  // size_t byteSize;
  // size_t depth;
  // size_t height;
  // size_t widthInBytes;
  // size_t srcPitch;
  // size_t dstPitch;
  // void *srcPtr;
  // void *dstPtr;
  // size_t ySize;

  // if (p->dstArray != nullptr) {
  //   if (p->dstArray->isDrv == false) {
  //     switch (p->dstArray->desc.f) {
  //       case hipChannelFormatKindSigned:
  //         byteSize = sizeof(int);
  //         break;
  //       case hipChannelFormatKindUnsigned:
  //         byteSize = sizeof(unsigned int);
  //         break;
  //       case hipChannelFormatKindFloat:
  //         byteSize = sizeof(float);
  //         break;
  //       case hipChannelFormatKindNone:
  //         byteSize = sizeof(size_t);
  //         break;
  //     }
  //     depth = p->extent.depth;
  //     height = p->extent.height;
  //     widthInBytes = p->extent.width * byteSize;
  //     srcPitch = p->srcPtr.pitch;
  //     srcPtr = p->srcPtr.ptr;
  //     ySize = p->srcPtr.ysize;
  //     dstPitch = p->dstArray->width * byteSize;
  //     dstPtr = p->dstArray->data;
  //   } else {
  //     depth = p->Depth;
  //     height = p->Height;
  //     widthInBytes = p->WidthInBytes;
  //     dstPitch = p->dstArray->width * 4;
  //     srcPitch = p->srcPitch;
  //     srcPtr = (void *)p->srcHost;
  //     ySize = p->srcHeight;
  //     dstPtr = p->dstArray->data;
  //   }
  // } else {
  //   // Non array destination
  //   depth = p->extent.depth;
  //   height = p->extent.height;
  //   widthInBytes = p->extent.width;
  //   srcPitch = p->srcPtr.pitch;
  //   srcPtr = p->srcPtr.ptr;
  //   dstPtr = p->dstPtr.ptr;
  //   ySize = p->srcPtr.ysize;
  //   dstPitch = p->dstPtr.pitch;
  // }

  // if ((widthInBytes == dstPitch) && (widthInBytes == srcPitch)) {
  //   return hipMemcpy((void *)dstPtr, (void *)srcPtr,
  //                    widthInBytes * height * depth, p->kind);
  // } else {
  //   for (size_t i = 0; i < depth; i++) {
  //     for (size_t j = 0; j < height; j++) {
  //       unsigned char *src =
  //           (unsigned char *)srcPtr + i * ySize * srcPitch + j * srcPitch;
  //       unsigned char *dst =
  //           (unsigned char *)dstPtr + i * height * dstPitch + j * dstPitch;
  //       if (hipMemcpyAsync(dst, src, widthInBytes, p->kind,
  //                          Backend->getActiveQueue()) != hipSuccess)
  //         RETURN(hipErrorLaunchFailure);
  //     }
  //   }

  //   Backend->getActiveQueue()->finish();
  //  RETURN(hipSuccess);
  // }
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipFuncGetAttributes(hipFuncAttributes *attr, const void *func) {
  CHIP_TRY
  CHIPInitialize();
  UNIMPLEMENTED(hipErrorUnknown);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipModuleGetGlobal(hipDeviceptr_t *dptr, size_t *bytes,
                              hipModule_t hmod, const char *name) {
  CHIP_TRY
  CHIPInitialize();

  ERROR_IF((!dptr || !bytes || !name || !hmod), hipErrorInvalidValue);
  CHIPDeviceVar *var = hmod->getGlobalVar(name);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGetSymbolSize(size_t *size, const void *symbol) {
  CHIP_TRY
  CHIPInitialize();

  CHIPDeviceVar *var = Backend->getActiveDevice()->getGlobalVar(symbol);
  ERROR_IF(!var, hipErrorInvalidSymbol);

  *size = var->getSize();

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemcpyToSymbol(const void *symbol, const void *src,
                             size_t sizeBytes, size_t offset,
                             hipMemcpyKind kind) {
  CHIP_TRY
  CHIPInitialize();

  hipError_t e = hipMemcpyToSymbolAsync(symbol, src, sizeBytes, offset, kind,
                                        Backend->getActiveQueue());
  if (e != hipSuccess) RETURN(e);

  Backend->getActiveQueue()->finish();

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemcpyToSymbolAsync(const void *symbol, const void *src,
                                  size_t sizeBytes, size_t offset,
                                  hipMemcpyKind kind, hipStream_t stream) {
  CHIP_TRY
  CHIPInitialize();
  if (!stream) stream = Backend->getActiveQueue();

  void *symPtr = NULL;
  size_t symSize = 0;
  CHIPDeviceVar *var = Backend->getActiveDevice()->getGlobalVar(symbol);
  ERROR_IF(!var, hipErrorInvalidSymbol);

  RETURN(hipMemcpyAsync((void *)((intptr_t)symPtr + offset), src, sizeBytes,
                        kind, stream));
  CHIP_CATCH
}

hipError_t hipMemcpyFromSymbol(void *dst, const void *symbol, size_t sizeBytes,
                               size_t offset, hipMemcpyKind kind) {
  CHIP_TRY
  CHIPInitialize();

  hipError_t e = hipMemcpyFromSymbolAsync(dst, symbol, sizeBytes, offset, kind,
                                          Backend->getActiveQueue());
  if (e != hipSuccess) RETURN(e);

  Backend->getActiveQueue()->finish();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemcpyFromSymbolAsync(void *dst, const void *symbol,
                                    size_t sizeBytes, size_t offset,
                                    hipMemcpyKind kind, hipStream_t stream) {
  CHIP_TRY
  CHIPInitialize();
  if (!stream) stream = Backend->getActiveQueue();

  void *symPtr;
  size_t symSize;

  CHIPDeviceVar *var = stream->getDevice()->getGlobalVar(symbol);
  ERROR_IF(!var, hipErrorInvalidSymbol);

  RETURN(hipMemcpyAsync(dst, (void *)((intptr_t)symPtr + offset), sizeBytes,
                        kind, stream));
  CHIP_CATCH
}

hipError_t hipModuleLoadData(hipModule_t *module, const void *image) {
  UNIMPLEMENTED(hipErrorUnknown);
  RETURN(hipSuccess);
}

hipError_t hipModuleLoadDataEx(hipModule_t *module, const void *image,
                               unsigned int numOptions, hipJitOption *options,
                               void **optionValues) {
  CHIP_TRY
  CHIPInitialize();
  RETURN(hipModuleLoadData(module, image));
  CHIP_CATCH
}

hipError_t hipLaunchKernel(const void *hostFunction, dim3 gridDim,
                           dim3 blockDim, void **args, size_t sharedMem,
                           hipStream_t stream) {
  CHIP_TRY
  CHIPInitialize();
  if (!stream) stream = Backend->getActiveQueue();

  if (!stream->launchHostFunc(hostFunction, gridDim, blockDim, args,
                              sharedMem)) {
    RETURN(hipErrorLaunchFailure);
  }
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipCreateTextureObject(
    hipTextureObject_t *pTexObject, const hipResourceDesc *pResDesc,
    const hipTextureDesc *pTexDesc,
    const struct hipResourceViewDesc *pResViewDesc) {
  CHIP_TRY
  CHIPInitialize();
  CHIPTexture *chip_texture = Backend->getActiveDevice()->createTexture(
      pResDesc, pTexDesc, pResViewDesc);
  hipTextureObject_t retObj = chip_texture->get();
  if (retObj != nullptr) {
    *pTexObject = retObj;
    RETURN(hipSuccess);
  } else
    RETURN(hipErrorLaunchFailure);
  CHIP_CATCH
}

hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject) {
  CHIP_TRY
  CHIPInitialize();
  // TODO CRITCAL look into the define for hipTextureObject_t
  if (textureObject == nullptr)
    CHIPERR_LOG_AND_THROW("hipTextureObject_t is null", hipErrorTbd);
  CHIPTexture *chip_texture = (CHIPTexture *)&textureObject;
  Backend->getActiveDevice()->destroyTexture(chip_texture);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipModuleLoad(hipModule_t *module, const char *fname) {
  CHIP_TRY
  CHIPInitialize();

  std::ifstream file(fname, std::ios::in | std::ios::binary | std::ios::ate);
  ERROR_IF((file.fail()), hipErrorFileNotFound);

  size_t size = file.tellg();
  char *memblock = new char[size];
  file.seekg(0, std::ios::beg);
  file.read(memblock, size);
  file.close();
  std::string content(memblock, size);
  delete[] memblock;

  // CHIPModule *chip_module = new CHIPModule(std::move(content));
  for (auto &dev : Backend->getDevices()) dev->addModule(&content);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipModuleUnload(hipModule_t module) { UNIMPLEMENTED(hipErrorTbd); }

hipError_t hipModuleGetFunction(hipFunction_t *function, hipModule_t module,
                                const char *kname) {
  CHIP_TRY
  CHIPInitialize();

  ERROR_IF(!module, hipErrorInvalidValue);
  CHIPKernel *kernel = module->getKernel(kname);

  ERROR_IF((kernel == nullptr), hipErrorInvalidDeviceFunction);

  *function = kernel;
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipModuleLaunchKernel(hipFunction_t k, unsigned int gridDimX,
                                 unsigned int gridDimY, unsigned int gridDimZ,
                                 unsigned int blockDimX, unsigned int blockDimY,
                                 unsigned int blockDimZ,
                                 unsigned int sharedMemBytes,
                                 hipStream_t stream, void **kernelParams,
                                 void **extra) {
  CHIP_TRY
  CHIPInitialize();
  if (!stream) stream = Backend->getActiveQueue();

  if (sharedMemBytes > 0) {
    logCritical("Dynamic shared memory not yet implemented");
    RETURN(hipErrorLaunchFailure);
  }

  if (kernelParams == nullptr && extra == nullptr) {
    logError("either kernelParams or extra is required!\n");
    RETURN(hipErrorLaunchFailure);
  }

  dim3 grid(gridDimX, gridDimY, gridDimZ);
  dim3 block(blockDimX, blockDimY, blockDimZ);

  if (kernelParams)
    RETURN(stream->launchWithKernelParams(grid, block, sharedMemBytes,
                                          kernelParams, k));
  else
    RETURN(
        stream->launchWithExtraParams(grid, block, sharedMemBytes, extra, k));
  CHIP_CATCH
}

//*****************************************************************************
//*****************************************************************************
//*****************************************************************************

hipError_t hipLaunchByPtr(const void *hostFunction) {
  CHIP_TRY
  CHIPInitialize();
  logTrace("hipLaunchByPtr");
  CHIPExecItem *exec_item = Backend->chip_execstack.top();
  Backend->chip_execstack.pop();

  RETURN(exec_item->launchByHostPtr(hostFunction));
  CHIP_CATCH
}

hipError_t hipConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                            hipStream_t stream) {
  CHIP_TRY
  CHIPInitialize();
  if (!stream) stream = Backend->getActiveQueue();
  logTrace("hipConfigureCall()");
  RETURN(Backend->configureCall(gridDim, blockDim, sharedMem, stream));
  RETURN(hipSuccess);
  CHIP_CATCH
}
extern "C" void **__hipRegisterFatBinary(const void *data) {
  CHIP_TRY
  CHIPInitialize();

  logTrace("__hipRegisterFatBinary");

  const __CudaFatBinaryWrapper *fbwrapper =
      reinterpret_cast<const __CudaFatBinaryWrapper *>(data);
  if (fbwrapper->magic != __hipFatMAGIC2 || fbwrapper->version != 1) {
    CHIPERR_LOG_AND_THROW("The given object is not hipFatBinary",
                          hipErrorInitializationError);
  }

  const __ClangOffloadBundleHeader *header = fbwrapper->binary;
  std::string magic(reinterpret_cast<const char *>(header),
                    sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1);
  if (magic.compare(CLANG_OFFLOAD_BUNDLER_MAGIC)) {
    CHIPERR_LOG_AND_THROW(
        "The bundled binaries are not Clang bundled "
        "(CLANG_OFFLOAD_BUNDLER_MAGIC is missing)",
        hipErrorInitializationError);
  }

  std::string *module = new std::string;
  if (!module) {
    CHIPERR_LOG_AND_THROW("Failed to allocate memory",
                          hipErrorInitializationError);
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
    CHIPERR_LOG_AND_THROW("Didn't find any suitable compiled binary!",
                          hipErrorInitializationError);
  }

  const char *string_data = reinterpret_cast<const char *>(
      reinterpret_cast<uintptr_t>(header) + (uintptr_t)desc->offset);
  size_t string_size = desc->size;
  module->assign(string_data, string_size);

  logDebug("Register module: {} \n", (void *)module);

  Backend->registerModuleStr(module);

  ++binaries_loaded;

  return (void **)module;
  CHIP_CATCH_NO_RETURN
  return nullptr;
}

extern "C" void __hipUnregisterFatBinary(void *data) {
  CHIP_TRY
  CHIPInitialize();
  std::string *module = reinterpret_cast<std::string *>(data);

  logDebug("Unregister module: {} \n", (void *)module);
  Backend->unregisterModuleStr(module);

  --binaries_loaded;
  logDebug("__hipUnRegisterFatBinary {}\n", binaries_loaded);

  if (binaries_loaded == 0) {
    CHIPUninitialize();
  }

  delete module;
  CHIP_CATCH_NO_RETURN
}

extern "C" void __hipRegisterFunction(void **data, const void *hostFunction,
                                      char *deviceFunction,
                                      const char *deviceName,
                                      unsigned int threadLimit, void *tid,
                                      void *bid, dim3 *blockDim, dim3 *gridDim,
                                      int *wSize) {
  CHIP_TRY
  CHIPInitialize();
  std::string *module_str = reinterpret_cast<std::string *>(data);

  std::string devFunc = deviceFunction;
  logDebug("RegisterFunction on module {}\n", (void *)module_str);

  logDebug("RegisterFunction on {} devices", Backend->getNumDevices());
  Backend->registerFunctionAsKernel(module_str, hostFunction, deviceName);
  CHIP_CATCH_NO_RETURN
}

hipError_t hipSetupArgument(const void *arg, size_t size, size_t offset) {
  logTrace("hipSetupArgument");

  CHIP_TRY
  CHIPInitialize();
  RETURN(Backend->setArg(arg, size, offset));
  RETURN(hipSuccess);
  CHIP_CATCH
}

extern "C" hipError_t hipInitFromOutside(void *driverPtr, void *devicePtr,
                                         void *ctxPtr, void *queuePtr) {
  logDebug("hipInitFromOutside");
  auto modules = std::move(Backend->getDevices()[0]->getModules());
  delete Backend;
  logDebug("deleting Backend object.");
  Backend = new CHIPBackendLevel0();

  ze_context_handle_t ctx = (ze_context_handle_t)ctxPtr;
  CHIPContextLevel0 *chip_ctx = new CHIPContextLevel0(ctx);
  Backend->addContext(chip_ctx);

  ze_device_handle_t dev = (ze_device_handle_t)devicePtr;
  CHIPDeviceLevel0 *chip_dev = new CHIPDeviceLevel0(&dev, chip_ctx);
  chip_dev->chip_modules = modules;
  Backend->chip_contexts[0]->getDevices().push_back(chip_dev);
  Backend->addDevice(chip_dev);

  // ze_command_queue_handle_t q = (ze_command_queue_handle_t)queuePtr;
  // CHIPQueueLevel0* chip_queue = CHIPQueueLevel0(q)
  CHIPQueueLevel0 *chip_queue = new CHIPQueueLevel0(chip_dev);
  Backend->addQueue(chip_queue);
  Backend->setActiveDevice(chip_dev);

  RETURN(hipSuccess);
}

extern "C" void __hipRegisterVar(
    void **data,  // std::vector<hipModule_t> *modules,
    char *hostVar, char *deviceVar, const char *deviceName, int ext, int size,
    int constant, int global) {
  UNIMPLEMENTED();
}

hipError_t hipGetSymbolAddress(void **devPtr, const void *symbol) {
  UNIMPLEMENTED(hipSuccess);
}

hipError_t hipIpcOpenEventHandle(hipEvent_t *event,
                                 hipIpcEventHandle_t handle) {
  UNIMPLEMENTED(hipSuccess);
}
hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t *handle, hipEvent_t event) {
  UNIMPLEMENTED(hipSuccess);
}

#endif