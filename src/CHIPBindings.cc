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
#include "CHIPException.hh"
#include "backend/backends.hh"
#include "hip/hip_fatbin.h"
#include "hip/hip_runtime_api.h"
#include "hip_conversions.hh"
#include "macros.hh"

#define SPIR_BUNDLE_ID "hip-spir64-unknown-unknown"

static unsigned NumBinariesLoaded = 0;

#define SVM_ALIGNMENT 128 // TODO Pass as CMAKE Define?

hipError_t hipPointerGetAttributes(hipPointerAttribute_t *attributes,
                                   const void *ptr) {
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t hipIpcOpenMemHandle(void **DevPtr, hipIpcMemHandle_t Handle,
                               unsigned int Flags) {
  UNIMPLEMENTED(hipErrorNotSupported);
}
hipError_t hipIpcCloseMemHandle(void *DevPtr) {
  UNIMPLEMENTED(hipErrorNotSupported);
}
hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t *Handle, void *DevPtr) {
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t hipMemcpyWithStream(void *Dst, const void *Src, size_t SizeBytes,
                               hipMemcpyKind Kind, hipStream_t Stream) {
  auto Status = hipMemcpyAsync(Dst, Src, SizeBytes, Kind, Stream);
  Stream->finish();
  RETURN(Status);
};

hipError_t hipMemcpyPeer(void *Dst, int DstDeviceId, const void *Src,
                         int SrcDeviceId, size_t SizeBytes) {
  UNIMPLEMENTED(hipErrorNotSupported);
};
hipError_t hipMemRangeGetAttribute(void *Data, size_t DataSize,
                                   hipMemRangeAttribute Attribute,
                                   const void *DevPtr, size_t Count) {
  UNIMPLEMENTED(hipErrorNotSupported);
};

hipError_t hipMalloc3DArray(hipArray **Array,
                            const struct hipChannelFormatDesc *Desc,
                            struct hipExtent Extent, unsigned int Flags) {
  UNIMPLEMENTED(hipErrorNotSupported);
};

hipError_t hipMemcpyPeerAsync(void *Dst, int DstDeviceId, const void *Src,
                              int SrcDevice, size_t SizeBytes,
                              hipStream_t Stream) {
  UNIMPLEMENTED(hipErrorNotSupported);
};

hipError_t hipMemcpyParam2DAsync(const hip_Memcpy2D *PCopy,
                                 hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();

  if (PCopy->dstPitch == 0)
    return hipSuccess;
  if (PCopy->srcPitch == 0)
    return hipSuccess;
  if (PCopy->Height * PCopy->WidthInBytes == 0)
    return hipSuccess;
  if (PCopy->srcDevice == nullptr && PCopy->dstDevice == nullptr)
    CHIPERR_LOG_AND_THROW("Source and Destination Device pointer is null",
                          hipErrorTbd);

  if (PCopy->dstDevice != nullptr && PCopy->srcDevice == nullptr)
    CHIPERR_LOG_AND_THROW("Source Device pointer is null", hipErrorTbd);
  if (PCopy->srcDevice != nullptr && PCopy->dstDevice == nullptr)
    CHIPERR_LOG_AND_THROW("Source Device pointer is null", hipErrorTbd);

  if ((PCopy->WidthInBytes > PCopy->dstPitch) ||
      (PCopy->WidthInBytes > PCopy->srcPitch))
    CHIPERR_LOG_AND_THROW("Width > src/dest pitches", hipErrorTbd);

  return hipMemcpy2D(PCopy->dstArray->data, PCopy->WidthInBytes, PCopy->srcHost,
                     PCopy->srcPitch, PCopy->WidthInBytes, PCopy->Height,
                     hipMemcpyDefault);
  CHIP_CATCH
}

//*****************************************************************************
//*****************************************************************************
//*****************************************************************************

hipError_t __hipPushCallConfiguration(dim3 GridDim, dim3 BlockDim,
                                      size_t SharedMem, hipStream_t Stream) {
  logDebug("__hipPushCallConfiguration()");
  CHIP_TRY
  CHIPInitialize();
  Stream = Backend->findQueue(Stream);

  RETURN(Backend->configureCall(GridDim, BlockDim, SharedMem, Stream));
  CHIP_CATCH
  RETURN(hipSuccess);
}

hipError_t __hipPopCallConfiguration(dim3 *GridDim, dim3 *BlockDim,
                                     size_t *SharedMem, hipStream_t *Stream) {
  logDebug("__hipPopCallConfiguration()");
  CHIP_TRY
  CHIPInitialize();

  auto *ExecItem = Backend->ChipExecStack.top();
  *GridDim = ExecItem->getGrid();
  *BlockDim = ExecItem->getBlock();
  *SharedMem = ExecItem->getSharedMem();
  *Stream = ExecItem->getQueue();
  Backend->ChipExecStack.pop();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGetDevice(int *DeviceId) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(DeviceId);

  CHIPDevice *ChipDev = Backend->getActiveDevice();
  *DeviceId = ChipDev->getDeviceId();

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGetDeviceCount(int *Count) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Count);

  *Count = Backend->getNumDevices();

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipSetDevice(int DeviceId) {
  CHIP_TRY
  CHIPInitialize();

  ERROR_CHECK_DEVNUM(DeviceId);

  CHIPDevice *SelectedDevice = Backend->getDevices()[DeviceId];
  Backend->setActiveDevice(SelectedDevice);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceSynchronize(void) {
  CHIP_TRY
  CHIPInitialize();

  for (auto Q : Backend->getQueues())
    Q->finish();

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceReset(void) {
  CHIP_TRY
  CHIPInitialize();

  CHIPDevice *ChipDev = Backend->getActiveDevice();

  ChipDev->reset();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceGet(hipDevice_t *Device, int Ordinal) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Device);
  ERROR_CHECK_DEVNUM(Ordinal);

  /// Since the tests are written such that hipDevice_t is an int, this function
  /// is strange
  *Device = Ordinal;

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceComputeCapability(int *Major, int *Minor,
                                      hipDevice_t Device) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Major, Minor);
  ERROR_CHECK_DEVNUM(Device);

  hipDeviceProp_t Props;
  Backend->getDevices()[Device]->copyDeviceProperties(&Props);

  if (Major)
    *Major = Props.major;
  if (Minor)
    *Minor = Props.minor;

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceGetAttribute(int *RetPtr, hipDeviceAttribute_t Attr,
                                 int DeviceId) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(RetPtr);
  ERROR_CHECK_DEVNUM(DeviceId);

  *RetPtr = Backend->getDevices()[DeviceId]->getAttr(Attr);
  if (*RetPtr == -1)
    RETURN(hipErrorInvalidValue);
  else
    RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipGetDeviceProperties(hipDeviceProp_t *Prop, int DeviceId) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Prop);
  ERROR_CHECK_DEVNUM(DeviceId);

  Backend->getDevices()[DeviceId]->copyDeviceProperties(Prop);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceGetLimit(size_t *PValue, enum hipLimit_t Limit) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(PValue);

  auto Device = Backend->getActiveDevice();
  switch (Limit) {
  case hipLimitMallocHeapSize:
    *PValue = Device->getMaxMallocSize();
    break;
  case hipLimitPrintfFifoSize:
    UNIMPLEMENTED(hipErrorNotSupported);
    break;
  default:
    CHIPERR_LOG_AND_THROW("Invalid Limit value", hipErrorInvalidHandle);
  }

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceGetName(char *Name, int Len, hipDevice_t Device) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Name);
  ERROR_CHECK_DEVNUM(Device);

  std::string DeviceName = (Backend->getDevices()[Device])->getName();

  size_t NameLen = DeviceName.size();
  NameLen = (NameLen < (size_t)Len ? NameLen : Len - 1);
  memcpy(Name, DeviceName.data(), NameLen);
  Name[NameLen] = 0;
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipDeviceTotalMem(size_t *Bytes, hipDevice_t Device) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Bytes);
  ERROR_CHECK_DEVNUM(Device);

  if (Bytes)
    *Bytes = (Backend->getDevices()[Device])->getGlobalMemSize();
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipDeviceSetCacheConfig(hipFuncCache_t CacheCfg) {
  CHIP_TRY
  CHIPInitialize();

  Backend->getActiveDevice()->setCacheConfig(CacheCfg);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceGetCacheConfig(hipFuncCache_t *CacheCfg) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(CacheCfg);

  if (CacheCfg)
    *CacheCfg = Backend->getActiveDevice()->getCacheConfig();
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig *Cfg) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Cfg);

  if (Cfg)
    *Cfg = Backend->getActiveDevice()->getSharedMemConfig();
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig Cfg) {
  CHIP_TRY
  CHIPInitialize();

  Backend->getActiveDevice()->setSharedMemConfig(Cfg);
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipFuncSetCacheConfig(const void *Func, hipFuncCache_t Cfg) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Func);

  UNIMPLEMENTED(hipErrorNotSupported);
  // RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipDeviceGetPCIBusId(char *PciBusId, int Len, int DeviceId) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(PciBusId);
  ERROR_CHECK_DEVNUM(DeviceId);
  if (Len < 1)
    RETURN(hipErrorInvalidResourceHandle);

  CHIPDevice *Dev = Backend->getDevices()[DeviceId];

  hipDeviceProp_t Prop;
  Dev->copyDeviceProperties(&Prop);
  snprintf(PciBusId, Len, "%04x:%02x:%02x", Prop.pciDomainID, Prop.pciBusID,
           Prop.pciDeviceID);
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipDeviceGetByPCIBusId(int *DeviceId, const char *PciBusId) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(DeviceId, PciBusId);

  int PciDomainID, PciBusID, PciDeviceID;
  int Err =
      sscanf(PciBusId, "%4x:%4x:%4x", &PciDomainID, &PciBusID, &PciDeviceID);
  if (Err == EOF || Err < 3)
    RETURN(hipErrorInvalidValue);
  for (size_t DevIdx = 0; DevIdx < Backend->getNumDevices(); DevIdx++) {
    CHIPDevice *Dev = Backend->getDevices()[DevIdx];
    if (Dev->hasPCIBusId(PciDomainID, PciBusID, PciDeviceID)) {
      *DeviceId = DevIdx;
      RETURN(hipSuccess);
    }
  }

  RETURN(hipErrorInvalidDevice);
  CHIP_CATCH
}

hipError_t hipSetDeviceFlags(unsigned Flags) {
  CHIP_TRY
  CHIPInitialize();
  UNIMPLEMENTED(hipErrorNotSupported);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceCanAccessPeer(int *CanAccessPeer, int DeviceId,
                                  int PeerDeviceId) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(CanAccessPeer);
  ERROR_CHECK_DEVNUM(DeviceId);
  ERROR_CHECK_DEVNUM(PeerDeviceId);

  if (DeviceId == PeerDeviceId) {
    *CanAccessPeer = 0;
    RETURN(hipSuccess);
  }

  CHIPDevice *Dev = Backend->getDevices()[DeviceId];
  CHIPDevice *Peer = Backend->getDevices()[PeerDeviceId];

  *CanAccessPeer = Dev->getPeerAccess(Peer);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDeviceEnablePeerAccess(int PeerDeviceId, unsigned int Flags) {
  CHIP_TRY
  CHIPInitialize();

  CHIPDevice *Dev = Backend->getActiveDevice();
  CHIPDevice *Peer = Backend->getDevices()[PeerDeviceId];

  RETURN(Dev->setPeerAccess(Peer, Flags, true));
  CHIP_CATCH
}

hipError_t hipDeviceDisablePeerAccess(int PeerDeviceId) {
  CHIP_TRY
  CHIPInitialize();

  CHIPDevice *Dev = Backend->getActiveDevice();
  CHIPDevice *Peer = Backend->getDevices()[PeerDeviceId];

  RETURN(Dev->setPeerAccess(Peer, 0, false));
  CHIP_CATCH
}

hipError_t hipChooseDevice(int *DeviceId, const hipDeviceProp_t *Prop) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(DeviceId, Prop);

  CHIPDevice *Dev = Backend->findDeviceMatchingProps(Prop);
  if (!Dev)
    RETURN(hipErrorInvalidValue);

  *DeviceId = Dev->getDeviceId();

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDriverGetVersion(int *DriverVersion) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(DriverVersion);

  if (DriverVersion) {
    *DriverVersion = 4;
    logWarn("Driver version is hardcoded to 4");
    RETURN(hipSuccess);
  } else
    RETURN(hipErrorInvalidValue);

  CHIP_CATCH
}

hipError_t hipRuntimeGetVersion(int *RuntimeVersion) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(RuntimeVersion);

  if (RuntimeVersion) {
    *RuntimeVersion = 1;
    RETURN(hipSuccess);
  } else
    RETURN(hipErrorInvalidValue);

  CHIP_CATCH
}

hipError_t hipGetLastError(void) {
  CHIPInitialize();

  hipError_t Temp = Backend->TlsLastError;
  Backend->TlsLastError = hipSuccess;
  return Temp;
}

hipError_t hipPeekAtLastError(void) {
  CHIPInitialize();

  return Backend->TlsLastError;
}

const char *hipGetErrorName(hipError_t HipError) {
  switch (HipError) {
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

const char *hipGetErrorString(hipError_t HipError) {
  return hipGetErrorName(HipError);
}

hipError_t hipStreamCreate(hipStream_t *Stream) {
  RETURN(hipStreamCreateWithFlags(Stream, 0));
}

hipError_t hipStreamCreateWithFlags(hipStream_t *Stream, unsigned int Flags) {
  RETURN(hipStreamCreateWithPriority(Stream, Flags, 0));
}

hipError_t hipStreamCreateWithPriority(hipStream_t *Stream, unsigned int Flags,
                                       int Priority) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Stream);

  CHIPDevice *Dev = Backend->getActiveDevice();
  CHIPQueue *ChipQueue = Dev->addQueue(Flags, Priority);
  Backend->addQueue(ChipQueue);
  *Stream = ChipQueue;
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipDeviceGetStreamPriorityRange(int *LeastPriority,
                                           int *GreatestPriority) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(LeastPriority, GreatestPriority);

  CHIPQueue *ChipQueue = Backend->getActiveQueue();

  if (LeastPriority)
    *LeastPriority = ChipQueue->getPriorityRange(0);
  if (GreatestPriority)
    *GreatestPriority = ChipQueue->getPriorityRange(1);
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipStreamDestroy(hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();

  Stream = Backend->findQueue(Stream);
  CHIPDevice *Dev = Backend->getActiveDevice();

  if (Dev->removeQueue(Stream))
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);

  CHIP_CATCH
}

hipError_t hipStreamQuery(hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();

  Stream = Backend->findQueue(Stream);
  if (Stream->query()) {
    RETURN(hipSuccess);
  } else
    RETURN(hipErrorNotReady);

  CHIP_CATCH
}

hipError_t hipStreamSynchronize(hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();

  Stream = Backend->findQueue(Stream);
  Stream->finish();
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipStreamWaitEvent(hipStream_t Stream, hipEvent_t Event,
                              unsigned int Flags) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Event);

  Stream = Backend->findQueue(Stream);
  ERROR_IF((Stream == nullptr), hipErrorInvalidResourceHandle);
  ERROR_IF((Event == nullptr), hipErrorInvalidResourceHandle);

  std::vector<CHIPEvent *> EventsToWaitOn = {Event};
  Stream->enqueueBarrier(&EventsToWaitOn);
  // event->barrier(Stream);
  // if (Stream->enqueueBarrier(event))
  // RETURN(hipSuccess);
  // else
  // RETURN(hipErrorInvalidValue);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipStreamGetFlags(hipStream_t Stream, unsigned int *Flags) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Flags);

  Stream = Backend->findQueue(Stream);
  *Flags = Stream->getFlags();
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipStreamGetPriority(hipStream_t Stream, int *Priority) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Priority);

  Stream = Backend->findQueue(Stream);
  *Priority = Stream->getPriority();
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipStreamAddCallback(hipStream_t Stream,
                                hipStreamCallback_t Callback, void *UserData,
                                unsigned int Flags) {
  CHIP_TRY
  CHIPInitialize();
  if (Flags)
    CHIPERR_LOG_AND_THROW(
        "hipStreamAddCallback: Flags are non-zero (reserved argument. Must be "
        "0)",
        hipErrorTbd);
  // TODO: Can't use NULLCHECK for this one
  if (Callback == nullptr)
    CHIPERR_LOG_AND_THROW("passed in nullptr", hipErrorInvalidValue);

  Stream = Backend->findQueue(Stream);
  if (Stream->addCallback(Callback, UserData))
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);
  CHIP_CATCH
}

hipError_t hipMemGetAddressRange(hipDeviceptr_t *Base, size_t *Size,
                                 hipDeviceptr_t Ptr) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Base, Size, Ptr);

  CHIPContext *ChipContext = Backend->getActiveContext();
  if (ChipContext->findPointerInfo(Base, Size, Ptr))
    RETURN(hipSuccess);
  else
    RETURN(hipErrorInvalidValue);

  CHIP_CATCH
}

hipError_t hipDevicePrimaryCtxGetState(hipDevice_t Device, unsigned int *Flags,
                                       int *Active) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Flags, Active);
  ERROR_CHECK_DEVNUM(Device);

  CHIPContext *CurrCtx = Backend->getActiveContext();

  // Currently device only has 1 context
  CHIPContext *PrimaryCtx = (Backend->getDevices()[Device])->getContext();

  *Active = (PrimaryCtx == CurrCtx) ? 1 : 0;
  *Flags = PrimaryCtx->getFlags();
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipDevicePrimaryCtxRelease(hipDevice_t Device) {
  CHIP_TRY
  CHIPInitialize();
  ERROR_CHECK_DEVNUM(Device);
  UNIMPLEMENTED(hipErrorNotSupported);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDevicePrimaryCtxRetain(hipCtx_t *Context, hipDevice_t Device) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Context);
  ERROR_CHECK_DEVNUM(Device);

  UNIMPLEMENTED(hipErrorNotSupported);
  *Context = (Backend->getDevices()[Device])->getContext()->retain();
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipDevicePrimaryCtxReset(hipDevice_t Device) {
  CHIP_TRY
  CHIPInitialize();
  ERROR_CHECK_DEVNUM(Device);

  (Backend->getDevices()[Device])->getContext()->reset();

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t Device, unsigned int Flags) {
  CHIP_TRY
  CHIPInitialize();
  ERROR_CHECK_DEVNUM(Device);

  UNIMPLEMENTED(hipErrorNotSupported);
  (Backend->getDevices()[Device])->getContext()->setFlags(Flags);
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipEventCreate(hipEvent_t *Event) {
  RETURN(hipEventCreateWithFlags(Event, 0));
}

hipError_t hipEventCreateWithFlags(hipEvent_t *Event, unsigned Flags) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Event);

  if (Flags > (hipEventDefault | hipEventBlockingSync | hipEventDisableTiming |
               hipEventInterprocess))
    CHIPERR_LOG_AND_THROW("Invalid hipEvent flag combination",
                          hipErrorInvalidValue);

  *Event = Backend->createCHIPEvent(Backend->getActiveContext(), Flags, true);
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipEventRecord(hipEvent_t Event, hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  // TODO: Why does this check fail for OpenCL but not for Level0
  NULLCHECK(Event);

  Stream = Backend->findQueue(Stream);
  Event->recordStream(Stream);
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipEventDestroy(hipEvent_t Event) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Event);

  // instead of destroying directly, decrement refc to 1 and  let
  // StaleEventMonitor destroy this event
  Event->decreaseRefCount();
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipEventSynchronize(hipEvent_t Event) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Event);

  Event->wait();
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipEventElapsedTime(float *Ms, hipEvent_t Start, hipEvent_t Stop) {
  CHIP_TRY
  CHIPInitialize();
  if (!Ms)
    CHIPERR_LOG_AND_THROW("Ms pointer is null", hipErrorInvalidValue);
  NULLCHECK(Start, Stop);

  if (Start->getFlags().isDisableTiming() || Stop->getFlags().isDisableTiming())
    CHIPERR_LOG_AND_THROW("One of the events has timings disabled. "
                          "Unable to return elasped time",
                          hipErrorInvalidResourceHandle);

  *Ms = Start->getElapsedTime(Stop);
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipEventQuery(hipEvent_t Event) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Event);

  if (Event->isFinished())
    RETURN(hipSuccess);
  else
    RETURN(hipErrorNotReady);

  CHIP_CATCH
}

hipError_t hipMalloc(void **Ptr, size_t Size) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Ptr);

  if (Size == 0) {
    *Ptr = nullptr;
    RETURN(hipSuccess);
  }
  void *RetVal =
      Backend->getActiveContext()->allocate(Size, CHIPMemoryType::Device);
  ERROR_IF((RetVal == nullptr), hipErrorMemoryAllocation);

  *Ptr = RetVal;
  RETURN(hipSuccess);

  CHIP_CATCH
}
hipError_t hipMallocManaged(void **DevPtr, size_t Size, unsigned int Flags) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(DevPtr);

  auto FlagsParsed = CHIPManagedMemFlags{Flags};
  switch (FlagsParsed) {
  case CHIPManagedMemFlags::AttachGlobal:
    break;
  case CHIPManagedMemFlags::AttachHost:
    break;
  default:
    CHIPERR_LOG_AND_THROW("Invalid value passed for hipMallocManaged flags",
                          hipErrorInvalidValue);
  }

  if (Size < 0)
    CHIPERR_LOG_AND_THROW("Negative Allocation size",
                          hipErrorInvalidResourceHandle);

  if (Size == 0) {
    *DevPtr = nullptr;
    RETURN(hipSuccess);
  }

  void *RetVal = Backend->getActiveDevice()->getContext()->allocate(
      Size, CHIPMemoryType::Shared);
  ERROR_IF((RetVal == nullptr), hipErrorMemoryAllocation);

  *DevPtr = RetVal;
  RETURN(hipSuccess);

  CHIP_CATCH
};

DEPRECATED("use hipHostMalloc instead")
hipError_t hipMallocHost(void **Ptr, size_t Size) {
  RETURN(hipMalloc(Ptr, Size));
}

hipError_t hipHostMalloc(void **Ptr, size_t Size, unsigned int Flags) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Ptr);

  void *RetVal =
      Backend->getActiveContext()->allocate(Size, 0x1000, CHIPMemoryType::Host);
  ERROR_IF((RetVal == nullptr), hipErrorMemoryAllocation);

  *Ptr = RetVal;
  RETURN(hipSuccess);
  CHIP_CATCH
}

DEPRECATED("use hipHostMalloc instead")
hipError_t hipHostAlloc(void **Ptr, size_t Size, unsigned int Flags) {
  RETURN(hipMalloc(Ptr, Size));
}

hipError_t hipFree(void *Ptr) {
  CHIP_TRY
  CHIPInitialize();

  ERROR_IF((Ptr == nullptr), hipSuccess);
  RETURN(Backend->getActiveContext()->free(Ptr));

  CHIP_CATCH
}

hipError_t hipHostFree(void *Ptr) { RETURN(hipFree(Ptr)); }

DEPRECATED("use hipHostFree instead")
hipError_t hipFreeHost(void *Ptr) { RETURN(hipHostFree(Ptr)); }

hipError_t hipMemPrefetchAsync(const void *Ptr, size_t Count, int DstDevId,
                               hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Ptr);
  Stream = Backend->findQueue(Stream);

  ERROR_CHECK_DEVNUM(DstDevId);
  CHIPDevice *Dev = Backend->getDevices()[DstDevId];

  // Check if given Stream belongs to the requested device
  ERROR_IF(Stream->getDevice() != Dev, hipErrorInvalidDevice);
  Stream->memPrefetch(Ptr, Count);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemAdvise(const void *Ptr, size_t Count, hipMemoryAdvise Advice,
                        int DstDevId) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Ptr);

  if (Ptr == 0 || Count == 0) {
    RETURN(hipSuccess);
  }

  UNIMPLEMENTED(hipErrorNotSupported);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipHostGetDevicePointer(void **DevPtr, void *HostPtr,
                                   unsigned int Flags) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(DevPtr, HostPtr);

  auto Device = Backend->getActiveDevice();
  auto AllocInfo = Device->AllocationTracker->getByHostPtr(HostPtr);
  if (!AllocInfo) {
    logWarn("host pointer was not mapped via hipHostRegister... Returning host "
            "pointer as device pointer (in case host pointer was mapped "
            "through hipMallocShared or hipMallocHost");
    *DevPtr = HostPtr;
  } else
    *DevPtr = AllocInfo->BasePtr;

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipHostGetFlags(unsigned int *FlagsPtr, void *HostPtr) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(FlagsPtr, HostPtr);

  UNIMPLEMENTED(hipErrorNotSupported);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipHostRegister(void *HostPtr, size_t SizeBytes,
                           unsigned int Flags) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(HostPtr);

  if (Flags)
    switch (Flags) {
    case hipHostRegisterDefault:
      break;
    case hipHostRegisterMapped:
      break;
    case hipHostRegisterPortable:
      break;
    default:
      CHIPERR_LOG_AND_THROW("Invalid hipHostRegister flag passed",
                            hipErrorInvalidValue);
    }

  void *DevPtr;
  auto Err = hipMalloc(&DevPtr, SizeBytes);
  ERROR_IF(Err != hipSuccess, Err);

  // Associate the pointer
  auto Device = Backend->getActiveDevice();
  Device->AllocationTracker->registerHostPointer(HostPtr, DevPtr);

  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipHostUnregister(void *HostPtr) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(HostPtr);

  auto Device = Backend->getActiveDevice();
  auto AllocInfo = Device->AllocationTracker->getByHostPtr(HostPtr);
  auto Err = hipFree(AllocInfo->BasePtr);
  Device->AllocationTracker->unregsiterHostPointer(HostPtr);
  RETURN(Err);

  CHIP_CATCH
}

static hipError_t hipMallocPitch3D(void **Ptr, size_t *Pitch, size_t Width,
                                   size_t Height, size_t Depth) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Ptr, Pitch);

  *Pitch = ((((int)Width - 1) / SVM_ALIGNMENT) + 1) * SVM_ALIGNMENT;
  const size_t SizeBytes = (*Pitch) * Height * ((Depth == 0) ? 1 : Depth);

  void *RetVal = Backend->getActiveContext()->allocate(SizeBytes);
  ERROR_IF((RetVal == nullptr), hipErrorMemoryAllocation);

  *Ptr = RetVal;
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMallocPitch(void **Ptr, size_t *Pitch, size_t Width,
                          size_t Height) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Ptr, Pitch);

  RETURN(hipMallocPitch3D(Ptr, Pitch, Width, Height, 0));

  CHIP_CATCH
}

hipError_t hipMallocArray(hipArray **Array, const hipChannelFormatDesc *Desc,
                          size_t Width, size_t Height, unsigned int Flags) {

  // TODO: Sink the logic here into hipMalloc3DArray and call it when
  // it is implemented.
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Array, Desc);

  ERROR_IF((Width == 0), hipErrorInvalidValue);

  *Array = new hipArray;
  ERROR_IF((*Array == nullptr), hipErrorOutOfMemory);

  auto TexType = Height ? hipTextureType2D : hipTextureType1D;
  Array[0]->type = Flags;
  Array[0]->width = Width;
  Array[0]->height = Height;
  Array[0]->depth = 0;
  Array[0]->desc = *Desc;
  Array[0]->isDrv = false;
  Array[0]->textureType = TexType;
  void **Ptr = &Array[0]->data;

  size_t AllocSize =
      Width * std::max<size_t>(Height, 1) * getChannelByteSize(*Desc);

  void *RetVal = Backend->getActiveContext()->allocate(AllocSize);
  ERROR_IF((RetVal == nullptr), hipErrorMemoryAllocation);

  *Ptr = RetVal;
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipArrayCreate(hipArray **Array,
                          const HIP_ARRAY_DESCRIPTOR *AllocateArray) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Array, AllocateArray);

  ERROR_IF((AllocateArray->Width == 0), hipErrorInvalidValue);

  *Array = new hipArray;
  ERROR_IF((*Array == nullptr), hipErrorOutOfMemory);

  Array[0]->width = AllocateArray->Width;
  Array[0]->height = AllocateArray->Height;
  Array[0]->isDrv = true;
  Array[0]->textureType = hipTextureType2D;
  void **Ptr = &Array[0]->data;

  size_t Size = AllocateArray->Width;
  if (AllocateArray->Height > 0) {
    Size = Size * AllocateArray->Height;
  }
  size_t AllocSize = 0;
  switch (AllocateArray->Format) {
  case HIP_AD_FORMAT_UNSIGNED_INT8:
    AllocSize = Size * sizeof(uint8_t);
    break;
  case HIP_AD_FORMAT_UNSIGNED_INT16:
    AllocSize = Size * sizeof(uint16_t);
    break;
  case HIP_AD_FORMAT_UNSIGNED_INT32:
    AllocSize = Size * sizeof(uint32_t);
    break;
  case HIP_AD_FORMAT_SIGNED_INT8:
    AllocSize = Size * sizeof(int8_t);
    break;
  case HIP_AD_FORMAT_SIGNED_INT16:
    AllocSize = Size * sizeof(int16_t);
    break;
  case HIP_AD_FORMAT_SIGNED_INT32:
    AllocSize = Size * sizeof(int32_t);
    break;
  case HIP_AD_FORMAT_HALF:
    AllocSize = Size * sizeof(int16_t);
    break;
  case HIP_AD_FORMAT_FLOAT:
    AllocSize = Size * sizeof(float);
    break;
  default:
    AllocSize = Size;
    break;
  }

  void *RetVal = Backend->getActiveContext()->allocate(AllocSize);
  ERROR_IF((RetVal == nullptr), hipErrorMemoryAllocation);

  *Ptr = RetVal;
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipFreeArray(hipArray *Array) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Array, Array->data);

  hipError_t Err = hipFree(Array->data);
  delete Array;
  RETURN(Err);

  CHIP_CATCH
}

hipError_t hipMalloc3D(hipPitchedPtr *PitchedDevPtr, hipExtent Extent) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(PitchedDevPtr);

  ERROR_IF((Extent.width == 0 || Extent.height == 0), hipErrorInvalidValue);
  ERROR_IF((PitchedDevPtr == nullptr), hipErrorInvalidValue);

  size_t Pitch;

  hipError_t HipStatus = hipMallocPitch3D(
      &PitchedDevPtr->ptr, &Pitch, Extent.width, Extent.height, Extent.depth);

  if (HipStatus == hipSuccess) {
    PitchedDevPtr->pitch = Pitch;
    PitchedDevPtr->xsize = Extent.width;
    PitchedDevPtr->ysize = Extent.height;
  }
  RETURN(HipStatus);

  CHIP_CATCH
}

hipError_t hipMemGetInfo(size_t *Free, size_t *Total) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Free, Total);

  ERROR_IF((Total == nullptr || Free == nullptr), hipErrorInvalidValue);

  auto Dev = Backend->getActiveDevice();
  *Total = Dev->getGlobalMemSize();
  assert(Dev->getGlobalMemSize() > Dev->getUsedGlobalMem());
  *Free = Dev->getGlobalMemSize() - Dev->getUsedGlobalMem();

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemPtrGetInfo(void *Ptr, size_t *Size) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Ptr, Size);

  AllocationInfo *AllocInfo =
      Backend->getActiveDevice()->AllocationTracker->getByDevPtr(Ptr);
  *Size = AllocInfo->Size;

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemcpyAsync(void *Dst, const void *Src, size_t SizeBytes,
                          hipMemcpyKind Kind, hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst);
  CHECK(Src);

  if (SizeBytes == 0)
    RETURN(hipSuccess);

  Stream = Backend->findQueue(Stream);

  // Stream->getDevice()->initializeDeviceVariables();
  // auto TargetDevAllocTracker = Stream->getDevice()->AllocationTracker;
  // auto ActiveDevAllocTracker = Backend->getActiveDevice()->AllocationTracker;

  // if ((Kind == hipMemcpyDeviceToDevice) || (Kind == hipMemcpyDeviceToHost)) {
  //   if (!TargetDevAllocTracker->getByHostPtr(Src))
  //     RETURN(hipErrorInvalidDevicePointer);
  // }

  // if ((Kind == hipMemcpyDeviceToDevice) || (Kind == hipMemcpyHostToDevice)) {
  //   if (!ActiveDevAllocTracker->getByHostPtr(Dst))
  //     RETURN(hipErrorInvalidDevicePointer);
  // }

  if (Kind == hipMemcpyHostToHost) {
    memcpy(Dst, Src, SizeBytes);
    RETURN(hipSuccess);
  } else {
    RETURN(Stream->memCopyAsync(Dst, Src, SizeBytes));
  }

  CHIP_CATCH
}

hipError_t hipMemcpy(void *Dst, const void *Src, size_t SizeBytes,
                     hipMemcpyKind Kind) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst);
  CHECK(Src);

  if (SizeBytes == 0)
    RETURN(hipSuccess);

  if (Kind == hipMemcpyHostToHost) {
    memcpy(Dst, Src, SizeBytes);
    RETURN(hipSuccess);
  } else
    Backend->getActiveDevice()->initializeDeviceVariables();
  RETURN(Backend->getActiveQueue()->memCopy(Dst, Src, SizeBytes));

  CHIP_CATCH
}

hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t Dst, hipDeviceptr_t Src,
                              size_t SizeBytes, hipStream_t Stream) {
  RETURN(hipMemcpyAsync(Dst, Src, SizeBytes, hipMemcpyDeviceToDevice, Stream));
}

hipError_t hipMemcpyDtoD(hipDeviceptr_t Dst, hipDeviceptr_t Src,
                         size_t SizeBytes) {
  RETURN(hipMemcpy(Dst, Src, SizeBytes, hipMemcpyDeviceToDevice));
}

hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t Dst, void *Src, size_t SizeBytes,
                              hipStream_t Stream) {
  RETURN(hipMemcpyAsync(Dst, Src, SizeBytes, hipMemcpyHostToDevice, Stream));
}

hipError_t hipMemcpyHtoD(hipDeviceptr_t Dst, void *Src, size_t SizeBytes) {
  RETURN(hipMemcpy(Dst, Src, SizeBytes, hipMemcpyHostToDevice));
}

hipError_t hipMemcpyDtoHAsync(void *Dst, hipDeviceptr_t Src, size_t SizeBytes,
                              hipStream_t Stream) {
  RETURN(hipMemcpyAsync(Dst, Src, SizeBytes, hipMemcpyDeviceToHost, Stream));
}

hipError_t hipMemcpyDtoH(void *Dst, hipDeviceptr_t Src, size_t SizeBytes) {
  RETURN(hipMemcpy(Dst, Src, SizeBytes, hipMemcpyDeviceToHost));
}

hipError_t hipMemset2DAsync(void *Dst, size_t Pitch, int Value, size_t Width,
                            size_t Height, hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst);
  hipError_t Res = hipSuccess;
  for (int i = 0; i < Height; i++) {
    size_t SizeBytes = Width * sizeof(int);
    auto Offset = Pitch * i;
    char *DstP = (char *)Dst;
    auto Res = hipMemset(DstP + Offset, Value, SizeBytes);
    if (Res != hipSuccess)
      break;
  }

  RETURN(Res);
  CHIP_CATCH
}

hipError_t hipMemset2D(void *Dst, size_t Pitch, int Value, size_t Width,
                       size_t Height) {
  CHIP_TRY
  CHIPInitialize();

  auto Stream = Backend->getActiveQueue();
  auto Res = hipMemset2DAsync(Dst, Pitch, Value, Width, Height, Stream);
  Stream->finish();

  RETURN(Res);
  CHIP_CATCH
}

hipError_t hipMemset3DAsync(hipPitchedPtr PitchedDevPtr, int Value,
                            hipExtent Extent, hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(PitchedDevPtr.ptr);

  if (Extent.height * Extent.width * Extent.depth == 0)
    return hipSuccess;

  if (Extent.height > PitchedDevPtr.ysize ||
      Extent.width > PitchedDevPtr.xsize || Extent.depth > PitchedDevPtr.pitch)
    CHIPERR_LOG_AND_THROW("Extent exceeds allocation", hipErrorTbd);

  // Check if pointer inside allocation range
  auto AllocTracker = Stream->getDevice()->AllocationTracker;
  AllocationInfo *AllocInfo = AllocTracker->findBaseDevPtr(PitchedDevPtr.ptr);
  if (!AllocInfo)
    CHIPERR_LOG_AND_THROW("PitchedDevPointer not found in allocation ranges",
                          hipErrorTbd);

  // Check if extents don't overextend the allocation?

  auto Height = Extent.height;
  auto Width = Extent.width;
  auto Depth = Extent.depth;
  auto Pitch = PitchedDevPtr.pitch;
  auto Dst = PitchedDevPtr.ptr;

  if (Height * Width * Depth == 0)
    return (hipSuccess);

  hipError_t Res = hipSuccess;
  for (int i = 0; i < Depth; i++)
    for (int j = 0; j < Height; j++) {
      size_t SizeBytes = Width;
      auto Offset = i * (Pitch * Height) + j * Pitch;
      char *DstP = (char *)Dst;
      auto Res = hipMemsetAsync(DstP + Offset, Value, SizeBytes, Stream);
      if (Res != hipSuccess)
        break;
    }

  RETURN(Res);
  CHIP_CATCH
}

hipError_t hipMemset3D(hipPitchedPtr PitchedDevPtr, int Value,
                       hipExtent Extent) {
  CHIP_TRY
  CHIPInitialize();

  auto Stream = Backend->getActiveQueue();
  auto Res = hipMemset3DAsync(PitchedDevPtr, Value, Extent, Stream);
  if (Res == hipSuccess)
    Stream->finish();

  RETURN(Res);
  CHIP_CATCH
}

hipError_t hipMemsetAsync(void *Dst, int Value, size_t SizeBytes,
                          hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst);
  Stream = Backend->findQueue(Stream);

  char CharVal = Value;
  Stream->memFillAsync(Dst, SizeBytes, &CharVal, 1);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemset(void *Dst, int Value, size_t SizeBytes) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst);

  char CharVal = Value;
  Backend->getActiveDevice()->initializeDeviceVariables();
  Backend->getActiveQueue()->memFill(Dst, SizeBytes, &CharVal, 1);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemsetD8Async(hipDeviceptr_t Dest, unsigned char Value,
                            size_t Count, hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dest);
  Stream = Backend->findQueue(Stream);

  Stream->getDevice()->initializeDeviceVariables();
  Stream->memFillAsync(Dest, 1 * Count, &Value, 1);
  RETURN(hipSuccess);

  CHIP_CATCH
};

hipError_t hipMemsetD8(hipDeviceptr_t Dest, unsigned char Value,
                       size_t SizeBytes) {
  RETURN(hipMemset(Dest, Value, SizeBytes));
}

hipError_t hipMemsetD16Async(hipDeviceptr_t Dest, unsigned short Value,
                             size_t Count, hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  Stream = Backend->findQueue(Stream);

  Stream->getDevice()->initializeDeviceVariables();
  Stream->memFillAsync(Dest, 2 * Count, &Value, 2);
  RETURN(hipSuccess);

  CHIP_CATCH
}
hipError_t hipMemsetD16(hipDeviceptr_t Dest, unsigned short Value,
                        size_t Count) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dest);

  Backend->getActiveDevice()->initializeDeviceVariables();
  Backend->getActiveQueue()->memFill(Dest, 2 * Count, &Value, 2);
  RETURN(hipSuccess);

  CHIP_CATCH
};

hipError_t hipMemsetD32Async(hipDeviceptr_t Dst, int Value, size_t Count,
                             hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  Stream = Backend->findQueue(Stream);

  Stream->getDevice()->initializeDeviceVariables();
  Stream->memFillAsync(Dst, 4 * Count, &Value, 4);
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipMemsetD32(hipDeviceptr_t Dst, int Value, size_t Count) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst);

  Backend->getActiveDevice()->initializeDeviceVariables();
  Backend->getActiveQueue()->memFill(Dst, 4 * Count, &Value, 4);
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipMemcpyParam2D(const hip_Memcpy2D *PCopy) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(PCopy);

  auto Err = hipMemcpyParam2DAsync(PCopy, Backend->getActiveQueue());

  Backend->getActiveQueue()->finish();
  RETURN(Err);
  CHIP_CATCH
}

hipError_t hipMemcpy2DAsync(void *Dst, size_t DPitch, const void *Src,
                            size_t SPitch, size_t Width, size_t Height,
                            hipMemcpyKind Kind, hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst, Src);

  if (DPitch < 1)
    CHIPERR_LOG_AND_THROW("DPitch <= 0", hipErrorInvalidValue);
  if (SPitch < 1)
    CHIPERR_LOG_AND_THROW("SPitch <= 0", hipErrorInvalidValue);
  if (Width > DPitch)
    CHIPERR_LOG_AND_THROW("Width > DPitch", hipErrorInvalidValue);
  if (Height * Width == 0)
    return hipSuccess;

  Stream = Backend->findQueue(Stream);
  Backend->getActiveDevice()->initializeDeviceVariables();

  if (SPitch == 0)
    SPitch = Width;
  if (DPitch == 0)
    DPitch = Width;

  if (SPitch == 0 || DPitch == 0)
    RETURN(hipErrorInvalidValue);

  for (size_t i = 0; i < Height; ++i) {
    if (hipMemcpyAsync(Dst, Src, Width, Kind, Stream) != hipSuccess)
      RETURN(hipErrorLaunchFailure);
    Src = (char *)Src + SPitch;
    Dst = (char *)Dst + DPitch;
  }
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipMemcpy2D(void *Dst, size_t DPitch, const void *Src, size_t SPitch,
                       size_t Width, size_t Height, hipMemcpyKind Kind) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst, Src);

  hipError_t Err = hipMemcpy2DAsync(Dst, DPitch, Src, SPitch, Width, Height,
                                    Kind, Backend->getActiveQueue());
  if (Err != hipSuccess)
    return Err;

  Backend->getActiveQueue()->finish();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemcpy2DToArray(hipArray *Dst, size_t WOffset, size_t HOffset,
                              const void *Src, size_t SPitch, size_t Width,
                              size_t Height, hipMemcpyKind Kind) {
  auto Stream = Backend->getActiveQueue();

  auto Res = hipMemcpy2DToArrayAsync(Dst, WOffset, HOffset, Src, SPitch, Width,
                                     Height, Kind, Stream);
  Stream->finish();
  RETURN(Res);
}

hipError_t hipMemcpy2DToArrayAsync(hipArray *Dst, size_t WOffset,
                                   size_t HOffset, const void *Src,
                                   size_t SPitch, size_t Width, size_t Height,
                                   hipMemcpyKind Kind, hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst, Src);

  if (!Dst)
    RETURN(hipErrorUnknown);

  size_t ByteSize = getChannelByteSize(Dst->desc);

  if ((WOffset + Width > (Dst->width * ByteSize)) || Width > SPitch)
    RETURN(hipErrorInvalidValue);

  size_t SrcW = SPitch;
  size_t DstW = (Dst->width) * ByteSize;

  for (size_t Offset = HOffset; Offset < Height; ++Offset) {
    void *DstP = ((unsigned char *)Dst->data + Offset * DstW);
    void *SrcP = ((unsigned char *)Src + Offset * SrcW);
    if (hipMemcpyAsync(DstP, SrcP, Width, Kind, Stream) != hipSuccess)
      RETURN(hipErrorLaunchFailure);
  }

  RETURN(hipSuccess);
  CHIP_CATCH
};

hipError_t hipMemcpy2DFromArray(void *Dst, size_t DPitch, hipArray_const_t Src,
                                size_t WOffset, size_t HOffset, size_t Width,
                                size_t Height, hipMemcpyKind Kind) {
  auto Stream = Backend->getActiveQueue();

  auto Res = hipMemcpy2DFromArrayAsync(Dst, DPitch, Src, WOffset, HOffset,
                                       Width, Height, Kind, Stream);
  Stream->finish();
  RETURN(Res);
}
hipError_t hipMemcpy2DFromArrayAsync(void *Dst, size_t DPitch,
                                     hipArray_const_t Src, size_t WOffset,
                                     size_t HOffset, size_t Width,
                                     size_t Height, hipMemcpyKind Kind,
                                     hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst, Src);

  if (!Width || !Height)
    RETURN(hipSuccess);

  size_t ByteSize;
  if (Src) {
    switch (Src[0].desc.f) {
    case hipChannelFormatKindSigned:
      ByteSize = sizeof(int);
      break;
    case hipChannelFormatKindUnsigned:
      ByteSize = sizeof(unsigned int);
      break;
    case hipChannelFormatKindFloat:
      ByteSize = sizeof(float);
      break;
    case hipChannelFormatKindNone:
      ByteSize = sizeof(size_t);
      break;
    }
  } else {
    RETURN(hipErrorUnknown);
  }

  if ((WOffset + Width > (Src->width * ByteSize)) || Width > DPitch) {
    RETURN(hipErrorInvalidValue);
  }

  size_t DstW = DPitch;
  size_t SrcW = (Src->width) * ByteSize;

  for (size_t Offset = 0; Offset < Height; ++Offset) {
    void *SrcP = ((unsigned char *)Src->data + Offset * SrcW);
    void *DstP = ((unsigned char *)Dst + Offset * DstW);
    auto Err = hipMemcpyAsync(DstP, SrcP, Width, Kind, Stream);
    ERROR_IF(Err != hipSuccess, Err);
  }

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemcpyToArray(hipArray *Dst, size_t WOffset, size_t HOffset,
                            const void *Src, size_t Count, hipMemcpyKind Kind) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst, Src);

  void *DstP = (unsigned char *)Dst->data + WOffset;
  RETURN(hipMemcpy(DstP, Src, Count, Kind));
  CHIP_CATCH
}

hipError_t hipMemcpyFromArray(void *Dst, hipArray_const_t SrcArray,
                              size_t WOffset, size_t HOffset, size_t Count,
                              hipMemcpyKind Kind) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst, SrcArray);

  void *SrcP = (unsigned char *)SrcArray->data + WOffset;
  RETURN(hipMemcpy(Dst, SrcP, Count, Kind));

  CHIP_CATCH
}

hipError_t hipMemcpyAtoH(void *Dst, hipArray *SrcArray, size_t SrcOffset,
                         size_t Count) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst, SrcArray);
  if (SrcOffset > Count)
    CHIPERR_LOG_AND_THROW("Offset larger than count", hipErrorTbd);

  auto Info = Backend->getActiveDevice()->AllocationTracker->getByDevPtr(
      SrcArray->data);
  if (Info->Size < Count)
    CHIPERR_LOG_AND_THROW("MemCopy larger than allocated size", hipErrorTbd);

  return hipMemcpy((char *)Dst, (char *)SrcArray->data + SrcOffset, Count,
                   hipMemcpyDeviceToHost);

  CHIP_CATCH
}

hipError_t hipMemcpyHtoA(hipArray *DstArray, size_t DstOffset,
                         const void *SrcHost, size_t Count) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(SrcHost, DstArray);

  auto AllocTracker = Backend->getActiveDevice()->AllocationTracker;
  auto AllocInfo = AllocTracker->getByDevPtr(DstArray->data);
  if (!AllocInfo)
    CHIPERR_LOG_AND_THROW("Destination device pointer not allocated on device",
                          hipErrorTbd);
  if (DstOffset > AllocInfo->Size)
    CHIPERR_LOG_AND_THROW("Offset greater than allocation size", hipErrorTbd);
  if (Count > AllocInfo->Size)
    CHIPERR_LOG_AND_THROW("Copy size greater than allocation size",
                          hipErrorTbd);

  return hipMemcpy((char *)DstArray->data + DstOffset, SrcHost, Count,
                   hipMemcpyHostToDevice);

  CHIP_CATCH
}

hipError_t hipMemcpy3D(const struct hipMemcpy3DParms *Params) {
  CHIP_TRY
  CHIPInitialize();

  auto Err = hipMemcpy3DAsync(Params, Backend->getActiveQueue());

  Backend->getActiveQueue()->finish();

  RETURN(Err);
  CHIP_CATCH
}

hipError_t hipMemcpy3DAsync(const struct hipMemcpy3DParms *Params,
                            hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Params);

  const HIP_MEMCPY3D PDrvI = getDrvMemcpy3DDesc(*Params);
  const HIP_MEMCPY3D *PDrv = &PDrvI;

  size_t ByteSize;
  size_t Depth;
  size_t Height;
  size_t WidthInBytes;
  size_t SrcPitch;
  size_t DstPitch;
  void *SrcPtr;
  void *DstPtr;
  size_t YSize;

  if (Params->dstArray != nullptr) {
    if (Params->dstArray->isDrv == false) {
      switch (Params->dstArray->desc.f) {
      case hipChannelFormatKindSigned:
        ByteSize = sizeof(int);
        break;
      case hipChannelFormatKindUnsigned:
        ByteSize = sizeof(unsigned int);
        break;
      case hipChannelFormatKindFloat:
        ByteSize = sizeof(float);
        break;
      case hipChannelFormatKindNone:
        ByteSize = sizeof(size_t);
        break;
      }
      Depth = Params->extent.depth;
      Height = Params->extent.height;
      WidthInBytes = Params->extent.width * ByteSize;
      SrcPitch = Params->srcPtr.pitch;
      SrcPtr = Params->srcPtr.ptr;
      YSize = Params->srcPtr.ysize;
      DstPitch = Params->dstArray->width * ByteSize;
      DstPtr = Params->dstArray->data;
    } else {
      Depth = PDrv->Depth;
      Height = PDrv->Height;
      WidthInBytes = PDrv->WidthInBytes;
      DstPitch = PDrv->dstArray->width * 4;
      SrcPitch = PDrv->srcPitch;
      SrcPtr = (void *)PDrv->srcHost;
      YSize = PDrv->srcHeight;
      DstPtr = PDrv->dstArray->data;
    }
  } else {
    // Non Array destination
    Depth = Params->extent.depth;
    Height = Params->extent.height;
    WidthInBytes = Params->extent.width;
    SrcPitch = Params->srcPtr.pitch;
    SrcPtr = Params->srcPtr.ptr;
    DstPtr = Params->dstPtr.ptr;
    YSize = Params->srcPtr.ysize;
    DstPitch = Params->dstPtr.pitch;
  }

  if ((WidthInBytes == DstPitch) && (WidthInBytes == SrcPitch)) {
    return hipMemcpy((void *)DstPtr, (void *)SrcPtr,
                     WidthInBytes * Height * Depth, Params->kind);
  } else {
    for (size_t i = 0; i < Depth; i++) {
      for (size_t j = 0; j < Height; j++) {
        unsigned char *Src =
            (unsigned char *)SrcPtr + i * YSize * SrcPitch + j * SrcPitch;
        unsigned char *Dst =
            (unsigned char *)DstPtr + i * Height * DstPitch + j * DstPitch;
        if (hipMemcpyAsync(Dst, Src, WidthInBytes, Params->kind,
                           Backend->getActiveQueue()) != hipSuccess)
          RETURN(hipErrorLaunchFailure);
      }
    }

    Backend->getActiveQueue()->finish();
    RETURN(hipSuccess);
  }
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipFuncGetAttributes(hipFuncAttributes *Attr, const void *Func) {
  CHIP_TRY
  CHIPInitialize();

  UNIMPLEMENTED(hipErrorNotSupported);
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipModuleGetGlobal(hipDeviceptr_t *Dptr, size_t *Bytes,
                              hipModule_t Hmod, const char *Name) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dptr, Bytes, Hmod, Name);

  CHIPDeviceVar *Var = Hmod->getGlobalVar(Name);
  *Dptr = Var->getDevAddr();

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGetSymbolSize(size_t *Size, const void *Symbol) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Size, Symbol);

  CHIPDeviceVar *Var =
      Backend->getActiveDevice()->getGlobalVar((const char *)Symbol);
  ERROR_IF(!Var, hipErrorInvalidSymbol);

  *Size = Var->getSize();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemcpyToSymbol(const void *Symbol, const void *Src,
                             size_t SizeBytes, size_t Offset,
                             hipMemcpyKind Kind) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Symbol, Src);

  hipError_t Err = hipMemcpyToSymbolAsync(Symbol, Src, SizeBytes, Offset, Kind,
                                          Backend->getActiveQueue());
  if (Err != hipSuccess)
    RETURN(Err);

  Backend->getActiveQueue()->finish();

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemcpyToSymbolAsync(const void *Symbol, const void *Src,
                                  size_t SizeBytes, size_t Offset,
                                  hipMemcpyKind Kind, hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Symbol, Src);
  Stream = Backend->findQueue(Stream);
  Backend->getActiveDevice()->initializeDeviceVariables();

  CHIPDeviceVar *Var = Backend->getActiveDevice()->getGlobalVar(Symbol);
  ERROR_IF(!Var, hipErrorInvalidSymbol);
  void *DevPtr = Var->getDevAddr();
  assert(DevPtr && "Found the symbol but not its device address?");

  RETURN(hipMemcpyAsync((void *)((intptr_t)DevPtr + Offset), Src, SizeBytes,
                        Kind, Stream));
  CHIP_CATCH
}

hipError_t hipMemcpyFromSymbol(void *Dst, const void *Symbol, size_t SizeBytes,
                               size_t Offset, hipMemcpyKind Kind) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst, Symbol);

  hipError_t Err = hipMemcpyFromSymbolAsync(Dst, Symbol, SizeBytes, Offset,
                                            Kind, Backend->getActiveQueue());
  if (Err != hipSuccess)
    RETURN(Err);

  Backend->getActiveQueue()->finish();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipMemcpyFromSymbolAsync(void *Dst, const void *Symbol,
                                    size_t SizeBytes, size_t Offset,
                                    hipMemcpyKind Kind, hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Dst, Symbol);
  Stream = Backend->findQueue(Stream);

  Backend->getActiveDevice()->initializeDeviceVariables();
  CHIPDeviceVar *Var = Stream->getDevice()->getGlobalVar(Symbol);
  ERROR_IF(!Var, hipErrorInvalidSymbol);
  void *DevPtr = Var->getDevAddr();

  RETURN(hipMemcpyAsync(Dst, (void *)((intptr_t)DevPtr + Offset), SizeBytes,
                        Kind, Stream));
  CHIP_CATCH
}

hipError_t hipModuleLoadData(hipModule_t *Module, const void *Image) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Module, Image);

  UNIMPLEMENTED(hipErrorNotSupported);

  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipModuleLoadDataEx(hipModule_t *Module, const void *Image,
                               unsigned int NumOptions, hipJitOption *Options,
                               void **OptionValues) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Module, Image);
  RETURN(hipModuleLoadData(Module, Image));
  CHIP_CATCH
}

hipError_t hipLaunchKernel(const void *HostFunction, dim3 GridDim,
                           dim3 BlockDim, void **Args, size_t SharedMem,
                           hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  logDebug("hipLaunchKernel()");
  NULLCHECK(HostFunction, Args);
  Stream = Backend->findQueue(Stream);
  Backend->getActiveDevice()->initializeDeviceVariables();

  Stream->launchHostFunc(HostFunction, GridDim, BlockDim, Args, SharedMem);

  RETURN(hipSuccess);
  CHIP_CATCH
}

static unsigned getNumTextureDimensions(const hipResourceDesc *ResDesc) {
  switch (ResDesc->resType) {
  default:
    CHIPASSERT(false && "Unknown resource type.");
    return 0;
  case hipResourceTypeLinear:
    return 1;
  case hipResourceTypePitch2D:
    return 2;
  case hipResourceTypeArray: {
    switch (ResDesc->res.array.array->textureType) {
    default:
      CHIPASSERT(false && "Unknown texture type.");
      return 0;
    case hipTextureType1D:
    case hipTextureType1DLayered:
      return 1;
    case hipTextureType2D:
    case hipTextureType2DLayered:
    case hipTextureTypeCubemap:
    case hipTextureTypeCubemapLayered:
      return 2;
    case hipTextureType3D:
      return 3;
    }
  }
  }
  CHIPASSERT(false && "Unreachable.");
  return 0;
}

hipError_t
hipCreateTextureObject(hipTextureObject_t *TexObject,
                       const hipResourceDesc *ResDesc,
                       const hipTextureDesc *TexDesc,
                       const struct hipResourceViewDesc *ResViewDesc) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(TexObject, ResDesc, TexDesc);

  // Check the descriptions are valid and supported.
  switch (ResDesc->resType) {
  default:
    RETURN(hipErrorInvalidValue);
  case hipResourceTypeArray: {
    if (!ResDesc->res.array.array || !ResDesc->res.array.array->data)
      RETURN(hipErrorInvalidValue);

    break;
  }
  case hipResourceTypeLinear: {
    if (!ResDesc->res.linear.devPtr)
      RETURN(hipErrorInvalidValue);

    size_t MaxTexInTexels = Backend->getActiveDevice()->getAttr(
        hipDeviceAttributeMaxTexture1DLinear);
    size_t MaxTexInBytes =
        MaxTexInTexels * getChannelByteSize(ResDesc->res.linear.desc);
    if (ResDesc->res.linear.sizeInBytes > MaxTexInBytes)
      RETURN(hipErrorInvalidValue);

    break;
  }
  case hipResourceTypePitch2D: {
    auto &Pitch2dDesc = ResDesc->res.pitch2D;
    if (!Pitch2dDesc.devPtr)
      RETURN(hipErrorInvalidValue);

    size_t PitchInTexels =
        Pitch2dDesc.pitchInBytes / getChannelByteSize(Pitch2dDesc.desc);
    if (PitchInTexels < Pitch2dDesc.width)
      RETURN(hipErrorInvalidValue);

    size_t MaxDimSize = Backend->getActiveDevice()->getAttr(
        hipDeviceAttributeMaxTexture2DLinear);
    if (Pitch2dDesc.width > MaxDimSize || Pitch2dDesc.height > MaxDimSize ||
        PitchInTexels > MaxDimSize)
      RETURN(hipErrorInvalidValue);

    break;
  }
  };

  unsigned NumDims = getNumTextureDimensions(ResDesc);
  bool AddrModeSupported =
      (NumDims < 2 || TexDesc->addressMode[0] == TexDesc->addressMode[1]) &&
      (NumDims < 3 || TexDesc->addressMode[0] == TexDesc->addressMode[2]);
  if (!AddrModeSupported)
    CHIPERR_LOG_AND_THROW(
        "Heterogeneous texture addressing modes are not supported yet",
        hipErrorTbd);

  CHIPTexture *RetObj =
      Backend->getActiveDevice()->createTexture(ResDesc, TexDesc, ResViewDesc);
  if (RetObj != nullptr) {
    *TexObject = reinterpret_cast<hipTextureObject_t>(RetObj);
    RETURN(hipSuccess);
  } else
    RETURN(hipErrorInvalidValue);
  CHIP_CATCH
}

hipError_t hipDestroyTextureObject(hipTextureObject_t TextureObject) {
  CHIP_TRY
  CHIPInitialize();
  // TODO CRITCAL look into the define for hipTextureObject_t
  if (TextureObject == nullptr)
    RETURN(hipSuccess);
  CHIPTexture *ChipTexture = (CHIPTexture *)TextureObject;
  Backend->getActiveDevice()->destroyTexture(ChipTexture);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc *ResDesc,
                                           hipTextureObject_t TextureObject) {
  CHIP_TRY
  CHIPInitialize();
  if (TextureObject == nullptr)
    RETURN(hipErrorInvalidValue);
  CHIPTexture *ChipTexture = (CHIPTexture *)TextureObject;
  *ResDesc = ChipTexture->getResourceDesc();
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipModuleLoad(hipModule_t *Module, const char *FuncName) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Module, FuncName);

  std::ifstream ModuleFile(FuncName,
                           std::ios::in | std::ios::binary | std::ios::ate);
  ERROR_IF((ModuleFile.fail()), hipErrorFileNotFound);

  size_t Size = ModuleFile.tellg();
  char *MemBlock = new char[Size];
  ModuleFile.seekg(0, std::ios::beg);
  ModuleFile.read(MemBlock, Size);
  ModuleFile.close();
  std::string Content(MemBlock, Size);
  delete[] MemBlock;

  // CHIPModule *chip_module = new CHIPModule(std::move(content));
  for (auto &Dev : Backend->getDevices())
    Dev->addModule(&Content);
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipModuleUnload(hipModule_t Module) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Module);

  UNIMPLEMENTED(hipErrorNotSupported);
  CHIP_CATCH
}

hipError_t hipModuleGetFunction(hipFunction_t *Function, hipModule_t Module,
                                const char *Name) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(Function, Module, Name);

  CHIPKernel *Kernel = Module->getKernel(Name);

  ERROR_IF((Kernel == nullptr), hipErrorInvalidDeviceFunction);

  *Function = Kernel;
  RETURN(hipSuccess);
  CHIP_CATCH
}

hipError_t hipModuleLaunchKernel(hipFunction_t Kernel, unsigned int GridDimX,
                                 unsigned int GridDimY, unsigned int GridDimZ,
                                 unsigned int BlockDimX, unsigned int BlockDimY,
                                 unsigned int BlockDimZ,
                                 unsigned int SharedMemBytes,
                                 hipStream_t Stream, void **KernelParams,
                                 void **Extra) {
  CHIP_TRY
  CHIPInitialize();
  Stream = Backend->findQueue(Stream);

  if (SharedMemBytes > 0)
    CHIPERR_LOG_AND_THROW("Dynamic shared memory not yet implemented",
                          hipErrorLaunchFailure);

  if (KernelParams == nullptr && Extra == nullptr)
    CHIPERR_LOG_AND_THROW("either kernelParams or extra is required",
                          hipErrorLaunchFailure);

  dim3 Gird(GridDimX, GridDimY, GridDimZ);
  dim3 Block(BlockDimX, BlockDimY, BlockDimZ);

  Backend->getActiveDevice()->initializeDeviceVariables();
  if (KernelParams)
    Stream->launchWithKernelParams(Gird, Block, SharedMemBytes, KernelParams,
                                   Kernel);
  else
    Stream->launchWithExtraParams(Gird, Block, SharedMemBytes, Extra, Kernel);
  return hipSuccess;
  CHIP_CATCH
}

//*****************************************************************************
//*****************************************************************************
//*****************************************************************************

hipError_t hipLaunchByPtr(const void *HostFunction) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(HostFunction);

  logTrace("hipLaunchByPtr");
  Backend->getActiveDevice()->initializeDeviceVariables();
  CHIPExecItem *ExecItem = Backend->ChipExecStack.top();
  Backend->ChipExecStack.pop();

  auto ChipQueue = ExecItem->getQueue();
  if (!ChipQueue) {
    std::string Msg = "Tried to launch CHIPExecItem but its queue is null";
    CHIPERR_LOG_AND_THROW(Msg, hipErrorLaunchFailure);
  }

  auto ChipDev = ChipQueue->getDevice();
  auto ChipKernel = ChipDev->findKernelByHostPtr(HostFunction);
  ExecItem->setKernel(ChipKernel);

  ChipQueue->launch(ExecItem);
  return hipSuccess;
  CHIP_CATCH
}

hipError_t hipConfigureCall(dim3 GridDim, dim3 BlockDim, size_t SharedMem,
                            hipStream_t Stream) {
  CHIP_TRY
  CHIPInitialize();
  Stream = Backend->findQueue(Stream);
  logDebug("hipConfigureCall()");
  RETURN(Backend->configureCall(GridDim, BlockDim, SharedMem, Stream));
  RETURN(hipSuccess);
  CHIP_CATCH
}
extern "C" void **__hipRegisterFatBinary(const void *Data) {
  CHIP_TRY
  CHIPInitialize();

  logDebug("__hipRegisterFatBinary");

  const __CudaFatBinaryWrapper *Wrapper =
      reinterpret_cast<const __CudaFatBinaryWrapper *>(Data);
  if (Wrapper->magic != __hipFatMAGIC2 || Wrapper->version != 1) {
    CHIPERR_LOG_AND_THROW("The given object is not hipFatBinary",
                          hipErrorInitializationError);
  }

  const __ClangOffloadBundleHeader *Header = Wrapper->binary;
  std::string Magic(reinterpret_cast<const char *>(Header),
                    sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1);
  if (Magic.compare(CLANG_OFFLOAD_BUNDLER_MAGIC)) {
    CHIPERR_LOG_AND_THROW("The bundled binaries are not Clang bundled "
                          "(CLANG_OFFLOAD_BUNDLER_MAGIC is missing)",
                          hipErrorInitializationError);
  }

  std::string *Module = new std::string;
  if (!Module) {
    CHIPERR_LOG_AND_THROW("Failed to allocate memory",
                          hipErrorInitializationError);
  }

  const __ClangOffloadBundleDesc *Desc = &Header->desc[0];
  bool Found = false;

  for (uint64_t i = 0; i < Header->numBundles;
       ++i, Desc = reinterpret_cast<const __ClangOffloadBundleDesc *>(
                reinterpret_cast<uintptr_t>(&Desc->triple[0]) +
                Desc->tripleSize)) {
    std::string EntryID{&Desc->triple[0], Desc->tripleSize};
    logDebug("Bundle entry ID {} is: '{}'\n", i, EntryID);

    // SPIR-V bundle entry ID for HIP-Clang 14+. Additional components
    // are ignored for now.
    const std::string SPIRVBundleID = "hip-spirv64";
    if (EntryID.substr(0, SPIRVBundleID.size()) == SPIRVBundleID) {
      Found = true;
      break;
    }

    if (EntryID == SPIR_BUNDLE_ID) {
      Found = true;
      break;
    }

    logDebug("not a SPIR-V triple, ignoring\n");
  }

  if (!Found) {
    CHIPERR_LOG_AND_THROW("Didn't find any suitable compiled binary!",
                          hipErrorInitializationError);
  }

  const char *StringData = reinterpret_cast<const char *>(
      reinterpret_cast<uintptr_t>(Header) + (uintptr_t)Desc->offset);
  size_t StringSize = Desc->size;
  Module->assign(StringData, StringSize);

  logDebug("Register module: {} \n", (void *)Module);

  Backend->registerModuleStr(Module);

  ++NumBinariesLoaded;

  return (void **)Module;
  CHIP_CATCH_NO_RETURN
  return nullptr;
}

extern "C" void __hipUnregisterFatBinary(void *Data) {
  CHIP_TRY
  CHIPInitialize();
  std::string *Module = reinterpret_cast<std::string *>(Data);

  logDebug("Unregister module: {} \n", (void *)Module);
  Backend->unregisterModuleStr(Module);

  --NumBinariesLoaded;
  logDebug("__hipUnRegisterFatBinary {}\n", NumBinariesLoaded);

  if (NumBinariesLoaded == 0) {
    CHIPUninitialize();
  }

  CHIP_CATCH_NO_RETURN
}

extern "C" void __hipRegisterFunction(void **Data, const void *HostFunction,
                                      char *DeviceFunction,
                                      const char *DeviceName,
                                      unsigned int ThreadLimit, void *Tid,
                                      void *Bid, dim3 *BlockDim, dim3 *GridDim,
                                      int *WSize) {
  CHIP_TRY
  CHIPInitialize();
  std::string *ModuleStr = reinterpret_cast<std::string *>(Data);

  std::string DevFunc = DeviceFunction;
  logDebug("RegisterFunction on module {}\n", (void *)ModuleStr);

  logDebug("RegisterFunction on {} devices", Backend->getNumDevices());
  Backend->registerFunctionAsKernel(ModuleStr, HostFunction, DeviceName);
  CHIP_CATCH_NO_RETURN
}

hipError_t hipSetupArgument(const void *Arg, size_t Size, size_t Offset) {
  logDebug("hipSetupArgument");

  CHIP_TRY
  CHIPInitialize();
  RETURN(Backend->setArg(Arg, Size, Offset));
  RETURN(hipSuccess);
  CHIP_CATCH
}

// TODO make generic with Size and pointer
extern "C" hipError_t hipInitFromOutside(void *DriverPtr, void *DevicePtr,
                                         void *ContexPtr, void *QueuePtr) {
  logDebug("hipInitFromOutside");
  auto Modules = std::move(Backend->getDevices()[0]->getModules());
  {
    std::lock_guard<std::mutex> LockCallbacks(Backend->CallbackStackMtx);
    std::lock_guard<std::mutex> LockEvents(Backend->EventsMtx);
    delete Backend;
  }
  logDebug("deleting Backend object.");
  Backend = new CHIPBackendLevel0();

  ze_context_handle_t Ctx = (ze_context_handle_t)ContexPtr;
  ze_driver_handle_t Driver = (ze_driver_handle_t)DriverPtr;
  CHIPContextLevel0 *ChipCtx = new CHIPContextLevel0(Driver, Ctx);
  Backend->addContext(ChipCtx);

  ze_device_handle_t Dev = (ze_device_handle_t)DevicePtr;
  auto Idx = 0; // All devices should have been deleted
  CHIPDeviceLevel0 *ChipDev = new CHIPDeviceLevel0(&Dev, ChipCtx, Idx);
  ChipDev->ChipModules = Modules;
  Backend->ChipContexts[0]->getDevices().push_back(ChipDev);
  Backend->addDevice(ChipDev);

  // ze_command_queue_handle_t q = (ze_command_queue_handle_t)queuePtr;
  // CHIPQueueLevel0* chip_queue = CHIPQueueLevel0(q)
  CHIPQueueLevel0 *ChipQueue = new CHIPQueueLevel0(ChipDev);
  Backend->addQueue(ChipQueue);
  Backend->setActiveDevice(ChipDev);

  RETURN(hipSuccess);
}

extern "C" void
__hipRegisterVar(void **Data,
                 void *Var,        // The shadow variable in host code
                 char *HostName,   // Variable name in host code
                 char *DeviceName, // Variable name in device code
                 int Ext,          // Whether this variable is external
                 int Size,         // Size of the variable
                 int Constant,     // Whether this variable is constant
                 int Global        // Unknown, always 0
) {
  assert(Ext == 0);    // Device code should be fully linked so no
                       // external variables.
  assert(Global == 0); // HIP-Clang fixes this to zero.
  assert(std::string(HostName) == std::string(DeviceName));

  CHIP_TRY
  CHIPInitialize();

  logTrace("Module {}: Register variable '{}' Size={} host-addr={}",
           (void *)Data, DeviceName, Size, (void *)Var);

  std::string *ModuleStr = reinterpret_cast<std::string *>(Data);
  Backend->registerDeviceVariable(ModuleStr, Var, DeviceName, Size);

  CHIP_CATCH_NO_RETURN
}

hipError_t hipGetSymbolAddress(void **DevPtr, const void *Symbol) {
  CHIP_TRY
  CHIPInitialize();
  NULLCHECK(DevPtr, Symbol);

  Backend->getActiveDevice()->initializeDeviceVariables();
  CHIPDeviceVar *Var = Backend->getActiveDevice()->getGlobalVar(Symbol);
  ERROR_IF(!Var, hipErrorInvalidSymbol);
  *DevPtr = Var->getDevAddr();
  assert(*DevPtr);
  RETURN(hipSuccess);

  CHIP_CATCH
}

hipError_t hipIpcOpenEventHandle(hipEvent_t *Event,
                                 hipIpcEventHandle_t Handle) {
  UNIMPLEMENTED(hipErrorNotSupported);
}
hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t *Handle, hipEvent_t Event) {
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t hipModuleOccupancyMaxPotentialBlockSize(int *GridSize,
                                                   int *BlockSize,
                                                   hipFunction_t Func,
                                                   size_t DynSharedMemPerBlk,
                                                   int BlockSizeLimit);

hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(
    int *GridSize, int *BlockSize, hipFunction_t Func,
    size_t DynSharedMemPerBlk, int BlockSizeLimit, unsigned int Flags) {
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
    int *NumBlocks, hipFunction_t Func, int BlockSize,
    size_t DynSharedMemPerBlk) {
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *NumBlocks, hipFunction_t Func, int BlockSize,
    size_t DynSharedMemPerBlk, unsigned int Flags) {
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t
hipOccupancyMaxActiveBlocksPerMultiprocessor(int *NumBlocks, const void *Func,
                                             int BlockSize,
                                             size_t DynSharedMemPerBlk) {
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *NumBlocks, const void *Func, int BlockSize, size_t DynSharedMemPerBlk,
    unsigned int Flags) {
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t hipOccupancyMaxPotentialBlockSize(int *GridSize, int *BlockSize,
                                             const void *Func,
                                             size_t DynSharedMemPerBlk,
                                             int BlockSizeLimit) {
  UNIMPLEMENTED(hipErrorNotSupported);
}

hipError_t hipGetDeviceFlags(unsigned int *Flags) {
  UNIMPLEMENTED(hipErrorNotSupported);
}

/**
 * Query the hip Stream related native informtions
 */
hipError_t hipStreamGetBackendHandles(hipStream_t Stream,
                                      unsigned long *NativeInfo, int *Size) {
  logDebug("hipStreamGetBackendHandles");
  ERROR_IF((Stream == nullptr), hipErrorInvalidValue);
  Stream->getBackendHandles(NativeInfo, Size);

  return hipSuccess;
}
#endif
