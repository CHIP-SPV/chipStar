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

#include "CHIPBackendLevel0.hh"
#include "Utils.hh"

// Auto-generated header that lives in <build-dir>/bitcode.
#include "rtdevlib-modules.h"

/// Converts driver version queried from zeDriverGetProperties to string.
static std::string driverVersionToString(uint32_t DriverVersion) noexcept {
  uint32_t Build = DriverVersion & 0xffffu;
  uint32_t Minor = (DriverVersion >> 16u) & 0xffu;
  uint32_t Major = (DriverVersion >> 24u) & 0xffu;
  std::string Str;
  Str += std::to_string(Major) + ".";
  Str += std::to_string(Minor) + ".";
  Str += std::to_string(Build);
  return Str;
}

static ze_image_type_t getImageType(unsigned HipTextureID) {
  switch (HipTextureID) {
  default:
  case hipTextureTypeCubemap:
  case hipTextureTypeCubemapLayered:
    break;
  case hipTextureType1D:
    return ZE_IMAGE_TYPE_1D;
  case hipTextureType2D:
    return ZE_IMAGE_TYPE_2D;
  case hipTextureType3D:
    return ZE_IMAGE_TYPE_3D;
  case hipTextureType1DLayered:
    return ZE_IMAGE_TYPE_1DARRAY;
  case hipTextureType2DLayered:
    return ZE_IMAGE_TYPE_2DARRAY;
  }
  CHIPASSERT(false && "Unknown or unsupported HIP texture type.");
  return ZE_IMAGE_TYPE_2D;
}

#define LAYOUT_KEY(_X, _Y, _Z, _W) (_W << 24 | _Z << 16 | _Y << 8 | _X)

#define LAYOUT_KEY_FROM_FORMAT_DESC(_DESC)                                     \
  LAYOUT_KEY(_DESC.x, _DESC.y, _DESC.z, _DESC.w)

#define DEF_LAYOUT_MAP(_X, _Y, _Z, _W, _LAYOUT)                                \
  case LAYOUT_KEY(_X, _Y, _Z, _W):                                             \
    Result.layout = _LAYOUT;                                                   \
    break

static ze_image_format_t getImageFormat(hipChannelFormatDesc FormatDesc,
                                        bool NormalizedFloat) {
  bool Supported = FormatDesc.f == hipChannelFormatKindUnsigned ||
                   FormatDesc.f == hipChannelFormatKindSigned ||
                   FormatDesc.f == hipChannelFormatKindFloat;
  if (!Supported)
    CHIPERR_LOG_AND_THROW("Unsupported channel description.", hipErrorTbd);

  CHIPASSERT(FormatDesc.x < (1 << 8) && FormatDesc.y < (1 << 8) &&
             FormatDesc.z < (1 << 8) && FormatDesc.w < (1 << 8) && "Overlap");

  ze_image_format_t Result{};
  switch (LAYOUT_KEY_FROM_FORMAT_DESC(FormatDesc)) {
  default:
    CHIPERR_LOG_AND_THROW("Unsupported channel description.", hipErrorTbd);
    break;
    DEF_LAYOUT_MAP(8, 0, 0, 0, ZE_IMAGE_FORMAT_LAYOUT_8);
    DEF_LAYOUT_MAP(8, 8, 0, 0, ZE_IMAGE_FORMAT_LAYOUT_8_8);
    DEF_LAYOUT_MAP(8, 8, 8, 8, ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8);
    DEF_LAYOUT_MAP(16, 0, 0, 0, ZE_IMAGE_FORMAT_LAYOUT_16);
    DEF_LAYOUT_MAP(16, 16, 0, 0, ZE_IMAGE_FORMAT_LAYOUT_16_16);
    DEF_LAYOUT_MAP(16, 16, 16, 16, ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16);
    DEF_LAYOUT_MAP(32, 0, 0, 0, ZE_IMAGE_FORMAT_LAYOUT_32);
    DEF_LAYOUT_MAP(32, 32, 0, 0, ZE_IMAGE_FORMAT_LAYOUT_32_32);
    DEF_LAYOUT_MAP(32, 32, 32, 32, ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32);
  }

  if (FormatDesc.x > 16 && (FormatDesc.f == hipChannelFormatKindUnsigned ||
                            FormatDesc.f == hipChannelFormatKindSigned)) {
    // "Note that this [cudaTextureReadMode] applies only to 8-bit and 16-bit
    // integer formats. 32-bit integer format would not be promoted, regardless
    // of whether or not this cudaTextureDesc::readMode is set
    // cudaReadModeNormalizedFloat is specified."
    // - CUDA 11.6.1/CUDA Runtime API.
    NormalizedFloat = false;
  }

  switch (FormatDesc.f) {
  default:
    CHIPASSERT(false && "Unsupported/unimplemented format type.");
    return Result;
  case hipChannelFormatKindSigned:
    Result.type = NormalizedFloat ? ZE_IMAGE_FORMAT_TYPE_SNORM
                                  : ZE_IMAGE_FORMAT_TYPE_SINT;
    break;
  case hipChannelFormatKindUnsigned:
    Result.type = NormalizedFloat ? ZE_IMAGE_FORMAT_TYPE_UNORM
                                  : ZE_IMAGE_FORMAT_TYPE_UINT;
    break;
  case hipChannelFormatKindFloat:
    Result.type = ZE_IMAGE_FORMAT_TYPE_FLOAT;
    break;
  }

  // These fields are for swizzle descriptions.
  Result.x = ZE_IMAGE_FORMAT_SWIZZLE_R;
  Result.y = ZE_IMAGE_FORMAT_SWIZZLE_G;
  Result.z = ZE_IMAGE_FORMAT_SWIZZLE_B;
  Result.w = ZE_IMAGE_FORMAT_SWIZZLE_A;
  return Result;
}

#undef LAYOUT_KEY
#undef LAYOUT_KEY_FROM_FORMAT_DESC
#undef DEF_LAYOUT_MAP

static ze_image_desc_t getImageDescription(unsigned int TextureType,
                                           hipChannelFormatDesc Format,
                                           bool NormalizedFloat, size_t Width,
                                           size_t Height = 0,
                                           size_t Depth = 0) {
  ze_image_desc_t Result{};
  Result.stype = ZE_STRUCTURE_TYPE_IMAGE_DESC;
  Result.pNext = nullptr;
  Result.flags = 0; // Read-only
  Result.type = getImageType(TextureType);
  Result.format = getImageFormat(Format, NormalizedFloat);
  Result.width = Width;
  Result.height = Height;
  Result.depth = Depth;       // L0 spec: Ignored for non-ZE_IMAGE_TYPE_3D;
  Result.arraylevels = Depth; // L0 spec: Ignored for non-array types.
  Result.miplevels = 0;
  return Result;
}

static ze_sampler_handle_t
createSampler(CHIPDeviceLevel0 *ChipDev, const hipResourceDesc *PResDesc,
              const hipTextureDesc *PTexDesc,
              const struct hipResourceViewDesc *PResViewDesc) {
  logTrace("CHIPTextureLevel0::createSampler");

  // Identify the normalization
  ze_bool_t IsNormalized = PTexDesc->normalizedCoords != 0;

  // Identify the address mode
  ze_sampler_address_mode_t AddressMode = ZE_SAMPLER_ADDRESS_MODE_NONE;
  if (PResDesc->resType == hipResourceTypeLinear)
    // "This [address mode] is ignored if cudaResourceDesc::resType is
    // cudaResourceTypeLinear." - CUDA 11.6.1/CUDA Runtime API.
    // Effectively out-of-bound references are undefined.
    AddressMode = ZE_SAMPLER_ADDRESS_MODE_NONE;
  else if (PTexDesc->addressMode[0] == hipAddressModeClamp)
    AddressMode = ZE_SAMPLER_ADDRESS_MODE_CLAMP;
  else if (PTexDesc->addressMode[0] == hipAddressModeBorder)
    AddressMode = ZE_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  else if (!IsNormalized)
    // "... if cudaTextureDesc::normalizedCoords is set to zero,
    // cudaAddressModeWrap and cudaAddressModeMirror won't be
    // supported and will be switched to cudaAddressModeClamp."
    // - CUDA 11.6.1/CUDA Runtime API.
    AddressMode = ZE_SAMPLER_ADDRESS_MODE_CLAMP;
  else if (PTexDesc->addressMode[0] == hipAddressModeWrap)
    AddressMode = ZE_SAMPLER_ADDRESS_MODE_REPEAT;
  else if (PTexDesc->addressMode[0] == hipAddressModeMirror)
    AddressMode = ZE_SAMPLER_ADDRESS_MODE_MIRROR;
  else
    CHIPASSERT(false && "Unknown address mode!");

  // Identify the filter mode
  ze_sampler_filter_mode_t FilterMode = ZE_SAMPLER_FILTER_MODE_NEAREST;
  if (PResDesc->resType == hipResourceTypeLinear)
    // "This [filter mode] is ignored if cudaResourceDesc::resType is
    // cudaResourceTypeLinear." - CUDA 11.6.1/CUDA Runtime API.
    FilterMode = ZE_SAMPLER_FILTER_MODE_NEAREST;
  else if (PTexDesc->filterMode == hipFilterModePoint)
    FilterMode = ZE_SAMPLER_FILTER_MODE_NEAREST;
  else if (PTexDesc->filterMode == hipFilterModeLinear)
    FilterMode = ZE_SAMPLER_FILTER_MODE_LINEAR;
  else
    CHIPASSERT(false && "Unknown filter mode!");

  ze_sampler_desc_t SamplerDesc = {ZE_STRUCTURE_TYPE_SAMPLER_DESC, nullptr,
                                   AddressMode, FilterMode, IsNormalized};

  // Create LZ samler handle
  CHIPContextLevel0 *ChipCtxLz = (CHIPContextLevel0 *)ChipDev->getContext();
  ze_sampler_handle_t Sampler{};
  ze_result_t Status =
      zeSamplerCreate(ChipCtxLz->get(), ChipDev->get(), &SamplerDesc, &Sampler);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  return Sampler;
}

// CHIPEventLevel0
// ***********************************************************************

void CHIPEventLevel0::reset() {
  logTrace("CHIPEventLevel0::reset() {} msg: {} handle: {}", (void *)this, Msg,
           (void *)Event_);
  DependsOnList.clear();
  auto Status = zeEventHostReset(Event_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  {
    LOCK(EventMtx); // chipstar::Event::TrackCalled_
    TrackCalled_ = false;
    UserEvent_ = false;
    if (EventStatus_ == EVENT_STATUS_RECORDING)
      logWarn("CHIPEventLevel0::reset() called while event is recording");

    EventStatus_ = EVENT_STATUS_INIT;
    Timestamp_ = 0;
    HostTimestamp_ = 0;
    DeviceTimestamp_ = 0;
    markDeleted(false);
  }
}

ze_event_handle_t &CHIPEventLevel0::peek() {
  isDeletedSanityCheck();
  return Event_;
}

CHIPEventLevel0::~CHIPEventLevel0() {
  logTrace("~CHIPEventLevel0() {} msg: {} handle: {}", (void *)this, Msg,
           (void *)Event_);
  // if in RECORDING state, wait to finish
  if (EventStatus_ == EVENT_STATUS_RECORDING) {
    logTrace("~CHIPEventLevel0({}) waiting for event to finish", (void *)this);
    wait();
  }

  auto Status = zeEventDestroy(Event_);
  assert(Status == ZE_RESULT_SUCCESS);

  if (isUserEvent()) {
    assert(!TrackCalled_ &&
           "chipstar::Event tracking was called for a user event");
    assert(EventPoolHandle_ && "UserEvent has a null event pool handle!");
    auto Status = zeEventPoolDestroy(EventPoolHandle_);
    assert(Status == ZE_RESULT_SUCCESS);
  }

  Event_ = nullptr;
  EventPoolHandle_ = nullptr;
  EventPool = nullptr;
}

CHIPEventLevel0::CHIPEventLevel0(CHIPContextLevel0 *ChipCtx,
                                 LZEventPool *TheEventPool,
                                 unsigned int ThePoolIndex,
                                 chipstar::EventFlags Flags)
    : chipstar::Event((chipstar::Context *)(ChipCtx), Flags), Event_(nullptr),
      EventPoolHandle_(nullptr) {
  LOCK(TheEventPool->EventPoolMtx); // CHIPEventPool::EventPool_ via get()
  EventPool = TheEventPool;
  EventPoolIndex = ThePoolIndex;
  EventPoolHandle_ = TheEventPool->get();

  ze_event_desc_t EventDesc = {
      ZE_STRUCTURE_TYPE_EVENT_DESC, // stype
      nullptr,                      // pNext
      EventPoolIndex,               // index
      ZE_EVENT_SCOPE_FLAG_HOST,     // ensure memory/cache coherency required on
                                    // signaauto l
      ZE_EVENT_SCOPE_FLAG_HOST      // ensure memory coherency across device and
                                    // Host after Event_ completes
  };
  // The application must not call this function from
  // simultaneous threads with the same event pool handle.
  // Done via EventPoolMtx
  auto Status = zeEventCreate(EventPoolHandle_, &EventDesc, &Event_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd,
                              "Level Zero Event_ creation fail! ");
}

CHIPEventLevel0::CHIPEventLevel0(CHIPContextLevel0 *ChipCtx,
                                 chipstar::EventFlags Flags)
    : chipstar::Event((chipstar::Context *)(ChipCtx), Flags), Event_(nullptr),
      EventPoolHandle_(nullptr), EventPoolIndex(0), EventPool(0) {
  CHIPContextLevel0 *ZeCtx = (CHIPContextLevel0 *)ChipContext_;

  unsigned int PoolFlags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
  // if (!flags.isDisableTiming())
  //   pool_flags = pool_flags | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;

  ze_event_pool_desc_t EventPoolDesc = {
      ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, // stype
      nullptr,                           // pNext
      PoolFlags,                         // Flags
      1                                  // count
  };

  ze_result_t Status = zeEventPoolCreate(ZeCtx->get(), &EventPoolDesc, 0,
                                         nullptr, &EventPoolHandle_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd,
                              "Level Zero Event_ pool creation fail! ");

  ze_event_desc_t EventDesc = {
      ZE_STRUCTURE_TYPE_EVENT_DESC, // stype
      nullptr,                      // pNext
      0,                            // index
      ZE_EVENT_SCOPE_FLAG_HOST,     // ensure memory/cache coherency required on
                                    // signal
      ZE_EVENT_SCOPE_FLAG_HOST      // ensure memory coherency across device and
                                    // Host after Event_ completes
  };
  // The application must not call this function from
  // simultaneous threads with the same event pool handle.
  // Done. chipstar::Event pool handle is local to this event + this is
  // constructor
  Status = zeEventCreate(EventPoolHandle_, &EventDesc, &Event_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd,
                              "Level Zero Event_ creation fail! ");
}

CHIPEventLevel0::CHIPEventLevel0(CHIPContextLevel0 *ChipCtx,
                                 ze_event_handle_t NativeEvent)
    : chipstar::Event((chipstar::Context *)(ChipCtx)), Event_(NativeEvent),
      EventPoolHandle_(nullptr), EventPoolIndex(0), EventPool(nullptr) {}

void CHIPQueueLevel0::recordEvent(chipstar::Event *ChipEvent) {
  ze_result_t Status;
  auto ChipEventLz = static_cast<CHIPEventLevel0 *>(ChipEvent);

  {
    LOCK(::Backend->EventsMtx);
    ChipEventLz->reset();
  }

  auto TimestampWriteCompleteLz = std::static_pointer_cast<CHIPEventLevel0>(
      Backend->createEventShared(this->ChipContext_));
  auto TimestampMemcpyCompleteLz = std::static_pointer_cast<CHIPEventLevel0>(
      Backend->createEventShared(this->ChipContext_));

  auto [EventsToWaitOn, EventLocks] =
      addDependenciesQueueSync(TimestampWriteCompleteLz);

  Status = zeDeviceGetGlobalTimestamps(ChipDevLz_->get(),
                                       &ChipEventLz->getHostTimestamp(),
                                       &ChipEventLz->getDeviceTimestamp());
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  Borrowed<FencedCmdList> CommandList = ChipCtxLz_->getCmdListReg();

  // The application must not call this function from
  // simultaneous threads with the same command list handle.
  Status = zeCommandListAppendWriteGlobalTimestamp(
      CommandList->getCmdList(), (uint64_t *)getSharedBufffer(),
      TimestampWriteCompleteLz->peek(), EventsToWaitOn.size(),
      EventsToWaitOn.data());
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  // The application must not call this function from
  // simultaneous threads with the same command list handle.
  Status = zeCommandListAppendMemoryCopy(
      CommandList->getCmdList(), &ChipEventLz->getTimestamp(),
      getSharedBufffer(), sizeof(uint64_t), TimestampMemcpyCompleteLz->peek(),
      1, &TimestampWriteCompleteLz->peek());
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  // Prevent these events from getting collection
  ChipEventLz->addDependency(TimestampWriteCompleteLz);
  ChipEventLz->addDependency(TimestampMemcpyCompleteLz);

  Status =
      zeCommandListAppendBarrier(CommandList->getCmdList(), ChipEventLz->get(),
                                 1, &TimestampMemcpyCompleteLz->get());
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  executeCommandList(CommandList, TimestampMemcpyCompleteLz);

  ChipEventLz->setRecording();
  ChipEventLz->Msg = "recordEvent";
}

bool CHIPEventLevel0::wait() {
  LOCK(EventMtx); // chipstar::Event::EventStatus_
  isDeletedSanityCheck();
  logTrace("CHIPEventLevel0::wait(timeout: {}) {} Msg: {} Handle: {}",
           ChipEnvVars.getL0EventTimeout(), (void *)this, Msg, (void *)Event_);

  ze_result_t Status =
      zeEventHostSynchronize(Event_, ChipEnvVars.getL0EventTimeout());
  if (Status == ZE_RESULT_NOT_READY) {
    logError("CHIPEventLevel0::wait() {} Msg {} handle {} timed out after {} "
             "seconds.\n"
             "Aborting now... segfaults, illegal instructions and other "
             "undefined behavior may follow.",
             (void *)this, Msg, (void *)Event_,
             ChipEnvVars.getL0EventTimeout() / 1e9);
    std::abort();
  }

  // LOCK(EventMtx); // chipstar::Event::EventStatus_
  EventStatus_ = EVENT_STATUS_RECORDED;
  return true;
}

bool CHIPEventLevel0::updateFinishStatus(bool ThrowErrorIfNotReady) {
  isDeletedSanityCheck();
  std::string EventStatusOld, EventStatusNew;

  EventStatusOld = getEventStatusStr();

  ze_result_t Status = zeEventQueryStatus(Event_);
  if (Status == ZE_RESULT_NOT_READY && ThrowErrorIfNotReady) {
    CHIPERR_LOG_AND_THROW("chipstar::Event Not Ready", hipErrorNotReady);
  }
  if (Status == ZE_RESULT_SUCCESS) {
    EventStatus_ = EVENT_STATUS_RECORDED;
    releaseDependencies();
    doActions();
  }

  EventStatusNew = getEventStatusStr();

  if (EventStatusNew != EventStatusOld)
    return true;
  return false;
}

uint32_t CHIPEventLevel0::getValidTimestampBits() {
  isDeletedSanityCheck();
  CHIPContextLevel0 *ChipCtxLz = (CHIPContextLevel0 *)ChipContext_;
  CHIPDeviceLevel0 *ChipDevLz = (CHIPDeviceLevel0 *)ChipCtxLz->getDevice();
  auto Props = ChipDevLz->getDeviceProps();
  return Props->timestampValidBits;
}

unsigned long CHIPEventLevel0::getFinishTime() {
  isDeletedSanityCheck();
  CHIPContextLevel0 *ChipCtxLz = (CHIPContextLevel0 *)ChipContext_;
  CHIPDeviceLevel0 *ChipDevLz = (CHIPDeviceLevel0 *)ChipCtxLz->getDevice();
  auto Props = ChipDevLz->getDeviceProps();

  uint64_t TimerResolution = Props->timerResolution;
  uint32_t TimestampValidBits = Props->timestampValidBits;

  uint32_t T = (Timestamp_ & (((uint64_t)1 << TimestampValidBits) - 1));
  T = T * TimerResolution;

  return T;
}

float CHIPEventLevel0::getElapsedTime(chipstar::Event *OtherIn) {
  /**
   * Modified HIPLZ Implementation
   * https://github.com/intel/pti-gpu/blob/master/chapters/device_activity_tracing/LevelZero.md
   */
  logTrace("CHIPEventLevel0::getElapsedTime()");
  CHIPEventLevel0 *Other = (CHIPEventLevel0 *)OtherIn;
  LOCK(Backend->EventsMtx); // chipstar::Backend::Events_
  this->updateFinishStatus();
  Other->updateFinishStatus();
  if (!this->isFinished() || !Other->isFinished())
    std::abort();
  // CHIPERR_LOG_AND_ABORT("One of the events for getElapsedTime() was done
  // yet",
  //                       hipErrorNotReady);

  uint32_t Started = this->getFinishTime();
  uint32_t Finished = Other->getFinishTime();
  auto StartedCPU = this->getHostTimestamp();
  auto FinishedCPU = Other->getHostTimestamp();

  /**
   *
   * Kernel timestamps execute along a device timeline but because of limited
   * range may wrap unexpectedly. Because of this, the temporal order of two
   * kernel timestamps shouldn’t be inferred despite coincidental START/END
   * values.
   * https://spec.oneapi.io/level-zero/latest/core/PROG.html#kernel-timestamp-events
   */
  // uint64_t Elapsed = std::fabs(Finished - Started);
  // Infering temporal order anyways because hipEvent unit tests expects it

  // Resolve overflows
  // hipEventElapsed(start, stop)
  bool ReversedEvents = false;
  if (FinishedCPU < StartedCPU) {
    ReversedEvents = true;
    auto Temp = Started;
    Started = Finished;
    Finished = Temp;
  }

  if (Finished < Started) {
    const uint64_t maxValue = (1ull << getValidTimestampBits()) - 1;
    Finished = Finished + (maxValue - Started);
  }

  int64_t Elapsed = (Finished - Started);

#define NANOSECS 1000000000
  int64_t MS = (Elapsed / NANOSECS) * 1000;
  int64_t NS = Elapsed % NANOSECS;
  float FractInMS = ((float)NS) / 1000000.0f;
  auto Ms = (float)MS + FractInMS;

  if (ReversedEvents)
    Ms = Ms * -1;
  return Ms;
}

void CHIPEventLevel0::hostSignal() {
  isDeletedSanityCheck();
  logTrace("CHIPEventLevel0::hostSignal() {} Msg: {} Handle: {}", (void *)this,
           Msg, (void *)Event_);
  auto Status = zeEventHostSignal(Event_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  LOCK(EventMtx); // chipstar::Event::EventStatus_
  EventStatus_ = EVENT_STATUS_RECORDED;
}

// End CHIPEventLevel0

// CHIPCallbackDataLevel0
// ***********************************************************************

CHIPCallbackDataLevel0::CHIPCallbackDataLevel0(hipStreamCallback_t CallbackF,
                                               void *CallbackArgs,
                                               chipstar::Queue *ChipQueue)
    : chipstar::CallbackData(CallbackF, CallbackArgs, ChipQueue) {
  LOCK(Backend->BackendMtx) // ensure callback enqueues are submitted as one

  auto BackendLz = static_cast<CHIPBackendLevel0 *>(Backend);
  auto ChipQueueLz = static_cast<CHIPQueueLevel0 *>(ChipQueue);
  auto ChipContextLz =
      static_cast<CHIPContextLevel0 *>(ChipQueue->getContext());

  Borrowed<FencedCmdList> CommandList = ChipContextLz->getCmdListReg();

  // GpuReady syncs with previous events
  GpuReady = BackendLz->createEventShared(ChipContextLz);
  GpuReady->Msg = "GpuReady";
  auto GpuReadyLz = std::static_pointer_cast<CHIPEventLevel0>(GpuReady);
  auto [QueueSyncEvents, EventLocks] =
      ChipQueueLz->addDependenciesQueueSync(GpuReady);
  // Add a barrier so that it signals
  auto Status = zeCommandListAppendBarrier(
      CommandList->getCmdList(), GpuReadyLz->get(), QueueSyncEvents.size(),
      QueueSyncEvents.data());
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  // This will get triggered manually
  CpuCallbackComplete = BackendLz->createEventShared(ChipContextLz);
  CpuCallbackComplete->Msg = "CpuCallbackComplete";
  auto CpuCallbackCompleteLz =
      std::static_pointer_cast<CHIPEventLevel0>(CpuCallbackComplete);

  // This will get triggered when the CPU is done
  GpuAck = BackendLz->createEventShared(ChipContextLz);
  GpuAck->Msg = "GpuAck";
  auto GpuAckLz = std::static_pointer_cast<CHIPEventLevel0>(GpuAck);
  Status =
      zeCommandListAppendBarrier(CommandList->getCmdList(), GpuAckLz->get(), 1,
                                 &CpuCallbackCompleteLz->get());
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  ChipQueueLz->executeCommandList(CommandList, GpuAck);
}

// End CHIPCallbackDataLevel0

// EventMonitorLevel0
// ***********************************************************************

void CHIPEventMonitorLevel0::checkCallbacks() {
  CHIPCallbackDataLevel0 *CbData;
  LOCK(EventMonitorMtx); // chipstar::EventMonitor::Stop
  {
    LOCK(Backend->CallbackQueueMtx); // Backend::CallbackQueue

    if ((Backend->CallbackQueue.size() == 0))
      return;

    // get the callback item
    CbData = (CHIPCallbackDataLevel0 *)Backend->CallbackQueue.front();

    // Lock the item and members
    assert(CbData);
    LOCK( // Backend::CallbackQueue
        CbData->CallbackDataMtx);
    Backend->CallbackQueue.pop();

    // Update Status
    logTrace("checkCallbacks: checking event "
             "status for {}",
             static_cast<void *>(CbData->GpuReady.get()));
    CbData->GpuReady->updateFinishStatus(false);
    if (CbData->GpuReady->getEventStatus() != EVENT_STATUS_RECORDED) {
      // if not ready, push to the back
      Backend->CallbackQueue.push(CbData);
      return;
    }
  }

  CbData->execute(hipSuccess);
  CbData->CpuCallbackComplete->hostSignal();
  CbData->GpuAck->wait();

  delete CbData;
  pthread_yield();
}

void CHIPEventMonitorLevel0::checkCmdLists() {
  auto BackendLz = static_cast<CHIPBackendLevel0 *>(Backend);
  LOCK(Backend->EventsMtx);
  LOCK(BackendLz->ActiveCmdListsMtx);
  // go through CmdLists, calling isFinished on each and remove those that are
  // finished
  BackendLz->ActiveCmdLists.erase(
      std::remove_if(BackendLz->ActiveCmdLists.begin(),
                     BackendLz->ActiveCmdLists.end(),
                     [](const auto &CmdList) { return CmdList->isFinished(); }),
      BackendLz->ActiveCmdLists.end());
}

void CHIPEventMonitorLevel0::checkEvents() {
  LOCK(Backend->EventsMtx);
  for (size_t EventIdx = 0; EventIdx < Backend->Events.size(); EventIdx++) {
    std::shared_ptr<CHIPEventLevel0> ChipEventLz =
        std::static_pointer_cast<CHIPEventLevel0>(Backend->Events[EventIdx]);
    ChipEventLz->isDeletedSanityCheck();
    LOCK(ChipEventLz->EventMtx); // chipstar::Event::EventStatus_

    assert(ChipEventLz);
    assert(!ChipEventLz->isUserEvent() &&
           "User events should not appear in EventMonitorLevel0");

    // updateFinishStatus will return true upon event state change.
    ChipEventLz->updateFinishStatus(false);

    if (ChipEventLz->DependsOnList.size() == 0) {
      Backend->Events.erase(Backend->Events.begin() + EventIdx);
    }

    ChipEventLz->isDeletedSanityCheck();

    // delete the event if refcount reached 1 (this->ChipEventLz)
    if (ChipEventLz.use_count() == 1) {
      if (ChipEventLz->EventPool) {
        ChipEventLz->isDeletedSanityCheck();
        ChipEventLz->EventPool->returnEvent(ChipEventLz);
      }
    }

  } // done collecting events to delete
}

void CHIPEventMonitorLevel0::checkExit() {
  LOCK(EventMonitorMtx); // chipstar::EventMonitor::Stop
  /**
   * In the case that a user doesn't destroy all the
   * created streams, we remove the streams and outstanding events in
   * Backend::waitForThreadExit() but Backend has no knowledge of
   * EventCommandListMap
   */
  if (Stop) {
    // get current host time in seconds
    int CurrTime = std::chrono::duration_cast<std::chrono::seconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();

    // if this is the first time we are stopping, set the time
    if (TimeSinceStopRequested_ == 0)
      TimeSinceStopRequested_ = CurrTime;

    int EpasedTime = CurrTime - this->TimeSinceStopRequested_;
    bool AllEventsCleared = Backend->Events.size() == 0;
    if (AllEventsCleared)
      pthread_exit(0);

    if (EpasedTime > ChipEnvVars.getL0CollectEventsTimeout()) {
      logError("CHIPEventMonitorLevel0 stop was called but not all events "
               "have been cleared. Timeout of {} seconds has been reached.",
               ChipEnvVars.getL0CollectEventsTimeout());
      size_t MaxPrintEntries = std::min(Backend->Events.size(), size_t(10));
      for (size_t i = 0; i < MaxPrintEntries; i++) {
        auto Event = Backend->Events[i];
        auto EventLz = std::static_pointer_cast<CHIPEventLevel0>(Event);
        logError("Uncollected Backend->Events: {} {}",
                 (void *)Event.get(), Event->Msg);
      }
      pthread_exit(0);
    }

    // print only once a second to avoid saturating stdout with logs
    if (CurrTime - LastPrint_ >= 1) {
      LastPrint_ = CurrTime;
      logDebug("CHIPEventMonitorLevel0 stop was called but not all "
               "events have been cleared. Timeout of {} seconds has not "
               "been reached yet. Elapsed time: {} seconds",
               ChipEnvVars.getL0CollectEventsTimeout(), EpasedTime);
    }
  }
}

void CHIPEventMonitorLevel0::monitor() {

  // Stop is false and I have more events
  while (true) {
    usleep(200);
    checkCallbacks();
    checkEvents();
    checkCmdLists();
    checkExit();
  } // endless loop
}
// End EventMonitorLevel0

// CHIPKernelLevelZero
// ***********************************************************************

ze_kernel_handle_t &CHIPKernelLevel0::get() { return ZeKernel_; }

CHIPKernelLevel0::CHIPKernelLevel0(ze_kernel_handle_t ZeKernel,
                                   CHIPDeviceLevel0 *Dev, std::string HostFName,
                                   SPVFuncInfo *FuncInfo,
                                   CHIPModuleLevel0 *Parent)
    : Kernel(HostFName, FuncInfo), ZeKernel_(ZeKernel), Module(Parent),
      Device(Dev) {
  logTrace("CHIPKernelLevel0 constructor via ze_kernel_handle");

  ze_kernel_properties_t Props = {ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES, 0};

  ze_result_t Status = zeKernelGetProperties(ZeKernel, &Props);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  PrivateSize_ = Props.privateMemSize;
  StaticLocalSize_ = Props.localMemSize;

  // TODO there doesn't seem to exist a way to get these from L0 API
  MaxDynamicLocalSize_ =
      Device->getAttr(hipDeviceAttributeMaxSharedMemoryPerBlock) -
      StaticLocalSize_;
  MaxWorkGroupSize_ = Device->getAttr(hipDeviceAttributeMaxThreadsPerBlock);
}
// End CHIPKernelLevelZero

hipError_t CHIPKernelLevel0::getAttributes(hipFuncAttributes *Attr) {

  Attr->binaryVersion = 10;
  Attr->cacheModeCA = 0;

  Attr->constSizeBytes = 0; // TODO
  Attr->localSizeBytes = PrivateSize_;

  Attr->maxThreadsPerBlock = MaxWorkGroupSize_;
  Attr->sharedSizeBytes = StaticLocalSize_;
  Attr->maxDynamicSharedSizeBytes = MaxDynamicLocalSize_;

  Attr->numRegs = 0;
  Attr->preferredShmemCarveout = 0;
  Attr->ptxVersion = 10;
  return hipSuccess;
}

// CHIPQueueLevelZero
// ***********************************************************************

std::vector<ze_event_handle_t> CHIPQueueLevel0::getEventListHandles(
    const std::vector<std::shared_ptr<chipstar::Event>> &EventsToWaitOn) {
  std::vector<ze_event_handle_t> EventHandles(EventsToWaitOn.size());
  for (size_t i = 0; i < EventsToWaitOn.size(); i++) {
    std::shared_ptr<chipstar::Event> ChipEvent = EventsToWaitOn[i];
    std::shared_ptr<CHIPEventLevel0> ChipEventLz =
        std::static_pointer_cast<CHIPEventLevel0>(ChipEvent);
    CHIPASSERT(ChipEventLz);
    EventHandles[i] = ChipEventLz->peek();
  }
  return EventHandles;
}

CHIPQueueLevel0::~CHIPQueueLevel0() {
  logTrace("~CHIPQueueLevel0() {}", (void *)this);

  // From destructor post query only when queue is owned by CHIP
  // Non-owned command queues can be destroyed independently by the owner
  if (zeCmdQOwnership_) {
    finish(); // must finish the queue because it's possible that that there are
              // outstanding operations which have an associated
              // chipstar::Event. If we do not finish we risk the chance of
              // EventMonitor of deadlocking while waiting for queue
              // completion and subsequent event status change
  }
  updateLastEvent(nullptr); // Just in case that unique_ptr destructor calls
                            // this, the generic ~Queue() (which calls
                            // updateLastEvent(nullptr)) hasn't been called yet,
                            // and the event monitor ends up waiting forever.

  // The application must not call this function from
  // simultaneous threads with the same command queue handle.
  // Done. Destructor should not be called by multiple threads
#ifdef CHIP_DUBIOUS_LOCKS
  LOCK(Backend->DubiousLockLevel0)
#endif
  if (zeCmdQOwnership_) {
    zeCommandQueueDestroy(ZeCmdQ_);
  } else {
    logTrace("CHIP does not own cmd queue");
  }
}

std::pair<std::vector<ze_event_handle_t>, chipstar::LockGuardVector>
CHIPQueueLevel0::addDependenciesQueueSync(
    std::shared_ptr<chipstar::Event> TargetEvent) {
  auto [EventsToWaitOn, EventLocks] = getSyncQueuesLastEvents(TargetEvent);
  for (auto &Event : EventsToWaitOn)
    Event->isDeletedSanityCheck();

// check that TargetEvent is not part of EventsToWaitOn
#ifdef DEBUG
  for (auto &Event : EventsToWaitOn) {
    if (Event == TargetEvent) {
      logError("CHIPQueueLevel0::addDependenciesQueueSync() TargetEvent is "
               "part of EventsToWaitOn");
      std::abort();
    }
  }
#endif

  // Every event in EventsToWaitOn should have a dependency on MemCopyEvent so
  // that they don't get destroyed before MemCopyEvent
  for (auto &Event : EventsToWaitOn) {
    // LOCK(Event->EventMtx);
    std::static_pointer_cast<CHIPEventLevel0>(TargetEvent)
        ->addDependency(Event);
  }

  for (auto &Event : EventsToWaitOn) {
    Event->isDeletedSanityCheck();
  }

  std::vector<ze_event_handle_t> EventHandles =
      getEventListHandles(EventsToWaitOn);
  return {EventHandles, std::move(EventLocks)};
}

void CHIPQueueLevel0::addCallback(hipStreamCallback_t Callback,
                                  void *UserData) {
  chipstar::CallbackData *Callbackdata =
      Backend->createCallbackData(Callback, UserData, this);

  {
    LOCK(Backend->CallbackQueueMtx); // Backend::CallbackQueue
    Backend->CallbackQueue.push(Callbackdata);
  }

  return;
}

ze_command_list_handle_t CHIPQueueLevel0::getCmdListImm() {
  return ZeCmdListImm_;
}

std::shared_ptr<CHIPEventLevel0> CHIPContextLevel0::getEventFromPool() {
  // go through all pools and try to get an allocated event
  LOCK(ContextMtx); // Context::EventPools
  EventsRequested_++;
  std::shared_ptr<CHIPEventLevel0> Event;
  for (auto EventPool : EventPools_) {
    LOCK(EventPool->EventPoolMtx); // LZEventPool::FreeSlots_
    if (EventPool->EventAvailable()) {
      EventsReused_++;
      return EventPool->getEvent();
    }
  }

  // no events available, create new pool, get event from there and return
  logTrace("No available events found in {} event pools. Creating a new "
           "event pool",
           EventPools_.size());
  auto NewEventPool = new LZEventPool(this, EventPoolSize_);
  EventPoolSize_ *= 2;
  Event = NewEventPool->getEvent();
  EventPools_.push_back(NewEventPool);
  return Event;
}

Borrowed<FencedCmdList> CHIPContextLevel0::getCmdListReg() {
  auto ReturnToPool = [&](FencedCmdList *CmdList) -> void {
    LOCK(FencedCmdListsMtx_);
    CmdList->reset();
    FencedCmdListsPool_.emplace(CmdList);
  };

  LOCK(FencedCmdListsMtx_) // CHIPQueueLevel0::FencedCmdListsPool
  CmdListsRequested_++;
  if (!FencedCmdListsPool_.empty()) {
    CmdListsReused_++;
    auto cmdList = std::move(FencedCmdListsPool_.top());
    FencedCmdListsPool_.pop();
    return Borrowed<FencedCmdList>(cmdList.release(), ReturnToPool);
  } else {
    // If the cmd list stack for this queue was empty, create a new one
    // This cmd list will eventually return to the stack for this queue
    // via CHIPEventLevel0::dissassociateCmdList()
    auto ChipDevLz = static_cast<CHIPDeviceLevel0 *>(ChipDevice_);
    ze_context_handle_t ZeCtx = get();
    ze_device_handle_t ZeDev = ChipDevLz->get();
    ze_command_list_desc_t ZeCmdListDesc =
        ChipDevLz->getCommandListComputeDesc();

    auto CmdList = std::make_unique<FencedCmdList>(ZeDev, ZeCtx, ZeCmdListDesc);
    return Borrowed<FencedCmdList>(CmdList.release(), ReturnToPool);
  }
}

CHIPQueueLevel0::CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev)
    : CHIPQueueLevel0(ChipDev, 0, L0_DEFAULT_QUEUE_PRIORITY,
                      LevelZeroQueueType::Compute) {}

CHIPQueueLevel0::CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev,
                                 chipstar::QueueFlags Flags)
    : CHIPQueueLevel0(ChipDev, Flags, L0_DEFAULT_QUEUE_PRIORITY,
                      LevelZeroQueueType::Compute) {}

CHIPQueueLevel0::CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev,
                                 chipstar::QueueFlags Flags, int Priority)
    : CHIPQueueLevel0(ChipDev, Flags, Priority, LevelZeroQueueType::Compute) {}

CHIPQueueLevel0::CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev,
                                 chipstar::QueueFlags Flags, int Priority,
                                 LevelZeroQueueType TheType)
    : Queue(ChipDev, Flags, Priority), ChipDevLz_(ChipDev),
      ChipCtxLz_(static_cast<CHIPContextLevel0 *>(ChipDev->getContext())) {
  logTrace("CHIPQueueLevel0() {}", (void *)this);
  ze_result_t Status;
  ChipDevLz_ = ChipDev;
  auto Ctx = ChipDevLz_->getContext();
  ChipCtxLz_ = (CHIPContextLevel0 *)Ctx;

  if (TheType == Compute) {
    QueueProperties_ = ChipDev->getComputeQueueProps();
    QueueDescriptor_ = ChipDev->getNextComputeQueueDesc(Priority);
    CommandListDesc_ = ChipDev->getCommandListComputeDesc();
  } else if (TheType == Copy) {
    QueueProperties_ = ChipDev->getCopyQueueProps();
    QueueDescriptor_ = ChipDev->getNextCopyQueueDesc(Priority);
    CommandListDesc_ = ChipDev->getCommandListCopyDesc();

  } else {
    CHIPERR_LOG_AND_THROW("Unknown queue type requested", hipErrorTbd);
  }
  QueueType = TheType;

  SharedBuf_ =
      ChipCtxLz_->allocateImpl(32, 8, hipMemoryType::hipMemoryTypeUnified);

  // Initialize the uint64_t part as 0
  *(uint64_t *)this->SharedBuf_ = 0;

  ZeCtx_ = ChipCtxLz_->get();
  ZeDev_ = ChipDevLz_->get();

  logTrace("CHIPQueueLevel0 constructor called via Flags and Priority");
#ifdef CHIP_DUBIOUS_LOCKS
  LOCK(Backend->DubiousLockLevel0)
#endif
  Status = zeCommandQueueCreate(ZeCtx_, ZeDev_, &QueueDescriptor_, &ZeCmdQ_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);

  initializeCmdListImm();
}

CHIPQueueLevel0::CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev,
                                 ze_command_queue_handle_t ZeCmdQ)
    : Queue(ChipDev, 0, L0_DEFAULT_QUEUE_PRIORITY) {
  ChipDevLz_ = ChipDev;
  auto Ctx = ChipDevLz_->getContext();
  ChipCtxLz_ = (CHIPContextLevel0 *)Ctx;

  QueueProperties_ = ChipDev->getComputeQueueProps();
  QueueDescriptor_ = ChipDev->getNextComputeQueueDesc();
  CommandListDesc_ = ChipDev->getCommandListComputeDesc();

  ZeCtx_ = ChipCtxLz_->get();
  ZeDev_ = ChipDevLz_->get();

  ZeCmdQ_ = ZeCmdQ;

  initializeCmdListImm();
}

void CHIPQueueLevel0::initializeCmdListImm() {
  auto Status = zeCommandListCreateImmediate(ZeCtx_, ZeDev_, &QueueDescriptor_,
                                             &ZeCmdListImm_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);
}

void CHIPDeviceLevel0::initializeQueueGroupProperties() {

  // Discover the number of command queues
  uint32_t CmdqueueGroupCount = 0;
  auto Status = zeDeviceGetCommandQueueGroupProperties(
      ZeDev_, &CmdqueueGroupCount, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  logTrace("CommandGroups found: {}", CmdqueueGroupCount);

  // Create a vector of command queue properties, fill it
  std::vector<ze_command_queue_group_properties_t> CmdqueueGroupProperties(
      CmdqueueGroupCount);

  for (uint32_t i = 0; i < CmdqueueGroupCount; i++) {
    CmdqueueGroupProperties[i] = {
        ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES, // stype
        nullptr,                                          // pNext
        0,                                                // flags
        0, // maxMemoryFillPatternSize
        0  // numQueues
    };
  }
  Status = zeDeviceGetCommandQueueGroupProperties(
      ZeDev_, &CmdqueueGroupCount, CmdqueueGroupProperties.data());
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  // Find a command queue type that support compute
  for (uint32_t i = 0; i < CmdqueueGroupCount; ++i) {
    if (ComputeQueueGroupOrdinal_ == -1 &&
        CmdqueueGroupProperties[i].flags &
            ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
      ComputeQueueGroupOrdinal_ = i;
      ComputeQueueProperties_ = CmdqueueGroupProperties[i];
      logTrace("Found compute command group");
      continue;
    }

    if (CopyQueueGroupOrdinal_ == -1 &&
        CmdqueueGroupProperties[i].flags &
            ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) {
      CopyQueueGroupOrdinal_ = i;
      CopyQueueProperties_ = CmdqueueGroupProperties[i];
      CopyQueueAvailable_ = true;
      logTrace("Found memory command group");
      continue;
    }
  }

  // initialize compute and copy list descriptors
  assert(ComputeQueueGroupOrdinal_ > -1);
  CommandListComputeDesc_ = {
      ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
      nullptr,
      (unsigned int)ComputeQueueGroupOrdinal_,
      0 /* CommandListFlags */,
  };

  if (CopyQueueAvailable_) {
    assert(CopyQueueGroupOrdinal_ > -1);
    CommandListCopyDesc_ = {
        ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
        nullptr,
        (unsigned int)CopyQueueGroupOrdinal_,
        0 /* CommandListFlags */,
    };
  }
}
ze_command_queue_desc_t CHIPDeviceLevel0::getQueueDesc_(int Priority) {
  ze_command_queue_desc_t QueueDesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                       nullptr, // pNext
                                       0,       // ordinal
                                       0,       // index
                                       0,       // flags
                                       ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
                                       ZE_COMMAND_QUEUE_PRIORITY_NORMAL};

  switch (Priority) {
  case 0:
    QueueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH;
    break;
  case 1:
    QueueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    break;
  case 2:
    QueueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW;
    break;
  default:
    CHIPERR_LOG_AND_THROW(
        "Invalid Priority range requested during L0 Queue init", hipErrorTbd);
  }

  return QueueDesc;
}

ze_command_queue_desc_t
CHIPDeviceLevel0::getNextComputeQueueDesc(int Priority) {

  assert(ComputeQueueGroupOrdinal_ > -1);
  ze_command_queue_desc_t CommandQueueComputeDesc = getQueueDesc_(Priority);
  CommandQueueComputeDesc.ordinal = ComputeQueueGroupOrdinal_;

  auto MaxQueues = ComputeQueueProperties_.numQueues;
  LOCK(NextQueueIndexMtx_); // CHIPDeviceLevel0::NextComputeQueueIndex_
  CommandQueueComputeDesc.index = NextComputeQueueIndex_;
  NextComputeQueueIndex_ = (NextComputeQueueIndex_ + 1) % MaxQueues;

  return CommandQueueComputeDesc;
}

ze_command_queue_desc_t CHIPDeviceLevel0::getNextCopyQueueDesc(int Priority) {
  assert(CopyQueueGroupOrdinal_ > -1);
  ze_command_queue_desc_t CommandQueueCopyDesc = getQueueDesc_(Priority);
  CommandQueueCopyDesc.ordinal = CopyQueueGroupOrdinal_;

  auto MaxQueues = CopyQueueProperties_.numQueues;
  LOCK(NextQueueIndexMtx_); // CHIPDeviceLevel0::NextCopyQueueIndex_
  CommandQueueCopyDesc.index = NextCopyQueueIndex_;
  NextCopyQueueIndex_ = (NextCopyQueueIndex_ + 1) % MaxQueues;

  return CommandQueueCopyDesc;
}

std::shared_ptr<chipstar::Event>
CHIPQueueLevel0::launchImpl(chipstar::ExecItem *ExecItem) {
  CHIPContextLevel0 *ChipCtxZe = (CHIPContextLevel0 *)ChipContext_;
  std::shared_ptr<chipstar::Event> LaunchEvent =
      static_cast<CHIPBackendLevel0 *>(Backend)->createEventShared(ChipCtxZe);

  CHIPKernelLevel0 *ChipKernel = (CHIPKernelLevel0 *)ExecItem->getKernel();
  LaunchEvent->Msg = "launch " + ChipKernel->getName();
  ze_kernel_handle_t KernelZe = ChipKernel->get();
  logTrace("Launching Kernel {}", ChipKernel->getName());

  {
    LOCK(ExecItem->ExecItemMtx) // required by zeKernelSetGroupSize
    // The application must not call this function from
    // simultaneous threads with the same kernel handle.
    // Done by locking ExecItemMtx
    ze_result_t Status =
        zeKernelSetGroupSize(KernelZe, ExecItem->getBlock().x,
                             ExecItem->getBlock().y, ExecItem->getBlock().z);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  }

  ExecItem->setupAllArgs();
  auto X = ExecItem->getGrid().x;
  auto Y = ExecItem->getGrid().y;
  auto Z = ExecItem->getGrid().z;
  ze_group_count_t LaunchArgs = {X, Y, Z};
  // if using immediate command lists, lock the mutex
  LOCK(CommandListMtx); // TODO this is probably not needed when using RCL
  auto CommandList = this->getCmdListImm();

  // Do we need to annotate indirect buffer accesses?
  auto *LzDev = static_cast<CHIPDeviceLevel0 *>(getDevice());
  if (!LzDev->hasOnDemandPaging()) {
    // The baseline answer is yes (unless we would know that the
    // kernel won't access buffers indirectly).
    auto Status = zeKernelSetIndirectAccess(
        KernelZe, ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE |
                      ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                                hipErrorInitializationError);
  }

  // This function may not be called from simultaneous threads with the same
  // command list handle.
  // Done via LOCK(CommandListMtx)
  auto [EventHandles, EventLocks] = addDependenciesQueueSync(LaunchEvent);
  auto Status = zeCommandListAppendLaunchKernel(
      CommandList, KernelZe, &LaunchArgs,
      std::static_pointer_cast<CHIPEventLevel0>(LaunchEvent)->peek(),
      EventHandles.size(), EventHandles.data());
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);
  executeCommandList(CommandList, LaunchEvent);

  if (std::shared_ptr<chipstar::ArgSpillBuffer> SpillBuf =
          ExecItem->getArgSpillBuffer())
    // Use an event action to prolong the lifetime of the spill buffer
    // in case the exec item gets destroyed before the kernel
    // completes (may happen when called from Queue::launchKernel()).
    std::static_pointer_cast<CHIPEventLevel0>(LaunchEvent)
        ->addAction([=]() -> void { auto Tmp = SpillBuf; });

  return LaunchEvent;
}

std::shared_ptr<chipstar::Event>
CHIPQueueLevel0::memFillAsyncImpl(void *Dst, size_t Size, const void *Pattern,
                                  size_t PatternSize) {
  CHIPContextLevel0 *ChipCtxZe = (CHIPContextLevel0 *)ChipContext_;
  std::shared_ptr<chipstar::Event> MemFillEvent =
      static_cast<CHIPBackendLevel0 *>(Backend)->createEventShared(ChipCtxZe);
  MemFillEvent->Msg = "memFill";

  // Check that requested pattern is a power of 2
  if (std::ceil(log2(PatternSize)) != std::floor(log2(PatternSize))) {
    logCritical("PatternSize: {} Max: {}", PatternSize,
                getMaxMemoryFillPatternSize());
    CHIPERR_LOG_AND_THROW("MemFill PatternSize is not a power of 2",
                          hipErrorTbd);
  }

  // Check that requested pattern is not too long for this queue
  if (PatternSize > getMaxMemoryFillPatternSize()) {
    logCritical("PatternSize: {} Max: {}", PatternSize,
                getMaxMemoryFillPatternSize());
    CHIPERR_LOG_AND_THROW("MemFill PatternSize exceeds the max for this queue",
                          hipErrorTbd);
  }

  LOCK(CommandListMtx);
  auto CommandList = this->getCmdListImm();
  // The application must not call this function from
  // simultaneous threads with the same command list handle.
  // Done via LOCK(CommandListMtx)
  auto [EventHandles, EventLocks] = addDependenciesQueueSync(MemFillEvent);
  ze_result_t Status = zeCommandListAppendMemoryFill(
      CommandList, Dst, Pattern, PatternSize, Size,
      std::static_pointer_cast<CHIPEventLevel0>(MemFillEvent)->peek(),
      EventHandles.size(), EventHandles.data());
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  executeCommandList(CommandList, MemFillEvent);

  return MemFillEvent;
};

std::shared_ptr<chipstar::Event>
CHIPQueueLevel0::memCopy2DAsyncImpl(void *Dst, size_t Dpitch, const void *Src,
                                    size_t Spitch, size_t Width,
                                    size_t Height) {
  return memCopy3DAsyncImpl(Dst, Dpitch, 0, Src, Spitch, 0, Width, Height, 0);
};

std::shared_ptr<chipstar::Event> CHIPQueueLevel0::memCopy3DAsyncImpl(
    void *Dst, size_t Dpitch, size_t Dspitch, const void *Src, size_t Spitch,
    size_t Sspitch, size_t Width, size_t Height, size_t Depth) {
  CHIPContextLevel0 *ChipCtxZe = (CHIPContextLevel0 *)ChipContext_;
  std::shared_ptr<chipstar::Event> MemCopyRegionEvent =
      static_cast<CHIPBackendLevel0 *>(Backend)->createEventShared(ChipCtxZe);
  MemCopyRegionEvent->Msg = "memCopy3DAsync";

  ze_copy_region_t DstRegion;
  DstRegion.originX = 0;
  DstRegion.originY = 0;
  DstRegion.originZ = 0;
  DstRegion.width = Width;
  DstRegion.height = Height;
  DstRegion.depth = Depth;
  ze_copy_region_t SrcRegion;
  SrcRegion.originX = 0;
  SrcRegion.originY = 0;
  SrcRegion.originZ = 0;
  SrcRegion.width = Width;
  SrcRegion.height = Height;
  SrcRegion.depth = Depth;
  LOCK(CommandListMtx);
  auto CommandList = this->getCmdListImm();
  // The application must not call this function from
  // simultaneous threads with the same command list handle.
  // Done via LOCK(CommandListMtx)
  auto [EventHandles, EventLocks] =
      addDependenciesQueueSync(MemCopyRegionEvent);

  ze_result_t Status = zeCommandListAppendMemoryCopyRegion(
      CommandList, Dst, &DstRegion, Dpitch, Dspitch, Src, &SrcRegion, Spitch,
      Sspitch,
      std::static_pointer_cast<CHIPEventLevel0>(MemCopyRegionEvent)->peek(),
      EventHandles.size(), EventHandles.data());
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  executeCommandList(CommandList, MemCopyRegionEvent);

  return MemCopyRegionEvent;
};

// Memory copy to texture object, i.e. image
std::shared_ptr<chipstar::Event>
CHIPQueueLevel0::memCopyToImage(ze_image_handle_t Image, const void *Src,
                                const chipstar::RegionDesc &SrcRegion) {
  logTrace("CHIPQueueLevel0::memCopyToImage");
  CHIPContextLevel0 *ChipCtxZe = (CHIPContextLevel0 *)ChipContext_;
  std::shared_ptr<chipstar::Event> ImageCopyEvent =
      static_cast<CHIPBackendLevel0 *>(Backend)->createEventShared(ChipCtxZe);
  ImageCopyEvent->Msg = "memCopyToImage";
  auto [EventHandles, EventLocks] = addDependenciesQueueSync(ImageCopyEvent);
  if (!SrcRegion.isPitched()) {
    LOCK(CommandListMtx);
    auto CommandList = this->getCmdListImm();
    // The application must not call this function from
    // simultaneous threads with the same command list handle.
    // Done via LOCK(CommandListMtx)
    ze_result_t Status = zeCommandListAppendImageCopyFromMemory(
        CommandList, Image, Src, 0,
        std::static_pointer_cast<CHIPEventLevel0>(ImageCopyEvent)->peek(),
        EventHandles.size(), EventHandles.data());
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
    executeCommandList(CommandList, ImageCopyEvent);

    return ImageCopyEvent;
  }

  // Copy image data row by row since level zero does not have pitched copy.
  CHIPASSERT(SrcRegion.getNumDims() == 2 &&
             "UNIMPLEMENTED: 3D pitched image copy.");
  const char *SrcRow = (const char *)Src;
  LOCK(CommandListMtx);
  auto CommandList = this->getCmdListImm();
  for (size_t Row = 0; Row < SrcRegion.Size[1]; Row++) {
    bool LastRow = Row == SrcRegion.Size[1] - 1;
    ze_image_region_t DstZeRegion{};
    DstZeRegion.originX = 0;
    DstZeRegion.originY = Row;
    DstZeRegion.originZ = 0;
    DstZeRegion.width = SrcRegion.Size[0];
    DstZeRegion.height = 1;
    DstZeRegion.depth = 1;

    // The application must not call this function from
    // simultaneous threads with the same command list handle.
    // Done via LOCK(CommandListMtx)
    ze_result_t Status = zeCommandListAppendImageCopyFromMemory(
        CommandList, Image, SrcRow, &DstZeRegion,
        LastRow
            ? std::static_pointer_cast<CHIPEventLevel0>(ImageCopyEvent)->peek()
            : nullptr,
        EventHandles.size(), EventHandles.data());
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
    SrcRow += SrcRegion.Pitch[0];
  }
  executeCommandList(CommandList, ImageCopyEvent);
  return ImageCopyEvent;
};

hipError_t CHIPQueueLevel0::getBackendHandles(uintptr_t *NativeInfo,
                                              int *NumHandles) {
  logTrace("CHIPQueueLevel0::getBackendHandles");
  if (NumHandles) {
    *NumHandles = 6;
    return hipSuccess;
  }

  // get the immediate command list handle
  NativeInfo[5] = (uintptr_t)ZeCmdListImm_;

  // Get queue handler
  NativeInfo[4] = (uintptr_t)ZeCmdQ_;

  // Get context handler
  CHIPContextLevel0 *Ctx = (CHIPContextLevel0 *)ChipContext_;
  NativeInfo[3] = (uintptr_t)Ctx->get();

  // Get device handler
  CHIPDeviceLevel0 *Dev = (CHIPDeviceLevel0 *)ChipDevice_;
  NativeInfo[2] = (uintptr_t)Dev->get();

  // Get driver handler
  NativeInfo[1] = (uintptr_t)Ctx->ZeDriver;

  NativeInfo[0] = (uintptr_t)ChipEnvVars.getBackend().str();
  return hipSuccess;
}

std::shared_ptr<chipstar::Event> CHIPQueueLevel0::enqueueMarkerImpl() {

  std::shared_ptr<chipstar::Event> MarkerEvent =
      static_cast<CHIPBackendLevel0 *>(Backend)->createEventShared(
          ChipContext_);
  addDependenciesQueueSync(MarkerEvent); // locks
  MarkerEvent->Msg = "marker";
  LOCK(CommandListMtx);
  auto CommandList = this->getCmdListImm();
  // The application must not call this function from
  // simultaneous threads with the same command list handle.
  // Done via LOCK(CommandListMtx)
  auto Status = zeCommandListAppendSignalEvent(
      CommandList,
      std::static_pointer_cast<CHIPEventLevel0>(MarkerEvent)->peek());
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  executeCommandList(CommandList, MarkerEvent);

  return MarkerEvent;
}

std::shared_ptr<chipstar::Event> CHIPQueueLevel0::enqueueBarrierImpl(
    const std::vector<std::shared_ptr<chipstar::Event>> &EventsToWaitFor) {
  std::shared_ptr<chipstar::Event> BarrierEvent =
      static_cast<CHIPBackendLevel0 *>(Backend)->createEventShared(
          ChipContext_);
  BarrierEvent->Msg = "barrier";

  auto [QueueSyncEvents, EventLocks] = addDependenciesQueueSync(BarrierEvent);
  size_t NumEventsToWaitFor = QueueSyncEvents.size() + EventsToWaitFor.size();

  ze_event_handle_t *EventHandles = nullptr;
  ze_event_handle_t SignalEventHandle = nullptr;

  SignalEventHandle =
      std::static_pointer_cast<CHIPEventLevel0>(BarrierEvent)->peek();

  if (NumEventsToWaitFor > 0) {
    EventHandles = new ze_event_handle_t[NumEventsToWaitFor];
    for (size_t i = 0; i < EventsToWaitFor.size(); i++) {
      std::shared_ptr<chipstar::Event> ChipEvent = EventsToWaitFor[i];
      std::shared_ptr<CHIPEventLevel0> ChipEventLz =
          std::static_pointer_cast<CHIPEventLevel0>(ChipEvent);
      CHIPASSERT(ChipEventLz);
      EventHandles[i] = ChipEventLz->peek();
      BarrierEvent->addDependency(ChipEventLz);
    }

    for (size_t i = 0; i < QueueSyncEvents.size(); i++) {
      EventHandles[i + EventsToWaitFor.size()] = QueueSyncEvents[i];
    }
  } // done gather Event_ handles to wait on

  // TODO Should this be memory or compute?
  LOCK(CommandListMtx);
  auto CommandList = this->getCmdListImm();
  // The application must not call this function from
  // simultaneous threads with the same command list handle.
  // Done via LOCK(CommandListMtx)
  auto Status = zeCommandListAppendBarrier(CommandList, SignalEventHandle,
                                           NumEventsToWaitFor, EventHandles);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  executeCommandList(CommandList, BarrierEvent);

  if (EventHandles)
    delete[] EventHandles;

  return BarrierEvent;
}

std::shared_ptr<chipstar::Event>
CHIPQueueLevel0::memCopyAsyncImpl(void *Dst, const void *Src, size_t Size) {
  logTrace("CHIPQueueLevel0::memCopyAsync");
  CHIPContextLevel0 *ChipCtxZe = (CHIPContextLevel0 *)ChipContext_;
  std::shared_ptr<chipstar::Event> MemCopyEvent =
      static_cast<CHIPBackendLevel0 *>(Backend)->createEventShared(ChipCtxZe);
  ze_result_t Status;
  LOCK(CommandListMtx);
  auto CommandList = this->getCmdListImm();
  // The application must not call this function from simultaneous threads with
  // the same command list handle
  // Done via LOCK(CommandListMtx)
  auto [EventHandles, EventLocks] = addDependenciesQueueSync(MemCopyEvent);
  Status = zeCommandListAppendMemoryCopy(
      CommandList, Dst, Src, Size,
      std::static_pointer_cast<CHIPEventLevel0>(MemCopyEvent)->peek(),
      EventHandles.size(), EventHandles.data());
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);
  executeCommandList(CommandList, MemCopyEvent);

  return MemCopyEvent;
}

void CHIPQueueLevel0::finish() {
#ifdef CHIP_DUBIOUS_LOCKS
  LOCK(Backend->DubiousLockLevel0)
#endif
  ze_result_t Status;
  Status = zeCommandListHostSynchronize(ZeCmdListImm_,
                                        ChipEnvVars.getL0EventTimeout());
  CHIPERR_CHECK_LOG_AND_ABORT(Status, ZE_RESULT_SUCCESS, hipErrorTbd,
                              "zeCommandListHostSynchronize timeout out");

  Status = zeCommandQueueSynchronize(ZeCmdQ_, ChipEnvVars.getL0EventTimeout());
  CHIPERR_CHECK_LOG_AND_ABORT(Status, ZE_RESULT_SUCCESS, hipErrorTbd,
                              "zeCommandQueueSynchronize timeout out");

  return;
}

void CHIPQueueLevel0::executeCommandList(
    Borrowed<FencedCmdList> &CmdList, std::shared_ptr<chipstar::Event> Event) {
  updateLastEvent(Event);
  CmdList->execute(getCmdQueue());
  auto BackendLz = static_cast<CHIPBackendLevel0 *>(Backend);
  LOCK(BackendLz->ActiveCmdListsMtx);
  BackendLz->ActiveCmdLists.push_back(std::move(CmdList));
  Backend->trackEvent(Event);
}

void CHIPQueueLevel0::executeCommandList(
    ze_command_list_handle_t &CmdList, std::shared_ptr<chipstar::Event> Event) {
  updateLastEvent(Event);
  Backend->trackEvent(Event);
}

// End CHIPQueueLevelZero

// EventPool
// ***********************************************************************
LZEventPool::LZEventPool(CHIPContextLevel0 *Ctx, unsigned int Size)
    : Ctx_(Ctx), Size_(Size) {

  unsigned int PoolFlags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
  // if (!flags.isDisableTiming())
  //   pool_flags = pool_flags | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;

  ze_event_pool_desc_t EventPoolDesc = {
      ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, // stype
      nullptr,                           // pNext
      PoolFlags,                         // Flags
      Size_                              // count
  };

  ze_result_t Status =
      zeEventPoolCreate(Ctx_->get(), &EventPoolDesc, 0, nullptr, &EventPool_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd,
                              "Level Zero Event_ pool creation fail! ");

  for (unsigned i = 0; i < Size_; i++) {
    chipstar::EventFlags Flags;
    auto NewEvent = std::make_shared<CHIPEventLevel0>(Ctx_, this, i, Flags);
    Events_.push(NewEvent);
  }
};

LZEventPool::~LZEventPool() {
  if (Backend->Events.size())
    logWarn("CHIPEventLevel0 objects still exist at the time of EventPool "
            "destruction");
  if (Backend->UserEvents.size())
    logWarn("CHIPUserEventLevel0 objects still exist at the time of EventPool "
            "destruction");

  while (Events_.size())
    Events_.pop();
  // The application must not call this function from
  // simultaneous threads with the same event pool handle.
  // Done via destructor should not be called from multiple threads
  auto Status = zeEventPoolDestroy(EventPool_);
  // '~CHIPEventLevel0' has a non-throwing exception specification
  assert(Status == ZE_RESULT_SUCCESS);
};

std::shared_ptr<CHIPEventLevel0> LZEventPool::getEvent() {
  std::shared_ptr<CHIPEventLevel0> Event;
  if (!Events_.size())
    return nullptr;

  Event = Events_.top();
  Events_.pop();

  return Event;
};

void LZEventPool::returnEvent(std::shared_ptr<CHIPEventLevel0> Event) {
  Event->isDeletedSanityCheck();
  Event->markDeleted();
  LOCK(EventPoolMtx);
  logTrace("Returning event {} handle {}", (void *)Event.get(),
           (void *)Event.get()->get());
  Events_.push(Event);
}

// End EventPool

// CHIPBackendLevel0
// ***********************************************************************
chipstar::ExecItem *CHIPBackendLevel0::createExecItem(dim3 GirdDim,
                                                      dim3 BlockDim,
                                                      size_t SharedMem,
                                                      hipStream_t ChipQueue) {
  CHIPExecItemLevel0 *ExecItem =
      new CHIPExecItemLevel0(GirdDim, BlockDim, SharedMem, ChipQueue);
  return ExecItem;
};

std::shared_ptr<chipstar::Event>
CHIPBackendLevel0::createEventShared(chipstar::Context *ChipCtx,
                                     chipstar::EventFlags Flags) {
  std::shared_ptr<chipstar::Event> Event;

  auto ZeCtx = (CHIPContextLevel0 *)ChipCtx;
  Event = ZeCtx->getEventFromPool();
  assert(Event && "LZEventPool returned a null event");

  std::static_pointer_cast<CHIPEventLevel0>(Event)->reset();
  logDebug("CHIPBackendLevel0::createEventShared: Context {} Event {}",
           (void *)ChipCtx, (void *)Event.get());
  Event->isDeletedSanityCheck();
  return Event;
}

chipstar::Event *CHIPBackendLevel0::createEvent(chipstar::Context *ChipCtx,
                                                chipstar::EventFlags Flags) {
  auto Event = new CHIPEventLevel0((CHIPContextLevel0 *)ChipCtx, Flags);
  Event->setUserEvent(true);
  logDebug("CHIPBackendLevel0::createEvent: Context {} Event {}",
           (void *)ChipCtx, (void *)Event);
  return Event;
}

void CHIPBackendLevel0::uninitialize() {
  /**
   * chipstar::Event Monitor expects to collect all events. To do this,
   * all events must reach the refcount of 0. At this point, all queues should
   * have their LastEvent as nullptr but in case a user didn't sync and destroy
   * a user-created stream, such stream might not have its LastEvent as nullptr.
   *
   * To be safe, we iterate through all the queues and update their last event.
   */
  waitForThreadExit();
  logTrace("Backend::uninitialize(): Setting the LastEvent to null for all "
           "user-created queues");

  {
    logTrace("Backend::uninitialize(): Killing EventMonitor");
    LOCK(EventMonitor_->EventMonitorMtx); // chipstar::EventMonitor::Stop
    EventMonitor_->Stop = true;
  }
  EventMonitor_->join();
  return;
}

std::string CHIPBackendLevel0::getDefaultJitFlags() {
  return std::string(
      "-cl-std=CL2.0 -cl-take-global-address -cl-match-sincospi");
}

void CHIPBackendLevel0::initializeCommon(ze_driver_handle_t ZeDriver) {
  ze_result_t Status;
  uint32_t ExtCount = 0;
  Status = zeDriverGetExtensionProperties(ZeDriver, &ExtCount, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  std::vector<ze_driver_extension_properties_t> Extensions(ExtCount);
  Status =
      zeDriverGetExtensionProperties(ZeDriver, &ExtCount, Extensions.data());
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  for (const auto &Ext : Extensions) {
    if (std::string_view(Ext.name) == "ZE_experimental_module_program")
      hasExperimentalModuleProgramExt_ = true;

    if (std::string_view(Ext.name) == "ZE_extension_float_atomics")
      hasFloatAtomics_ = true;
  }
}

void CHIPBackendLevel0::initializeImpl() {
  logTrace("CHIPBackendLevel0 Initialize");
  MinQueuePriority_ = ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH;
  ze_result_t Status;
  Status = zeInit(0);
  if (Status != ZE_RESULT_SUCCESS) {
    logCritical("Level Zero failed to initialize any devices");
    std::exit(1);
  }

  bool AnyDeviceType = false;
  ze_device_type_t ZeDeviceType;
  if (ChipEnvVars.getDevice().getType() == DeviceType::GPU) {
    ZeDeviceType = ZE_DEVICE_TYPE_GPU;
  } else if (ChipEnvVars.getDevice().getType() == DeviceType::FPGA) {
    ZeDeviceType = ZE_DEVICE_TYPE_FPGA;
  } else if (ChipEnvVars.getDevice().getType() == DeviceType::Default) {
    // For 'default' pick all devices of any type.
    AnyDeviceType = true;
  } else {
    CHIPERR_LOG_AND_THROW("CHIP_DEVICE_TYPE must be either gpu or fpga",
                          hipErrorInitializationError);
  }
  std::vector<ze_driver_handle_t> ZeDrivers;
  std::vector<ze_device_handle_t> ZeDevices;

  // Get number of drivers
  uint32_t DriverCount = 0, DeviceCount = 0;
  Status = zeDriverGet(&DriverCount, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  logTrace("Found Level0 Drivers: {}", DriverCount);
  // Resize and fill ZeDriver vector with drivers
  ZeDrivers.resize(DriverCount);
  Status = zeDriverGet(&DriverCount, ZeDrivers.data());
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  if (ChipEnvVars.getPlatformIdx() >= DriverCount) {
    CHIPERR_LOG_AND_THROW("CHIP_PLATFORM for Level0 backend must be"
                          " < number of drivers",
                          hipErrorInitializationError);
  }

  // TODO Allow for multilpe platforms(drivers)
  // TODO Check platform ID is not the same as OpenCL. You can have
  // two OCL platforms but only one level0 driver
  ze_driver_handle_t ZeDriver = ZeDrivers[ChipEnvVars.getPlatformIdx()];

  assert(ZeDriver != nullptr);
  initializeCommon(ZeDriver);

  ze_driver_properties_t DriverProps;
  DriverProps.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
  DriverProps.pNext = nullptr;
  Status = zeDriverGetProperties(ZeDriver, &DriverProps);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  logDebug("Driver version: {}",
           driverVersionToString(DriverProps.driverVersion));

  // Load devices to device vector
  Status = zeDeviceGet(ZeDriver, &DeviceCount, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  ZeDevices.resize(DeviceCount);
  Status = zeDeviceGet(ZeDriver, &DeviceCount, ZeDevices.data());
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  const ze_context_desc_t CtxDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr,
                                     0};

  ze_context_handle_t ZeCtx;
  Status = zeContextCreateEx(ZeDriver, &CtxDesc, DeviceCount, ZeDevices.data(),
                             &ZeCtx);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  CHIPContextLevel0 *ChipL0Ctx = new CHIPContextLevel0(ZeDriver, ZeCtx);
  ::Backend->addContext(ChipL0Ctx);

  // Filter in only devices of selected type and add them to the
  // backend as derivates of Device
  auto Dev = ZeDevices[ChipEnvVars.getDeviceIdx()];
  ze_device_properties_t DeviceProperties{};
  DeviceProperties.pNext = nullptr;
  DeviceProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;

  Status = zeDeviceGetProperties(Dev, &DeviceProperties);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  std::string DeviceName = DeviceProperties.name;
  logInfo("Device: {}", DeviceName);
  if (AnyDeviceType || ZeDeviceType == DeviceProperties.type) {
    CHIPDeviceLevel0 *ChipL0Dev = CHIPDeviceLevel0::create(Dev, ChipL0Ctx, 0);
    ChipL0Ctx->setDevice(ChipL0Dev);
  }

  EventMonitor_ = (CHIPEventMonitorLevel0 *)::Backend->createEventMonitor_();
}

void CHIPBackendLevel0::initializeFromNative(const uintptr_t *NativeHandles,
                                             int NumHandles) {
  logTrace("CHIPBackendLevel0 InitializeNative");
  MinQueuePriority_ = ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH;

  ze_driver_handle_t Drv = (ze_driver_handle_t)NativeHandles[0];
  ze_device_handle_t Dev = (ze_device_handle_t)NativeHandles[1];
  ze_context_handle_t Ctx = (ze_context_handle_t)NativeHandles[2];

  initializeCommon(Drv);

  CHIPContextLevel0 *ChipCtx = new CHIPContextLevel0(Drv, Ctx);
  ChipCtx->setZeContextOwnership(false);
  addContext(ChipCtx);

  CHIPDeviceLevel0 *ChipDev = CHIPDeviceLevel0::create(Dev, ChipCtx, 0);
  ChipCtx->setDevice(ChipDev);

  LOCK(::Backend->BackendMtx); // CHIPBackendLevel0::EventMonitor
  ChipDev->LegacyDefaultQueue = ChipDev->createQueue(NativeHandles, NumHandles);
  ChipDev->LegacyDefaultQueue->setDefaultLegacyQueue(true);

  EventMonitor_ = (CHIPEventMonitorLevel0 *)::Backend->createEventMonitor_();
  setActiveDevice(ChipDev);
}

hipEvent_t CHIPBackendLevel0::getHipEvent(void *NativeEvent) {
  ze_event_handle_t E = (ze_event_handle_t)NativeEvent;
  CHIPEventLevel0 *NewEvent =
      new CHIPEventLevel0((CHIPContextLevel0 *)ActiveCtx_, E);
  //   NewEvent->increaseRefCount("getHipEvent");
  return NewEvent;
}

void *CHIPBackendLevel0::getNativeEvent(hipEvent_t HipEvent) {
  CHIPEventLevel0 *E = static_cast<CHIPEventLevel0 *>(HipEvent);
  if (!E->isRecordingOrRecorded())
    return nullptr;
  // TODO should we retain here?
  return (void *)E->peek();
}

// CHIPContextLevelZero
// ***********************************************************************

void CHIPContextLevel0::freeImpl(void *Ptr) {
  LOCK(this->ContextMtx); // required by zeMemFree
  logTrace("{} CHIPContextLevel0::freeImpl({})", (void *)this, Ptr);
  // The application must not call this function from
  // simultaneous threads with the same pointer.
  // Done via ContextMtx. Too broad?
  zeMemFree(this->ZeCtx, Ptr);
}

CHIPContextLevel0::~CHIPContextLevel0() {
  logTrace("~CHIPContextLevel0() {}", (void *)this);

  // print out reuse statistics
  if (CmdListsRequested_ != 0)
    logInfo("Command list reuse: {}%",
            100 * (CmdListsReused_ / CmdListsRequested_));
  else
    logInfo("Command list reuse: N/A (No command lists requested)");

  if (EventsRequested_ != 0)
    logInfo("Events reuse: {}%", 100 * (EventsReused_ / EventsRequested_));
  else
    logInfo("Events reuse: N/A (No events requested)");

  // delete all event pools
  for (LZEventPool *Pool : EventPools_)
    delete Pool;

  if (Backend->Events.size()) {
    logWarn("Backend->Events still exist at the time of Context "
            "destruction...");
    Backend->Events.clear();
  }

  EventPools_.clear();

  // delete all devicesA
  delete static_cast<CHIPDeviceLevel0 *>(ChipDevice_);

  // The application must not call this function from
  // simultaneous threads with the same context handle.
  // Done via destructor should not be called from multiple threads
  if (ownsZeContext)
    zeContextDestroy(this->ZeCtx);
}

void *CHIPContextLevel0::allocateImpl(size_t Size, size_t Alignment,
                                      hipMemoryType MemTy,
                                      chipstar::HostAllocFlags Flags) {
  LOCK(ContextMtx);
  void *Ptr = 0;

  ze_device_mem_alloc_flags_t DeviceFlags =
      ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED;

  ze_device_mem_alloc_desc_t DmaDesc{
      /* DmaDesc.stype   = */ ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
      /* DmaDesc.pNext   = */ nullptr,
      /* DmaDesc.flags   = */ DeviceFlags,
      /* DmaDesc.ordinal = */ 0,
  };
  ze_host_mem_alloc_flags_t HostFlags = ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED;
  if (Flags.isWriteCombined())
    HostFlags += ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED;
  ze_host_mem_alloc_desc_t HmaDesc{
      /* HmaDesc.stype = */ ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
      /* HmaDesc.pNext = */ nullptr,
      /* HmaDesc.flags = */ HostFlags,
  };
  if (MemTy == hipMemoryType::hipMemoryTypeUnified) {

    // TODO Check if devices support cross-device sharing?
    // ze_device_handle_t ZeDev = ((CHIPDeviceLevel0
    // *)getDevices()[0])->get();
    ze_device_handle_t ZeDev = nullptr; // Do not associate allocation

    ze_result_t Status = zeMemAllocShared(ZeCtx, &DmaDesc, &HmaDesc, Size,
                                          Alignment, ZeDev, &Ptr);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                                hipErrorMemoryAllocation);
  } else if (MemTy == hipMemoryType::hipMemoryTypeDevice) {
    auto ChipDev = (CHIPDeviceLevel0 *)Backend->getActiveDevice();
    ze_device_handle_t ZeDev = ChipDev->get();

    ze_result_t Status =
        zeMemAllocDevice(ZeCtx, &DmaDesc, Size, Alignment, ZeDev, &Ptr);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                                hipErrorMemoryAllocation);
  } else if (MemTy == hipMemoryType::hipMemoryTypeHost) {
    // TODO Check if devices support cross-device sharing?
    ze_result_t Status = zeMemAllocHost(ZeCtx, &HmaDesc, Size, Alignment, &Ptr);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                                hipErrorMemoryAllocation);
  } else
    CHIPERR_LOG_AND_THROW("Failed to allocate memory",
                          hipErrorMemoryAllocation);

#ifdef CHIP_L0_FIRST_TOUCH
  /*
  Normally this would not be necessary but on some systems where the runtime is
  not up-to-date, this issue persists.
  https://github.com/intel/compute-runtime/issues/631
  */
  if (auto *ChipDev = static_cast<CHIPDeviceLevel0 *>(getDevice())) {
    ze_device_handle_t ZeDev = ChipDev->get();
    auto Status = zeContextMakeMemoryResident(ZeCtx, ZeDev, Ptr, Size);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                                hipErrorMemoryAllocation);
  }
#endif
  return Ptr;
}

// CHIPDeviceLevelZero
// ***********************************************************************
CHIPDeviceLevel0::CHIPDeviceLevel0(ze_device_handle_t ZeDev,
                                   CHIPContextLevel0 *ChipCtx, int Idx)
    : Device(ChipCtx, Idx), ZeDev_(ZeDev), ZeCtx_(ChipCtx->get()),
      ZeDeviceProps_() {
  initializeQueueGroupProperties();
  ZeDeviceProps_.pNext = nullptr;
  assert(Ctx_ != nullptr);
}

CHIPDeviceLevel0 *CHIPDeviceLevel0::create(ze_device_handle_t ZeDev,
                                           CHIPContextLevel0 *ChipCtx,
                                           int Idx) {
  CHIPDeviceLevel0 *Dev = new CHIPDeviceLevel0(ZeDev, ChipCtx, Idx);
  Dev->init();
  return Dev;
}

void CHIPDeviceLevel0::resetImpl() { UNIMPLEMENTED(); }

void CHIPDeviceLevel0::populateDevicePropertiesImpl() {
  ze_result_t Status = ZE_RESULT_SUCCESS;

  // Initialize members used as input for zeDeviceGet*Properties() calls.
  ZeDeviceProps_.pNext = nullptr;
  ZeDeviceProps_.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;

  ze_device_memory_properties_t DeviceMemProps;
  DeviceMemProps.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
  DeviceMemProps.pNext = nullptr;

  ze_device_compute_properties_t DeviceComputeProps;
  DeviceComputeProps.pNext = nullptr;
  DeviceMemProps.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;

  ze_device_cache_properties_t DeviceCacheProps;
  DeviceCacheProps.pNext = nullptr;
  DeviceMemProps.stype = ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES;

  std::memset(&FpAtomicProps_, 0, sizeof(FpAtomicProps_));
  FpAtomicProps_.stype = ZE_STRUCTURE_TYPE_FLOAT_ATOMIC_EXT_PROPERTIES;

  ze_device_module_properties_t DeviceModuleProps;
  bool HasFPAtomicsExt =
      static_cast<CHIPBackendLevel0 *>(Backend)->hasFloatAtomicsExt();
  DeviceModuleProps.pNext = HasFPAtomicsExt ? &FpAtomicProps_ : nullptr;
  DeviceModuleProps.stype = ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES;

  ze_device_image_properties_t DeviceImageProps;
  DeviceImageProps.pNext = nullptr;
  DeviceImageProps.stype = ZE_STRUCTURE_TYPE_DEVICE_IMAGE_PROPERTIES;

  // Query device properties
  Status = zeDeviceGetProperties(ZeDev_, &ZeDeviceProps_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);
  this->MaxMallocSize_ = ZeDeviceProps_.maxMemAllocSize;

  // Query device memory properties
  uint32_t Count = 1;
  Status = zeDeviceGetMemoryProperties(ZeDev_, &Count, &DeviceMemProps);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);

  // Query device computation properties
  Status = zeDeviceGetComputeProperties(ZeDev_, &DeviceComputeProps);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);

  // Query device cache properties
  Count = 1;
  Status = zeDeviceGetCacheProperties(ZeDev_, &Count, &DeviceCacheProps);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);

  // Query device module properties
  Status = zeDeviceGetModuleProperties(ZeDev_, &DeviceModuleProps);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);

  // Query device image properties
  Status = zeDeviceGetImageProperties(ZeDev_, &DeviceImageProps);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);

  // Copy device name
  strncpy(HipDeviceProps_.name, ZeDeviceProps_.name,
          std::min<size_t>(255, ZE_MAX_DEVICE_NAME));
  HipDeviceProps_.name[255] = 0;

  // Get total device memory
  HipDeviceProps_.totalGlobalMem = DeviceMemProps.totalSize;

  HipDeviceProps_.sharedMemPerBlock = DeviceComputeProps.maxSharedLocalMemory;
  //??? Dev.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>(&err);

  HipDeviceProps_.maxThreadsPerBlock = DeviceComputeProps.maxTotalGroupSize;
  //??? Dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&err);

  HipDeviceProps_.maxThreadsDim[0] = DeviceComputeProps.maxGroupSizeX;
  HipDeviceProps_.maxThreadsDim[1] = DeviceComputeProps.maxGroupSizeY;
  HipDeviceProps_.maxThreadsDim[2] = DeviceComputeProps.maxGroupSizeZ;

  // Maximum configured clock frequency of the device in MHz.
  HipDeviceProps_.clockRate =
      1000 * ZeDeviceProps_.coreClockRate; // deviceMemoryProps.maxClockRate;
  // Dev.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();

  HipDeviceProps_.multiProcessorCount =
      ZeDeviceProps_.numEUsPerSubslice * ZeDeviceProps_.numSubslicesPerSlice *
      ZeDeviceProps_.numSlices; // DeviceComputeProps.maxTotalGroupSize;
  //??? Dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  HipDeviceProps_.l2CacheSize = DeviceCacheProps.cacheSize;
  // Dev.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();

  // not actually correct
  HipDeviceProps_.totalConstMem = DeviceMemProps.totalSize;
  // ??? Dev.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();

  // as per gen architecture doc
  HipDeviceProps_.regsPerBlock = 4096;

  HipDeviceProps_.warpSize = CHIP_DEFAULT_WARP_SIZE;
  if (std::find(DeviceComputeProps.subGroupSizes,
                DeviceComputeProps.subGroupSizes +
                    DeviceComputeProps.numSubGroupSizes,
                CHIP_DEFAULT_WARP_SIZE) !=
      DeviceComputeProps.subGroupSizes + DeviceComputeProps.numSubGroupSizes) {
  } else {
    logWarn(
        "The device might not support subgroup size {}, warp-size sensitive "
        "kernels might not work correctly.",
        CHIP_DEFAULT_WARP_SIZE);
  }

  // Replicate from OpenCL implementation

  // HIP and LZ uses int and uint32_t, respectively, for storing the
  // group count. Clamp the group count to INT_MAX to avoid 2^31+ size
  // being interpreted as negative number.
  constexpr unsigned IntMax = std::numeric_limits<int>::max();
  HipDeviceProps_.maxGridSize[0] =
      std::min(DeviceComputeProps.maxGroupCountX, IntMax);
  HipDeviceProps_.maxGridSize[1] =
      std::min(DeviceComputeProps.maxGroupCountY, IntMax);
  HipDeviceProps_.maxGridSize[2] =
      std::min(DeviceComputeProps.maxGroupCountZ, IntMax);
  HipDeviceProps_.memoryClockRate = DeviceMemProps.maxClockRate;
  HipDeviceProps_.memoryBusWidth = DeviceMemProps.maxBusWidth;
  HipDeviceProps_.major = 2;
  HipDeviceProps_.minor = 0;

  HipDeviceProps_.maxThreadsPerMultiProcessor =
      ZeDeviceProps_.numEUsPerSubslice * ZeDeviceProps_.numThreadsPerEU; //  10;

  HipDeviceProps_.computeMode = hipComputeModeDefault;
  HipDeviceProps_.arch = {};

  HipDeviceProps_.arch.hasGlobalInt32Atomics = 1;
  HipDeviceProps_.arch.hasSharedInt32Atomics = 1;

  HipDeviceProps_.arch.hasGlobalInt64Atomics =
      (DeviceModuleProps.flags & ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS) ? 1 : 0;
  HipDeviceProps_.arch.hasSharedInt64Atomics =
      (DeviceModuleProps.flags & ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS) ? 1 : 0;

  HipDeviceProps_.arch.hasDoubles =
      (DeviceModuleProps.flags & ZE_DEVICE_MODULE_FLAG_FP64) ? 1 : 0;

  // Ballot seems to work despite not having ZE_extension_subgroups
  // being advertised.
  HipDeviceProps_.arch.hasWarpBallot = 1;

  HipDeviceProps_.clockInstructionRate = ZeDeviceProps_.coreClockRate;
  HipDeviceProps_.concurrentKernels = 1;
  HipDeviceProps_.pciDomainID = 0;
  HipDeviceProps_.pciBusID = 0x10;
  HipDeviceProps_.pciDeviceID = 0x40 + getDeviceId();
  HipDeviceProps_.isMultiGpuBoard = 0;
  HipDeviceProps_.canMapHostMemory = 1;
  HipDeviceProps_.integrated =
      (ZeDeviceProps_.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) ? 1 : 0;
  HipDeviceProps_.maxSharedMemoryPerMultiProcessor =
      DeviceComputeProps.maxSharedLocalMemory;

  // Clamp texture dimensions to [0, INT_MAX] because the return value
  // of hipDeviceGetAttribute() is int type.
  auto MaxDim0 = clampToInt(DeviceImageProps.maxImageDims1D);
  auto MaxDim1 = clampToInt(DeviceImageProps.maxImageDims2D);
  auto MaxDim2 = clampToInt(DeviceImageProps.maxImageDims3D);

  HipDeviceProps_.maxTexture1DLinear = MaxDim0;
  HipDeviceProps_.maxTexture1D = MaxDim0;
  HipDeviceProps_.maxTexture2D[0] = MaxDim0;
  HipDeviceProps_.maxTexture2D[1] = MaxDim1;
  HipDeviceProps_.maxTexture3D[0] = MaxDim0;
  HipDeviceProps_.maxTexture3D[1] = MaxDim1;
  HipDeviceProps_.maxTexture3D[2] = MaxDim2;

  // Level0 does not have alignment requirements for images that
  // clients should follow.
  HipDeviceProps_.textureAlignment = 1;
  HipDeviceProps_.texturePitchAlignment = 1;

  // Level0 devices support basic CUDA managed memory via USM,
  // but some of the functions such as prefetch and advice are unimplemented
  // in chipStar.
  HipDeviceProps_.managedMemory = 0;
  // TODO: Populate these from SVM/USM properties. Advertise the safe
  // defaults for now. Uninitialized properties cause undeterminism.
  HipDeviceProps_.directManagedMemAccessFromHost = 0;
  HipDeviceProps_.concurrentManagedAccess = 0;
  HipDeviceProps_.pageableMemoryAccess = 0;
  HipDeviceProps_.pageableMemoryAccessUsesHostPageTables = 0;

  HipDeviceProps_.cooperativeLaunch = 0;
  HipDeviceProps_.cooperativeMultiDeviceLaunch = 0;
  HipDeviceProps_.cooperativeMultiDeviceUnmatchedFunc = 0;
  HipDeviceProps_.cooperativeMultiDeviceUnmatchedGridDim = 0;
  HipDeviceProps_.cooperativeMultiDeviceUnmatchedBlockDim = 0;
  HipDeviceProps_.cooperativeMultiDeviceUnmatchedSharedMem = 0;
  HipDeviceProps_.memPitch = 1;
  HipDeviceProps_.textureAlignment = 1;
  HipDeviceProps_.texturePitchAlignment = 1;
  HipDeviceProps_.kernelExecTimeoutEnabled = 0;
  HipDeviceProps_.ECCEnabled = 0;
  HipDeviceProps_.asicRevision = 1;

  constexpr char ArchName[] = "unavailable";
  static_assert(sizeof(ArchName) <= sizeof(HipDeviceProps_.gcnArchName),
                "Buffer overflow!");
  std::strncpy(HipDeviceProps_.gcnArchName, ArchName, sizeof(ArchName));
}

chipstar::Queue *CHIPDeviceLevel0::createQueue(chipstar::QueueFlags Flags,
                                               int Priority) {
  CHIPQueueLevel0 *NewQ = new CHIPQueueLevel0(this, Flags, Priority);
  return NewQ;
}

chipstar::Queue *CHIPDeviceLevel0::createQueue(const uintptr_t *NativeHandles,
                                               int NumHandles) {
  ze_command_queue_handle_t CmdQ = (ze_command_queue_handle_t)NativeHandles[3];
  CHIPQueueLevel0 *NewQ;
  if (!CmdQ) {
    logWarn("initializeFromNative: native queue pointer is null. Creating a "
            "new queue");
    NewQ = new CHIPQueueLevel0(this, 0, 0);
  } else {
    NewQ = new CHIPQueueLevel0(this, CmdQ);
    // In this case CHIP does not own the queue hence setting right ownership
    if (NewQ != nullptr) {
      NewQ->setCmdQueueOwnership(false);
    }
  }

  return NewQ;
}

ze_image_handle_t CHIPDeviceLevel0::allocateImage(unsigned int TextureType,
                                                  hipChannelFormatDesc Format,
                                                  bool NormalizedFloat,
                                                  size_t Width, size_t Height,
                                                  size_t Depth) {
  logTrace("CHIPContextLevel0::allocateImage()");
  auto ImageDesc = getImageDescription(TextureType, Format, NormalizedFloat,
                                       Width, Height, Depth);
  ze_image_handle_t ImageHandle{};

  ze_result_t Status = zeImageCreate(ZeCtx_, ZeDev_, &ImageDesc, &ImageHandle);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorMemoryAllocation);
  return ImageHandle;
}

chipstar::Texture *CHIPDeviceLevel0::createTexture(
    const hipResourceDesc *PResDesc, const hipTextureDesc *PTexDesc,
    const struct hipResourceViewDesc *PResViewDesc) {
  logTrace("CHIPDeviceLevel0::createTexture");

  bool NormalizedFloat = PTexDesc->readMode == hipReadModeNormalizedFloat;
  auto *Q = (CHIPQueueLevel0 *)getDefaultQueue();

  ze_sampler_handle_t SamplerHandle =
      createSampler(this, PResDesc, PTexDesc, PResViewDesc);

  if (PResDesc->resType == hipResourceTypeArray) {
    hipArray *Array = PResDesc->res.array.array;
    // Checked in CHIPBindings already.
    CHIPASSERT(Array->data && "Invalid hipArray.");
    CHIPASSERT(!Array->isDrv && "Not supported/implemented yet.");
    size_t Width = Array->width;
    size_t Height = Array->height;
    size_t Depth = Array->depth;

    ze_image_handle_t ImageHandle = reinterpret_cast<ze_image_handle_t>(
        allocateImage(Array->textureType, Array->desc, NormalizedFloat, Width,
                      Height, Depth));

    auto Tex = std::make_unique<CHIPTextureLevel0>(*PResDesc, ImageHandle,
                                                   SamplerHandle);
    logTrace("Created texture: {}", (void *)Tex.get());

    chipstar::RegionDesc SrcRegion = chipstar::RegionDesc::from(*Array);
    Q->memCopyToImage(ImageHandle, Array->data, SrcRegion);
    Q->finish(); // Finish for safety.

    return Tex.release();
  }

  if (PResDesc->resType == hipResourceTypeLinear) {
    auto &Res = PResDesc->res.linear;
    auto TexelByteSize = getChannelByteSize(Res.desc);
    size_t Width = Res.sizeInBytes / TexelByteSize;

    ze_image_handle_t ImageHandle = reinterpret_cast<ze_image_handle_t>(
        allocateImage(hipTextureType1D, Res.desc, NormalizedFloat, Width));

    auto Tex = std::make_unique<CHIPTextureLevel0>(*PResDesc, ImageHandle,
                                                   SamplerHandle);
    logTrace("Created texture: {}", (void *)Tex.get());

    // Copy data to image.
    auto SrcDesc = chipstar::RegionDesc::get1DRegion(Width, TexelByteSize);
    Q->memCopyToImage(ImageHandle, Res.devPtr, SrcDesc);
    Q->finish(); // Finish for safety.

    return Tex.release();
  }

  if (PResDesc->resType == hipResourceTypePitch2D) {
    auto &Res = PResDesc->res.pitch2D;

    CHIPASSERT(Res.pitchInBytes >= Res.width); // Checked in CHIPBindings.

    ze_image_handle_t ImageHandle = reinterpret_cast<ze_image_handle_t>(
        allocateImage(hipTextureType2D, Res.desc, NormalizedFloat, Res.width,
                      Res.height));

    auto Tex = std::make_unique<CHIPTextureLevel0>(*PResDesc, ImageHandle,
                                                   SamplerHandle);
    logTrace("Created texture: {}", (void *)Tex.get());

    // Copy data to image.
    auto SrcDesc = chipstar::RegionDesc::from(*PResDesc);
    Q->memCopyToImage(ImageHandle, Res.devPtr, SrcDesc);
    Q->finish(); // Finish for safety.

    return Tex.release();
  }

  CHIPASSERT(false && "Unsupported/unimplemented texture resource type.");
  return nullptr;
}

CHIPModuleLevel0 *CHIPDeviceLevel0::compile(const SPVModule &SrcMod) {
  auto CompiledModule = std::make_unique<CHIPModuleLevel0>(SrcMod);
  CompiledModule->compile(this);
  return CompiledModule.release();
}

// Other
// ***********************************************************************
std::string resultToString(ze_result_t Status) {
  switch (Status) {
  case ZE_RESULT_SUCCESS:
    return "ZE_RESULT_SUCCESS";
  case ZE_RESULT_NOT_READY:
    return "ZE_RESULT_NOT_READY";
  case ZE_RESULT_ERROR_DEVICE_LOST:
    return "ZE_RESULT_ERROR_DEVICE_LOST";
  case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
    return "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
  case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
    return "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
  case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:
    return "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE";
  case ZE_RESULT_ERROR_MODULE_LINK_FAILURE:
    return "ZE_RESULT_ERROR_MODULE_LINK_FAILURE";
  case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS:
    return "ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS";
  case ZE_RESULT_ERROR_NOT_AVAILABLE:
    return "ZE_RESULT_ERROR_NOT_AVAILABLE";
  case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE:
    return "ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE";
  case ZE_RESULT_ERROR_UNINITIALIZED:
    return "ZE_RESULT_ERROR_UNINITIALIZED";
  case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
    return "ZE_RESULT_ERROR_UNSUPPORTED_VERSION";
  case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
    return "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
  case ZE_RESULT_ERROR_INVALID_ARGUMENT:
    return "ZE_RESULT_ERROR_INVALID_ARGUMENT";
  case ZE_RESULT_ERROR_INVALID_NULL_HANDLE:
    return "ZE_RESULT_ERROR_INVALID_NULL_HANDLE";
  case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
    return "ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE";
  case ZE_RESULT_ERROR_INVALID_NULL_POINTER:
    return "ZE_RESULT_ERROR_INVALID_NULL_POINTER";
  case ZE_RESULT_ERROR_INVALID_SIZE:
    return "ZE_RESULT_ERROR_INVALID_SIZE";
  case ZE_RESULT_ERROR_UNSUPPORTED_SIZE:
    return "ZE_RESULT_ERROR_UNSUPPORTED_SIZE";
  case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
    return "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT";
  case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
    return "ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT";
  case ZE_RESULT_ERROR_INVALID_ENUMERATION:
    return "ZE_RESULT_ERROR_INVALID_ENUMERATION";
  case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
    return "ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION";
  case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
    return "ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT";
  case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY:
    return "ZE_RESULT_ERROR_INVALID_NATIVE_BINARY";
  case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME:
    return "ZE_RESULT_ERROR_INVALID_GLOBAL_NAME";
  case ZE_RESULT_ERROR_INVALID_KERNEL_NAME:
    return "ZE_RESULT_ERROR_INVALID_KERNEL_NAME";
  case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME:
    return "ZE_RESULT_ERROR_INVALID_FUNCTION_NAME";
  case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
    return "ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION";
  case ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
    return "ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION";
  case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
    return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX";
  case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
    return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE";
  case ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
    return "ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE";
  case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED:
    return "ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED";
  case ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE:
    return "ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE";
  case ZE_RESULT_ERROR_OVERLAPPING_REGIONS:
    return "ZE_RESULT_ERROR_OVERLAPPING_REGIONS";
  case ZE_RESULT_ERROR_UNKNOWN:
    return "ZE_RESULT_ERROR_UNKNOWN";
  default:
    return "Unknown Error Code";
  }
}

// CHIPModuleLevel0
// ***********************************************************************

/// Dumps build/link log into the error log stream. The 'Log' value must be
/// valid handle. This function will destroy the log handle.
static void dumpBuildLog(ze_module_build_log_handle_t &&Log) {
  size_t LogSize;
  auto Status = zeModuleBuildLogGetString(Log, &LogSize, nullptr);
  if (Status == ZE_RESULT_SUCCESS) {
    std::vector<char> LogVec(LogSize);
    Status = zeModuleBuildLogGetString(Log, &LogSize, LogVec.data());
    if (Status == ZE_RESULT_SUCCESS)
      logError("ZE Build Log:\n{}", std::string_view(LogVec.data(), LogSize));
  }

  CHIPERR_CHECK_LOG_AND_THROW(zeModuleBuildLogDestroy(Log), ZE_RESULT_SUCCESS,
                              hipErrorTbd);
}

static ze_module_handle_t compileIL(ze_context_handle_t ZeCtx,
                                    ze_device_handle_t ZeDev,
                                    const ze_module_desc_t &ModuleDesc) {

  ze_module_build_log_handle_t Log;
  ze_module_handle_t Object;
  auto BuildStatus = zeModuleCreate(ZeCtx, ZeDev, &ModuleDesc, &Object, &Log);

  if (BuildStatus != ZE_RESULT_SUCCESS)
    dumpBuildLog(std::move(Log));

  CHIPERR_CHECK_LOG_AND_THROW(BuildStatus, ZE_RESULT_SUCCESS, hipErrorTbd);
  logTrace("LZ CREATE MODULE via calling zeModuleCreate {} ",
           resultToString(BuildStatus));

  return Object;
}

static void appendDeviceLibrarySources(
    std::vector<size_t> &SrcSizes, std::vector<const uint8_t *> &Sources,
    std::vector<const char *> &BuildFlags,
    const ze_float_atomic_ext_properties_t &FpAtomicProps) {

  auto AppendSource = [&](auto &Source) -> void {
    SrcSizes.push_back(Source.size());
    Sources.push_back(Source.data());
    BuildFlags.push_back("");
  };

  if (FpAtomicProps.fp32Flags & ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_ADD &&
      FpAtomicProps.fp32Flags & ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_ADD)
    AppendSource(chipstar::atomicAddFloat_native);
  else
    AppendSource(chipstar::atomicAddFloat_emulation);

  if (FpAtomicProps.fp64Flags & ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_ADD &&
      FpAtomicProps.fp64Flags & ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_ADD)
    AppendSource(chipstar::atomicAddDouble_native);
  else
    AppendSource(chipstar::atomicAddDouble_emulation);

  // OpGroupNonUniformBallot instructions seems to compile and work
  // despite not having ZE_extension_subgroups.
  AppendSource(chipstar::ballot_native);

  assert(SrcSizes.size() == Sources.size() &&
         Sources.size() == BuildFlags.size());
}

void CHIPModuleLevel0::compile(chipstar::Device *ChipDev) {
  logTrace("CHIPModuleLevel0.compile()");
  consumeSPIRV();

  auto *LzBackend = static_cast<CHIPBackendLevel0 *>(Backend);
  if (!LzBackend->hasExperimentalModuleProgramExt())
    CHIPERR_LOG_AND_THROW("Can't compile module. Level zero does not support "
                          "multi-input compilation.",
                          hipErrorTbd);

  auto *LzDev = static_cast<CHIPDeviceLevel0 *>(ChipDev);
  std::vector<size_t> ILSizes(1, IlSize_);
  std::vector<const uint8_t *> ILInputs(1, FuncIL_);
  std::vector<const char *> BuildFlags(1, ChipEnvVars.getJitFlags().c_str());

  appendDeviceLibrarySources(ILSizes, ILInputs, BuildFlags,
                             LzDev->getFpAtomicProps());

  ze_module_program_exp_desc_t ProgramDesc = {
      ZE_STRUCTURE_TYPE_MODULE_PROGRAM_EXP_DESC,
      nullptr,
      static_cast<uint32_t>(ILSizes.size()),
      ILSizes.data(),
      ILInputs.data(),
      BuildFlags.data(),
      nullptr};

  ze_module_desc_t ModuleDesc = {ZE_STRUCTURE_TYPE_MODULE_DESC, &ProgramDesc,
                                 ZE_MODULE_FORMAT_IL_SPIRV,
                                 // The driver ignores the following
                                 // members when module program
                                 // description is provided.
                                 0, nullptr, nullptr, nullptr};

  auto *ChipCtxLz = static_cast<CHIPContextLevel0 *>(ChipDev->getContext());
  ZeModule_ = compileIL(ChipCtxLz->get(), LzDev->get(), ModuleDesc);

  uint32_t KernelCount = 0;
  auto Status = zeModuleGetKernelNames(ZeModule_, &KernelCount, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  logTrace("Found {} kernels in this module.", KernelCount);

  const char *KernelNames[KernelCount];
  Status = zeModuleGetKernelNames(ZeModule_, &KernelCount, KernelNames);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  for (auto &Kernel : KernelNames)
    logTrace("Kernel {}", Kernel);
  for (uint32_t i = 0; i < KernelCount; i++) {
    std::string HostFName = KernelNames[i];
    logTrace("Registering kernel {}", HostFName);

    auto *FuncInfo = findFunctionInfo(HostFName);
    if (!FuncInfo) {
      // TODO: __syncthreads() gets turned into
      // Intel_Symbol_Table_Void_Program This is a call to OCML so it
      // shouldn't be turned into a Kernel
      continue;
      // CHIPERR_LOG_AND_THROW("Failed to find kernel in
      // OpenCLFunctionInfoMap",
      //                      hipErrorInitializationError);
    }

    // Create kernel
    ze_kernel_handle_t ZeKernel;
    ze_kernel_desc_t KernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr,
                                   0, // flags
                                   HostFName.c_str()};

    if (!LzDev->hasOnDemandPaging())
      // TODO: This is not needed if the kernel does not access allocations
      //       indirectly. This requires kernel code inspection.
      KernelDesc.flags |= ZE_KERNEL_FLAG_FORCE_RESIDENCY;

    Status = zeKernelCreate(ZeModule_, &KernelDesc, &ZeKernel);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
    logTrace("LZ KERNEL CREATION via calling zeKernelCreate {} ", Status);
    CHIPKernelLevel0 *ChipZeKernel =
        new CHIPKernelLevel0(ZeKernel, LzDev, HostFName, FuncInfo, this);
    addKernel(ChipZeKernel);
  }
}

void CHIPExecItemLevel0::setupAllArgs() {
  LOCK(this->ExecItemMtx); // required by zeKernelSetArgumentValue
  if (!ArgsSetup) {
    ArgsSetup = true;
  } else {
    return;
  }
  CHIPKernelLevel0 *Kernel = (CHIPKernelLevel0 *)ChipKernel_;

  SPVFuncInfo *FuncInfo = ChipKernel_->getFuncInfo();

  if (FuncInfo->hasByRefArgs()) {
    ArgSpillBuffer_ =
        std::make_shared<chipstar::ArgSpillBuffer>(ChipQueue_->getContext());
    ArgSpillBuffer_->computeAndReserveSpace(*FuncInfo);
  }

  auto ArgVisitor = [&](const SPVFuncInfo::KernelArg &Arg) -> void {
    ze_result_t Status;
    switch (Arg.Kind) {
    default:
      CHIPERR_LOG_AND_THROW("Internal chipStar error: Unknown argument kind.",
                            hipErrorTbd);

    case SPVTypeKind::Image: {
      auto *TexObj =
          *reinterpret_cast<const CHIPTextureLevel0 *const *>(Arg.Data);
      ze_image_handle_t ImageHandle = TexObj->getImage();
      logTrace("setImageArg {} size {}\n", Arg.Index,
               sizeof(ze_image_handle_t));
      Status = zeKernelSetArgumentValue(
          Kernel->get(), Arg.Index, sizeof(ze_image_handle_t), &ImageHandle);
      break;
    }
    case SPVTypeKind::Sampler: {
      auto *TexObj =
          *reinterpret_cast<const CHIPTextureLevel0 *const *>(Arg.Data);
      ze_sampler_handle_t SamplerHandle = TexObj->getSampler();
      logTrace("setSamplerArg {} size {}\n", Arg.Index,
               sizeof(ze_sampler_handle_t));
      Status =
          zeKernelSetArgumentValue(Kernel->get(), Arg.Index,
                                   sizeof(ze_sampler_handle_t), &SamplerHandle);
      break;
    }
    case SPVTypeKind::POD:
    case SPVTypeKind::Pointer: {
      const auto *ArgData = Arg.Data;
      auto ArgSize = Arg.Size;

      if (Arg.Kind == SPVTypeKind::Pointer) {
        if (Arg.isWorkgroupPtr()) {
          // Undocumented way to allocate Workgroup memory (which is
          // similar to OpenCL's way to allocate __local memory).
          ArgData = nullptr;
          ArgSize = SharedMem_;
        } else if (*(const void **)Arg.Data == nullptr) {
          // zeKernelSetArgumentValue does not accept nullptrs as
          // pointer argument values.  Work-around this by allocating a small
          // piece of Workgroup memory (via nullptr magic).
          ArgData = nullptr;
          ArgSize = 0;
        }
      }

      logTrace("setArg {} size {} addr {}\n", Arg.Index, ArgSize, ArgData);
      Status =
          zeKernelSetArgumentValue(Kernel->get(), Arg.Index, ArgSize, ArgData);

      if (Status != ZE_RESULT_SUCCESS) {
        logWarn("zeKernelSetArgumentValue returned error, "
                "setting the ptr arg to nullptr");
        Status = zeKernelSetArgumentValue(Kernel->get(), Arg.Index, 0, nullptr);
      }
      break;
    }
    case SPVTypeKind::PODByRef: {
      auto *SpillSlot = ArgSpillBuffer_->allocate(Arg);
      assert(SpillSlot);
      Status = zeKernelSetArgumentValue(Kernel->get(), Arg.Index,
                                        sizeof(void *), &SpillSlot);
      break;
    }
    }
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  };
  FuncInfo->visitKernelArgs(getArgs(), ArgVisitor);

  if (FuncInfo->hasByRefArgs())
    ChipQueue_->memCopyAsync(ArgSpillBuffer_->getDeviceBuffer(),
                             ArgSpillBuffer_->getHostBuffer(),
                             ArgSpillBuffer_->getSize());

  return;
}

void CHIPExecItemLevel0::setKernel(chipstar::Kernel *Kernel) {
  ChipKernel_ = static_cast<CHIPKernelLevel0 *>(Kernel);
}

chipstar::Kernel *CHIPExecItemLevel0::getKernel() { return ChipKernel_; }
