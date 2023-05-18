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

#include "CHIPBackendLevel0.hh"
#include "Utils.hh"

/**
 *  CHIPQueueLevel0::getCmdList() will return an immediate command list handle
 * if L0_IMM_QUEUES is used. There is only one such handle for a queue and a
 * queue can be shared between multiple threads thus this lock is necessary.
 *
 * If immediate command lists are not used, getCmdList will create a new
 * handle which is a thread safe operation
 */
#ifdef L0_IMM_QUEUES
#define GET_COMMAND_LIST(Queue)                                                \
  ze_command_list_handle_t CommandList;                                        \
  LOCK(Queue->QueueMtx); /* CHIPQueueLevel0::ZeCmdList_ */                     \
  CommandList = Queue->getCmdList();
#else
#define GET_COMMAND_LIST(Queue)                                                \
  ze_command_list_handle_t CommandList;                                        \
  CommandList = Queue->getCmdList();
#endif

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
  auto Status = zeEventHostReset(Event_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  LOCK(EventMtx); // CHIPEvent::TrackCalled_
  TrackCalled_ = false;
  EventStatus_ = EVENT_STATUS_INIT;
  *Refc_ = 1;
#ifndef NDEBUG
  markDeleted(false);
#endif
}

ze_event_handle_t CHIPEventLevel0::peek() {
  assert(!Deleted_ && "Event use after delete!");
  return Event_;
}

ze_event_handle_t CHIPEventLevel0::get(std::string Msg) {
  assert(!Deleted_ && "Event use after delete!");
  if (Msg.size() > 0) {
    increaseRefCount(Msg);
  } else {
    increaseRefCount("get()");
  }
  return Event_;
}

CHIPEventLevel0::~CHIPEventLevel0() {
  logTrace("chipEventLevel0 DEST {}", (void *)this);
  if (Event_) {
    auto Status = zeEventDestroy(Event_);
    // '~CHIPEventLevel0' has a non-throwing exception specification
    assert(Status == ZE_RESULT_SUCCESS);
  }

  if (isUserEvent()) {
    assert(EventPool && "EventPoolHandle_ is set but EventPool is nullptr");
    auto Status = zeEventPoolDestroy(EventPoolHandle_);
    assert(Status == ZE_RESULT_SUCCESS);
  }

  Event_ = nullptr;
  EventPoolHandle_ = nullptr;
  EventPool = nullptr;
  Timestamp_ = 0;
}

CHIPEventLevel0::CHIPEventLevel0(CHIPContextLevel0 *ChipCtx,
                                 LZEventPool *TheEventPool,
                                 unsigned int ThePoolIndex,
                                 CHIPEventFlags Flags)
    : CHIPEvent((CHIPContext *)(ChipCtx), Flags), Event_(nullptr),
      EventPoolHandle_(nullptr), Timestamp_(0) {
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
                                 CHIPEventFlags Flags)
    : CHIPEvent((CHIPContext *)(ChipCtx), Flags), Event_(nullptr),
      EventPoolHandle_(nullptr), Timestamp_(0), EventPoolIndex(0),
      EventPool(0) {
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
  // Done. Event pool handle is local to this event + this is constructor
  Status = zeEventCreate(EventPoolHandle_, &EventDesc, &Event_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd,
                              "Level Zero Event_ creation fail! ");
}

CHIPEventLevel0::CHIPEventLevel0(CHIPContextLevel0 *ChipCtx,
                                 ze_event_handle_t NativeEvent)
    : CHIPEvent((CHIPContext *)(ChipCtx)), Event_(NativeEvent),
      EventPoolHandle_(nullptr), Timestamp_(0), EventPoolIndex(0),
      EventPool(nullptr) {}

// Must use this for now - Level Zero hangs when events are host visible +
// kernel timings are enabled
void CHIPEventLevel0::recordStream(CHIPQueue *ChipQueue) {
  ze_result_t Status;

  {
    LOCK(EventMtx); // CHIPEvent::EventStatus_
    if (EventStatus_ == EVENT_STATUS_RECORDED) {
      logTrace("Event {}: EVENT_STATUS_RECORDED ... Resetting event.",
               (void *)this);
      ze_result_t Status = zeEventHostReset(Event_);
      EventStatus_ = EVENT_STATUS_INIT;
      HostTimestamp_ = 0;
      DeviceTimestamp_ = 0;
      CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
    }
  }

  auto Dev = (CHIPDeviceLevel0 *)ChipQueue->getDevice();
  Status = zeDeviceGetGlobalTimestamps(Dev->get(), &HostTimestamp_,
                                       &DeviceTimestamp_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  if (ChipQueue == nullptr)
    CHIPERR_LOG_AND_THROW("Queue passed in is null", hipErrorTbd);

  CHIPQueueLevel0 *Q = (CHIPQueueLevel0 *)ChipQueue;
  GET_COMMAND_LIST(Q)
  // The application must not call this function from
  // simultaneous threads with the same command list handle.
  Status = zeCommandListAppendBarrier(CommandList, nullptr, 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  // The application must not call this function from
  // simultaneous threads with the same command list handle.
  Status = zeCommandListAppendWriteGlobalTimestamp(
      CommandList, (uint64_t *)(Q->getSharedBufffer()), nullptr, 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  // The application must not call this function from
  // simultaneous threads with the same command list handle.
  Status = zeCommandListAppendBarrier(CommandList, nullptr, 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  // The application must not call this function from
  // simultaneous threads with the same command list handle.
  Status = zeCommandListAppendMemoryCopy(CommandList, &Timestamp_,
                                         Q->getSharedBufffer(),
                                         sizeof(uint64_t), Event_, 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  auto DestoyCommandListEvent =
      ((CHIPBackendLevel0 *)Backend)->createCHIPEvent(this->ChipContext_);
  DestoyCommandListEvent->Msg = "recordStreamComplete";
  // The application must not call this function from
  // simultaneous threads with the same command list handle.
  // Done via GET_COMMAND_LIST
  Status = zeCommandListAppendBarrier(
      CommandList, DestoyCommandListEvent->peek(), 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  Q->executeCommandList(CommandList);
  DestoyCommandListEvent->track();

  LOCK(EventMtx); // CHIPEvent::EventStatus_
  EventStatus_ = EVENT_STATUS_RECORDING;
  Msg = "recordStream";
}

bool CHIPEventLevel0::wait() {
  assert(!Deleted_ && "Event use after delete!");
  logTrace("CHIPEventLevel0::wait() {} msg={}", (void *)this, Msg);

  ze_result_t Status = zeEventHostSynchronize(Event_, UINT64_MAX);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  LOCK(EventMtx); // CHIPEvent::EventStatus_
  EventStatus_ = EVENT_STATUS_RECORDED;
  return true;
}

bool CHIPEventLevel0::updateFinishStatus(bool ThrowErrorIfNotReady) {
  assert(!Deleted_ && "Event use after delete!");
  std::string EventStatusOld, EventStatusNew;
  {
    LOCK(EventMtx); // CHIPEvent::EventStatus_

    EventStatusOld = getEventStatusStr();

    ze_result_t Status = zeEventQueryStatus(Event_);
    if (Status == ZE_RESULT_NOT_READY && ThrowErrorIfNotReady) {
      CHIPERR_LOG_AND_THROW("Event Not Ready", hipErrorNotReady);
    }
    if (Status == ZE_RESULT_SUCCESS)
      EventStatus_ = EVENT_STATUS_RECORDED;

    EventStatusNew = getEventStatusStr();
  }
  // logTrace("CHIPEventLevel0::updateFinishStatus() {} Refc: {} {}: {} -> {}",
  //          (void *)this, getCHIPRefc(), Msg, EventStatusOld, EventStatusNew);
  if (EventStatusNew != EventStatusOld) {
    return true;
  }

  return false;
}

/** This Doesn't work right now due to Level Zero Backend hanging?
  unsinged long CHIPEventLevel0::getFinishTime() {

    ze_kernel_timestamp_result_t res{};
    auto Status = zeEventQueryKernelTimestamp(Event_, &res);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

    CHIPContextLevel0* chip_ctx_lz = (CHIPContextLevel0*)chip_context;
    CHIPDeviceLevel0* chip_dev_lz =
        (CHIPDeviceLevel0*)chip_ctx_lz->getDevices()[0];

    auto props = chip_dev_lz->getDeviceProps();

    uint64_t timerResolution = props->timerResolution;
    uint32_t timestampValidBits = props->timestampValidBits;

    return res.context.kernelEnd * timerResolution;
  }
  */

uint32_t CHIPEventLevel0::getValidTimestampBits() {
  CHIPContextLevel0 *ChipCtxLz = (CHIPContextLevel0 *)ChipContext_;
  CHIPDeviceLevel0 *ChipDevLz = (CHIPDeviceLevel0 *)ChipCtxLz->getDevice();
  auto Props = ChipDevLz->getDeviceProps();
  return Props->timestampValidBits;
}

unsigned long CHIPEventLevel0::getFinishTime() {
  CHIPContextLevel0 *ChipCtxLz = (CHIPContextLevel0 *)ChipContext_;
  CHIPDeviceLevel0 *ChipDevLz = (CHIPDeviceLevel0 *)ChipCtxLz->getDevice();
  auto Props = ChipDevLz->getDeviceProps();

  uint64_t TimerResolution = Props->timerResolution;
  uint32_t TimestampValidBits = Props->timestampValidBits;

  uint32_t T = (Timestamp_ & (((uint64_t)1 << TimestampValidBits) - 1));
  T = T * TimerResolution;

  return T;
}

float CHIPEventLevel0::getElapsedTime(CHIPEvent *OtherIn) {
  /**
   * Modified HIPLZ Implementation
   * https://github.com/intel/pti-gpu/blob/master/chapters/device_activity_tracing/LevelZero.md
   */
  logTrace("CHIPEventLevel0::getElapsedTime()");
  CHIPEventLevel0 *Other = (CHIPEventLevel0 *)OtherIn;

  this->updateFinishStatus();
  Other->updateFinishStatus();
  if (!this->isFinished() || !Other->isFinished())
    CHIPERR_LOG_AND_THROW("One of the events for getElapsedTime() was done yet",
                          hipErrorNotReady);

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
  assert(!Deleted_ && "Event use after delete!");
  logTrace("CHIPEventLevel0::hostSignal()");
  auto Status = zeEventHostSignal(Event_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  LOCK(EventMtx); // CHIPEvent::EventStatus_
  EventStatus_ = EVENT_STATUS_RECORDED;
}

// End CHIPEventLevel0

// CHIPCallbackDataLevel0
// ***********************************************************************

CHIPCallbackDataLevel0::CHIPCallbackDataLevel0(hipStreamCallback_t CallbackF,
                                               void *CallbackArgs,
                                               CHIPQueue *ChipQueue)
    : CHIPCallbackData(CallbackF, CallbackArgs, ChipQueue) {
  LOCK(Backend->BackendMtx) // ensure callback enqueues are submitted as one

  CHIPContext *Ctx = ChipQueue->getContext();

  CpuCallbackComplete = (CHIPEventLevel0 *)Backend->createCHIPEvent(Ctx);
  CpuCallbackComplete->Msg = "CpuCallbackComplete";

  GpuReady = ChipQueue->enqueueBarrierImpl(nullptr);
  GpuReady->Msg = "GpuReady";

  std::vector<CHIPEvent *> ChipEvs = {CpuCallbackComplete};
  auto WaitForCpuComplete = ChipQueue->enqueueBarrierImpl(&ChipEvs);
  ChipQueue->updateLastEvent(WaitForCpuComplete);

  GpuAck = ChipQueue->enqueueMarkerImpl();
  GpuAck->Msg = "GpuAck";
}

// End CHIPCallbackDataLevel0

// CHIPEventMonitorLevel0
// ***********************************************************************

void CHIPCallbackEventMonitorLevel0::monitor() {
  CHIPCallbackDataLevel0 *CallbackData;
  while (true) {
    usleep(20000);
    LOCK(EventMonitorMtx); // CHIPEventMonitor::Stop
    {

      if (Stop) {
        logTrace(
            "CHIPCallbackEventMonitorLevel0 out of callbacks. Exiting thread");
        if (Backend->CallbackQueue.size())
          logError("Callback thread exiting while there are still active "
                   "callbacks in the queue");
        pthread_exit(0);
      }

      LOCK(Backend->CallbackQueueMtx); // CHIPBackend::CallbackQueue

      if ((Backend->CallbackQueue.size() == 0))
        continue;

      // get the callback item
      CallbackData = (CHIPCallbackDataLevel0 *)Backend->CallbackQueue.front();

      // Lock the item and members
      assert(CallbackData);
      LOCK( // CHIPBackend::CallbackQueue
          CallbackData->CallbackDataMtx);
      Backend->CallbackQueue.pop();

      // Update Status
      logTrace("CHIPCallbackEventMonitorLevel0::monitor() checking event "
               "status for {}",
               (void *)CallbackData->GpuReady);
      CallbackData->GpuReady->updateFinishStatus(false);
      if (CallbackData->GpuReady->getEventStatus() != EVENT_STATUS_RECORDED) {
        // if not ready, push to the back
        Backend->CallbackQueue.push(CallbackData);
        continue;
      }
    }

    CallbackData->execute(hipSuccess);
    CallbackData->CpuCallbackComplete->hostSignal();
    CallbackData->GpuAck->wait();

    delete CallbackData;
    pthread_yield();
  }
}

void CHIPStaleEventMonitorLevel0::monitor() {
  // Stop is false and I have more events
  while (true) {
    usleep(20000);
    LOCK(EventMonitorMtx); // CHIPEventMonitor::Stop
    std::vector<CHIPEvent *> EventsToDelete;
    std::vector<ze_command_list_handle_t> CommandListsToDelete;

    LOCK(Backend->EventsMtx); // CHIPBackend::Events
    LOCK(                     // CHIPBackendLevel0::EventCommandListMap
        ((CHIPBackendLevel0 *)Backend)->CommandListsMtx);
    // auto LzBackend = (CHIPBackendLevel0 *)Backend;
    // logTrace("CHIPStaleEventMonitorLevel0::monitor() # events {} # queues
    // {}",
    //          Backend->Events.size(), LzBackend->EventCommandListMap.size());

    auto EventCommandListMap =
        &((CHIPBackendLevel0 *)Backend)->EventCommandListMap;

    for (size_t i = 0; i < Backend->Events.size(); i++) {
      CHIPEvent *ChipEvent = Backend->Events[i];

      assert(ChipEvent);
      auto E = (CHIPEventLevel0 *)ChipEvent;

      // do not change refcount for user events
      if (E->updateFinishStatus(false) && E->EventPool) {
        E->decreaseRefCount("Event became ready");
        E->releaseDependencies();
      }

      // delete the event if refcount reached 0
      if (E->getCHIPRefc() == 0) {
        // Purpose of the stale event monitor is to release events
        // when it's safe to do so which is indicated by their ready
        // status.
        assert(E->isFinished() &&
               "Event refcount reached zero while it's not ready!");
        auto Found =
            std::find(Backend->Events.begin(), Backend->Events.end(), E);
        if (Found == Backend->Events.end())
          CHIPERR_LOG_AND_THROW("StaleEventMonitor is trying to destroy an "
                                "event which is already "
                                "removed from backend event list",
                                hipErrorTbd);
        Backend->Events.erase(Found); // TODO fix-251 segfault here

        E->doActions();

        // Check if this event is associated with a CommandList
        bool CommandListFound = EventCommandListMap->count(E);
        if (CommandListFound) {
          logTrace("Erase cmdlist assoc w/ event: {}", (void *)E);
          auto CommandList = (*EventCommandListMap)[E];
          EventCommandListMap->erase(E);

#ifdef DUBIOUS_LOCKS
          LOCK(Backend->DubiousLockLevel0)
#endif
          // The application must not call this function
          // from simultaneous threads with the same command list handle.
          // Done via this is the only thread that calls it
          auto Status = zeCommandListDestroy(CommandList);
          CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
        }

        if (E->EventPool)
          E->EventPool->returnSlot(E->EventPoolIndex);
#ifndef NDEBUG
        E->markDeleted();
#endif
      }

    } // done collecting events to delete

    /**
     * In the case that a user doesn't destroy all the
     * created streams, we remove the streams and outstanding events in
     * CHIPBackend::waitForThreadExit() but CHIPBackend has no knowledge of
     * EventCommandListMap
     */
    // TODO libCEED - re-enable this check
    if (Stop && !EventCommandListMap->size()) {
      if (Backend->Events.size() > 0) {
        logError(
            "CHIPStaleEventMonitorLevel0 stop was called but not all events "
            "have been cleared");
      } else {
        logTrace(
            "CHIPStaleEventMonitorLevel0 stop was called and all events have "
            "been cleared");
      }
      pthread_exit(0);
    }

  } // endless loop
}
// End CHIPEventMonitorLevel0

// CHIPKernelLevelZero
// ***********************************************************************

ze_kernel_handle_t CHIPKernelLevel0::get() { return ZeKernel_; }

CHIPKernelLevel0::CHIPKernelLevel0(ze_kernel_handle_t ZeKernel,
                                   CHIPDeviceLevel0 *Dev, std::string HostFName,
                                   SPVFuncInfo *FuncInfo,
                                   CHIPModuleLevel0 *Parent)
    : CHIPKernel(HostFName, FuncInfo), ZeKernel_(ZeKernel), Module(Parent),
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

CHIPQueueLevel0::~CHIPQueueLevel0() {
  logTrace("~CHIPQueueLevel0() {}", (void *)this);
  // From destructor post query only when queue is owned by CHIP
  // Non-owned command queues can be destroyed independently by the owner
  if (zeCmdQOwnership_) {
    finish(); // must finish the queue because it's possible that that there are
              // outstanding operations which have an associated CHIPEvent. If
              // we do not finish we risk the chance of StaleEventMonitor of
              // deadlocking while waiting for queue completion and subsequent
              // event status change
  }
  updateLastEvent(
      nullptr); // Just in case that unique_ptr destructor calls this, the
                // generic ~CHIPQueue() (which calls updateLastEvent(nullptr))
                // hasn't been called yet, and the stale event monitor ends up
                // waiting forever.

  // The application must not call this function from
  // simultaneous threads with the same command queue handle.
  // Done. Destructor should not be called by multiple threads
#ifdef DUBIOUS_LOCKS
  LOCK(Backend->DubiousLockLevel0)
#endif
  if (zeCmdQOwnership_) {
    zeCommandQueueDestroy(ZeCmdQ_);
  } else {
    logTrace("CHIP does not own cmd queue");
  }
}

void CHIPQueueLevel0::addCallback(hipStreamCallback_t Callback,
                                  void *UserData) {
  CHIPCallbackData *Callbackdata =
      Backend->createCallbackData(Callback, UserData, this);

  {
    LOCK(Backend->CallbackQueueMtx); // CHIPBackend::CallbackQueue
    Backend->CallbackQueue.push(Callbackdata);
  }

  return;
}

CHIPEventLevel0 *CHIPQueueLevel0::getLastEvent() {
  LOCK(LastEventMtx); // CHIPQueue::LastEvent_
  return (CHIPEventLevel0 *)LastEvent_;
}

ze_command_list_handle_t CHIPQueueLevel0::getCmdList() {
#ifdef L0_IMM_QUEUES
  return ZeCmdList_;
#else
  ze_command_list_handle_t ZeCmdList;
#ifdef DUBIOUS_LOCKS
  LOCK(Backend->DubiousLockLevel0)
#endif
  auto Status =
      zeCommandListCreate(ZeCtx_, ZeDev_, &CommandListDesc_, &ZeCmdList);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);
  return ZeCmdList;
#endif
}

CHIPQueueLevel0::CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev)
    : CHIPQueueLevel0(ChipDev, 0, L0_DEFAULT_QUEUE_PRIORITY,
                      LevelZeroQueueType::Compute) {}

CHIPQueueLevel0::CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev,
                                 CHIPQueueFlags Flags)
    : CHIPQueueLevel0(ChipDev, Flags, L0_DEFAULT_QUEUE_PRIORITY,
                      LevelZeroQueueType::Compute) {}

CHIPQueueLevel0::CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev,
                                 CHIPQueueFlags Flags, int Priority)
    : CHIPQueueLevel0(ChipDev, Flags, Priority, LevelZeroQueueType::Compute) {}

CHIPQueueLevel0::CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev,
                                 CHIPQueueFlags Flags, int Priority,
                                 LevelZeroQueueType TheType)
    : CHIPQueue(ChipDev, Flags, Priority) {
  logTrace("CHIPQueueLevel0() {}", (void *)this);
  ze_result_t Status;
  auto ChipDevLz = ChipDev;
  auto Ctx = ChipDevLz->getContext();
  auto ChipContextLz = (CHIPContextLevel0 *)Ctx;

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
      ChipContextLz->allocateImpl(32, 8, hipMemoryType::hipMemoryTypeUnified);

  // Initialize the uint64_t part as 0
  *(uint64_t *)this->SharedBuf_ = 0;

  ZeCtx_ = ChipContextLz->get();
  ZeDev_ = ChipDevLz->get();

  logTrace("CHIPQueueLevel0 constructor called via Flags and Priority");
#ifdef DUBIOUS_LOCKS
  LOCK(Backend->DubiousLockLevel0)
#endif
  Status = zeCommandQueueCreate(ZeCtx_, ZeDev_, &QueueDescriptor_, &ZeCmdQ_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);

#ifdef L0_IMM_QUEUEs
  initializeCmdListImm();
#endif
}

CHIPQueueLevel0::CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev,
                                 ze_command_queue_handle_t ZeCmdQ)
    : CHIPQueue(ChipDev, 0, L0_DEFAULT_QUEUE_PRIORITY) {
  auto ChipDevLz = ChipDev;
  auto Ctx = ChipDevLz->getContext();
  auto ChipContextLz = (CHIPContextLevel0 *)Ctx;

  QueueProperties_ = ChipDev->getComputeQueueProps();
  QueueDescriptor_ = ChipDev->getNextComputeQueueDesc();
  CommandListDesc_ = ChipDev->getCommandListComputeDesc();

  ZeCtx_ = ChipContextLz->get();
  ZeDev_ = ChipDevLz->get();

  ZeCmdQ_ = ZeCmdQ;

#ifdef L0_IMM_QUEUES
  initializeCmdListImm();
#endif
}

void CHIPQueueLevel0::initializeCmdListImm() {
  assert(QueueType != Unknown);

  auto Status = zeCommandListCreateImmediate(ZeCtx_, ZeDev_, &QueueDescriptor_,
                                             &ZeCmdList_);
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

CHIPEvent *CHIPQueueLevel0::launchImpl(CHIPExecItem *ExecItem) {
  CHIPContextLevel0 *ChipCtxZe = (CHIPContextLevel0 *)ChipContext_;
  CHIPEventLevel0 *LaunchEvent =
      (CHIPEventLevel0 *)Backend->createCHIPEvent(ChipCtxZe);
  LaunchEvent->Msg = "launch";

  CHIPKernelLevel0 *ChipKernel = (CHIPKernelLevel0 *)ExecItem->getKernel();
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
  GET_COMMAND_LIST(this);

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
  // Done via GET_COMMAND_LIST
  auto Status = zeCommandListAppendLaunchKernel(
      CommandList, KernelZe, &LaunchArgs, LaunchEvent->peek(), 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);
  auto StatusReadyCheck = zeEventQueryStatus(LaunchEvent->peek());
  if (StatusReadyCheck != ZE_RESULT_NOT_READY) {
    logCritical("KernelLaunch event immediately ready!");
  }
  executeCommandList(CommandList);

  if (std::shared_ptr<CHIPArgSpillBuffer> SpillBuf =
          ExecItem->getArgSpillBuffer())
    // Use an event action to prolong the lifetime of the spill buffer
    // in case the exec item gets destroyed before the kernel
    // completes (may happen when called from CHIPQueue::launchKernel()).
    LaunchEvent->addAction([=]() -> void { auto Tmp = SpillBuf; });

  return LaunchEvent;
}

CHIPEvent *CHIPQueueLevel0::memFillAsyncImpl(void *Dst, size_t Size,
                                             const void *Pattern,
                                             size_t PatternSize) {
  CHIPContextLevel0 *ChipCtxZe = (CHIPContextLevel0 *)ChipContext_;
  CHIPEventLevel0 *Ev = (CHIPEventLevel0 *)Backend->createCHIPEvent(ChipCtxZe);
  Ev->Msg = "memFill";

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

  GET_COMMAND_LIST(this);
  // The application must not call this function from
  // simultaneous threads with the same command list handle.
  // Done via GET_COMMAND_LIST
  ze_result_t Status = zeCommandListAppendMemoryFill(
      CommandList, Dst, Pattern, PatternSize, Size, Ev->peek(), 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  executeCommandList(CommandList);

  return Ev;
};

CHIPEvent *CHIPQueueLevel0::memCopy2DAsyncImpl(void *Dst, size_t Dpitch,
                                               const void *Src, size_t Spitch,
                                               size_t Width, size_t Height) {
  return memCopy3DAsyncImpl(Dst, Dpitch, 0, Src, Spitch, 0, Width, Height, 0);
};

CHIPEvent *CHIPQueueLevel0::memCopy3DAsyncImpl(void *Dst, size_t Dpitch,
                                               size_t Dspitch, const void *Src,
                                               size_t Spitch, size_t Sspitch,
                                               size_t Width, size_t Height,
                                               size_t Depth) {
  CHIPContextLevel0 *ChipCtxZe = (CHIPContextLevel0 *)ChipContext_;
  CHIPEventLevel0 *Ev = (CHIPEventLevel0 *)Backend->createCHIPEvent(ChipCtxZe);
  Ev->Msg = "memCopy3DAsync";

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
  GET_COMMAND_LIST(this);
  // The application must not call this function from
  // simultaneous threads with the same command list handle.
  // Done via GET_COMMAND_LIST
  ze_result_t Status = zeCommandListAppendMemoryCopyRegion(
      CommandList, Dst, &DstRegion, Dpitch, Dspitch, Src, &SrcRegion, Spitch,
      Sspitch, Ev->peek(), 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  executeCommandList(CommandList);

  return Ev;
};

// Memory copy to texture object, i.e. image
CHIPEvent *CHIPQueueLevel0::memCopyToImage(ze_image_handle_t Image,
                                           const void *Src,
                                           const CHIPRegionDesc &SrcRegion) {
  logTrace("CHIPQueueLevel0::memCopyToImage");
  CHIPContextLevel0 *ChipCtxZe = (CHIPContextLevel0 *)ChipContext_;
  CHIPEventLevel0 *Ev = (CHIPEventLevel0 *)Backend->createCHIPEvent(ChipCtxZe);
  Ev->Msg = "memCopyToImage";

  if (!SrcRegion.isPitched()) {
    GET_COMMAND_LIST(this)
    // The application must not call this function from
    // simultaneous threads with the same command list handle.
    // Done via GET_COMMAND_LIST
    ze_result_t Status = zeCommandListAppendImageCopyFromMemory(
        CommandList, Image, Src, 0,
        Ev->get("zeCommandListAppendImageCopyFromMemory"), 0, nullptr);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
    executeCommandList(CommandList);

    return Ev;
  }

  // Copy image data row by row since level zero does not have pitched copy.
  CHIPASSERT(SrcRegion.getNumDims() == 2 &&
             "UNIMPLEMENTED: 3D pitched image copy.");
  const char *SrcRow = (const char *)Src;
  for (size_t Row = 0; Row < SrcRegion.Size[1]; Row++) {
    bool LastRow = Row == SrcRegion.Size[1] - 1;
    ze_image_region_t DstZeRegion{};
    DstZeRegion.originX = 0;
    DstZeRegion.originY = Row;
    DstZeRegion.originZ = 0;
    DstZeRegion.width = SrcRegion.Size[0];
    DstZeRegion.height = 1;
    DstZeRegion.depth = 1;

    GET_COMMAND_LIST(this)
    // The application must not call this function from
    // simultaneous threads with the same command list handle.
    // Done via GET_COMMAND_LIST
    ze_result_t Status = zeCommandListAppendImageCopyFromMemory(
        CommandList, Image, SrcRow, &DstZeRegion,
        LastRow ? Ev->get("zeCommandListAppendImageCopyFromMemory") : nullptr,
        0, nullptr);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
    executeCommandList(CommandList);
    SrcRow += SrcRegion.Pitch[0];
  }
  return Ev;
};

hipError_t CHIPQueueLevel0::getBackendHandles(uintptr_t *NativeInfo,
                                              int *NumHandles) {
  logTrace("CHIPQueueLevel0::getBackendHandles");
  if (*NumHandles < 4) {
    logError("getBackendHandles requires space for 4 handles");
    return hipErrorInvalidValue;
  }
  *NumHandles = 4;

  // Get queue handler
  NativeInfo[3] = (uintptr_t)ZeCmdQ_;

  // Get context handler
  CHIPContextLevel0 *Ctx = (CHIPContextLevel0 *)ChipContext_;
  NativeInfo[2] = (uintptr_t)Ctx->get();

  // Get device handler
  CHIPDeviceLevel0 *Dev = (CHIPDeviceLevel0 *)ChipDevice_;
  NativeInfo[1] = (uintptr_t)Dev->get();

  // Get driver handler
  NativeInfo[0] = (uintptr_t)Ctx->ZeDriver;
  return hipSuccess;
}

CHIPEvent *CHIPQueueLevel0::enqueueMarkerImpl() {

  CHIPEventLevel0 *MarkerEvent =
      (CHIPEventLevel0 *)Backend->createCHIPEvent(ChipContext_);

  MarkerEvent->Msg = "marker";
  GET_COMMAND_LIST(this)
  // The application must not call this function from
  // simultaneous threads with the same command list handle.
  // Done via GET_COMMAND_LIST
  auto Status = zeCommandListAppendSignalEvent(
      CommandList,
      MarkerEvent->get("MarkerEvent: zeCommandListAppendSignalEvent"));
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  executeCommandList(CommandList);

  return MarkerEvent;
}

CHIPEvent *
CHIPQueueLevel0::enqueueBarrierImpl(std::vector<CHIPEvent *> *EventsToWaitFor) {
  // Create an event, refc=2, add it to EventList
  CHIPEventLevel0 *EventToSignal =
      (CHIPEventLevel0 *)Backend->createCHIPEvent(ChipContext_);
  EventToSignal->Msg = "barrier";
  size_t NumEventsToWaitFor = 0;

  NumEventsToWaitFor = EventsToWaitFor ? EventsToWaitFor->size() : 0;

  ze_event_handle_t *EventHandles = nullptr;
  ze_event_handle_t SignalEventHandle = nullptr;

  SignalEventHandle = ((CHIPEventLevel0 *)(EventToSignal))->peek();

  if (NumEventsToWaitFor > 0) {
    EventHandles = new ze_event_handle_t[NumEventsToWaitFor];
    for (size_t i = 0; i < NumEventsToWaitFor; i++) {
      CHIPEventLevel0 *ChipEventLz = (CHIPEventLevel0 *)(*EventsToWaitFor)[i];
      CHIPASSERT(ChipEventLz);
      EventHandles[i] = ChipEventLz->get("enqueueBarrierImpl addDependency");
      EventToSignal->addDependency(ChipEventLz);
    }
  } // done gather Event_ handles to wait on

  // TODO Should this be memory or compute?
  GET_COMMAND_LIST(this)
  // The application must not call this function from
  // simultaneous threads with the same command list handle.
  // Done via GET_COMMAND_LIST
  auto Status = zeCommandListAppendBarrier(CommandList, SignalEventHandle,
                                           NumEventsToWaitFor, EventHandles);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  executeCommandList(CommandList);

  if (EventHandles)
    delete[] EventHandles;

  return EventToSignal;
}

CHIPEvent *CHIPQueueLevel0::memCopyAsyncImpl(void *Dst, const void *Src,
                                             size_t Size) {
  logTrace("CHIPQueueLevel0::memCopyAsync");
  CHIPContextLevel0 *ChipCtxZe = (CHIPContextLevel0 *)ChipContext_;
  CHIPEventLevel0 *MemCopyEvent =
      (CHIPEventLevel0 *)Backend->createCHIPEvent(ChipCtxZe);

  ze_result_t Status;
  CHIPASSERT(MemCopyEvent->peek());
  GET_COMMAND_LIST(this);
  // The application must not call this function from simultaneous threads with
  // the same command list handle
  // Done via GET_COMMAND_LIST
  Status = zeCommandListAppendMemoryCopy(CommandList, Dst, Src, Size,
                                         MemCopyEvent->peek(), 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);
  executeCommandList(CommandList);

  return MemCopyEvent;
}

void CHIPQueueLevel0::finish() {
  // The finish Event_ that denotes the finish of current command list items
  pthread_yield();
  // Using zeCommandQueueSynchronize() for ensuring the device printf
  // buffers get flushed.
#ifdef DUBIOUS_LOCKS
  LOCK(Backend->DubiousLockLevel0)
#endif
  zeCommandQueueSynchronize(ZeCmdQ_, UINT64_MAX);

  return;
}

void CHIPQueueLevel0::executeCommandList(ze_command_list_handle_t CommandList) {
#ifdef L0_IMM_QUEUES
#else

  auto LastCmdListEvent =
      ((CHIPBackendLevel0 *)Backend)->createCHIPEvent(ChipContext_);
  LastCmdListEvent->Msg = "CmdListFinishTracker";

  ze_result_t Status;

  {
    LOCK( // CHIPBackendLevel0::EventCommandListMap
        ((CHIPBackendLevel0 *)Backend)->CommandListsMtx);

    // Associate this event with the command list. Once the events are signaled,
    // CHIPEventMonitorLevel0 will destroy the command list

    logTrace("assoc event {} w/ cmdlist", (void *)LastCmdListEvent);
    ((CHIPBackendLevel0 *)Backend)
        ->EventCommandListMap[(CHIPEventLevel0 *)LastCmdListEvent] =
        CommandList;
    // The application must not call this function from
    // simultaneous threads with the same command list handle.
    // Done via GET_COMMAND_LIST
    Status = zeCommandListAppendBarrier(CommandList, LastCmdListEvent->peek(),
                                        0, nullptr);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
    // The application must not call this function from
    // simultaneous threads with the same command list handle.
    // Done via GET_COMMAND_LIST
    Status = zeCommandListClose(CommandList);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
#ifdef DUBIOUS_LOCKS
    LOCK(Backend->DubiousLockLevel0)
#endif
    Status =
        zeCommandQueueExecuteCommandLists(ZeCmdQ_, 1, &CommandList, nullptr);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  }

  LastCmdListEvent->track();
#endif
};

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
    CHIPEventFlags Flags;
    Events_.push_back(new CHIPEventLevel0(Ctx_, this, i, Flags));
    FreeSlots_.push(i);
  }
};

LZEventPool::~LZEventPool() {
  for (unsigned i = 0; i < Size_; i++) {
    delete Events_[i];
  }
  // The application must not call this function from
  // simultaneous threads with the same event pool handle.
  // Done via destructor should not be called from multiple threads
  auto Status = zeEventPoolDestroy(EventPool_);
  // '~CHIPEventLevel0' has a non-throwing exception specification
  assert(Status == ZE_RESULT_SUCCESS);
};

CHIPEventLevel0 *LZEventPool::getEvent() {
  int PoolIndex = getFreeSlot();
  if (PoolIndex == -1)
    return nullptr;
  auto Event = Events_[PoolIndex];
  Event->reset();

  return Event;
};

int LZEventPool::getFreeSlot() {
  LOCK(EventPoolMtx); // LZEventPool::FreeSlots_
  if (FreeSlots_.size() == 0)
    return -1;

  auto Slot = FreeSlots_.top();
  FreeSlots_.pop();

  return Slot;
}

void LZEventPool::returnSlot(int Slot) {
  LOCK(EventPoolMtx); // LZEventPool::FreeSlots_
  FreeSlots_.push(Slot);
  return;
}

// End EventPool

// CHIPBackendLevel0
// ***********************************************************************
CHIPExecItem *CHIPBackendLevel0::createCHIPExecItem(dim3 GirdDim, dim3 BlockDim,
                                                    size_t SharedMem,
                                                    hipStream_t ChipQueue) {
  CHIPExecItemLevel0 *ExecItem =
      new CHIPExecItemLevel0(GirdDim, BlockDim, SharedMem, ChipQueue);
  return ExecItem;
};

CHIPEventLevel0 *CHIPBackendLevel0::createCHIPEvent(CHIPContext *ChipCtx,
                                                    CHIPEventFlags Flags,
                                                    bool UserEvent) {
  CHIPEventLevel0 *Event;
  if (UserEvent) {
    Event = new CHIPEventLevel0((CHIPContextLevel0 *)ChipCtx, Flags);
    // Event->increaseRefCount("hipEventCreate");
  } else {
    auto ZeCtx = (CHIPContextLevel0 *)ChipCtx;
    Event = ZeCtx->getEventFromPool();
  }

  return Event;
}

void CHIPBackendLevel0::uninitialize() {
  /**
   * Stale Event Monitor expects to collect all events. To do this, all events
   * must reach the refcount of 0. At this point, all queues should have their
   * LastEvent as nullptr but in case a user didn't sync and destroy a
   * user-created stream, such stream might not have its LastEvent as nullptr.
   *
   * To be safe, we iterate through all the queues and update their last event.
   */
  waitForThreadExit();
  logTrace("CHIPBackend::uninitialize(): Setting the LastEvent to null for all "
           "user-created queues");

  if (CallbackEventMonitor_) {
    logTrace("CHIPBackend::uninitialize(): Killing CallbackEventMonitor");
    LOCK(CallbackEventMonitor_->EventMonitorMtx); // CHIPEventMonitor::Stop
    CallbackEventMonitor_->Stop = true;
  }
  CallbackEventMonitor_->join();

  {
    logTrace("CHIPBackend::uninitialize(): Killing StaleEventMonitor");
    LOCK(StaleEventMonitor_->EventMonitorMtx); // CHIPEventMonitor::Stop
    StaleEventMonitor_->Stop = true;
  }
  StaleEventMonitor_->join();

  if (Backend->Events.size()) {
    logTrace("Remaining {} events that haven't been collected:",
             Backend->Events.size());
    for (auto *E : Backend->Events) {
      logTrace("{} status= {} refc={}", E->Msg, E->getEventStatusStr(),
               E->getCHIPRefc());
      if (!E->isUserEvent()) {
        // A strong indicator that we are missing decreaseRefCount() call
        // for events which are solely managed by the CHIP-SPV.
        assert(!(E->isFinished() && E->getCHIPRefc() > 0) &&
               "Missed decreaseRefCount()?");
        assert(E->isFinished() && "Uncollected non-user events!");
      }
    }
    logTrace("Remaining {} command lists that haven't been collected:",
             ((CHIPBackendLevel0 *)Backend)->EventCommandListMap.size());
  }
  return;
}

std::string CHIPBackendLevel0::getDefaultJitFlags() {
  return std::string(
      "-cl-std=CL2.0 -cl-take-global-address -cl-match-sincospi");
}

void CHIPBackendLevel0::initializeImpl(std::string CHIPPlatformStr,
                                       std::string CHIPDeviceTypeStr,
                                       std::string CHIPDeviceStr) {
  logTrace("CHIPBackendLevel0 Initialize");
  MinQueuePriority_ = ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH;
  ze_result_t Status;
  Status = zeInit(0);
  if (Status != ZE_RESULT_SUCCESS) {
    logCritical("Level Zero failed to initialize any devices");
    std::exit(1);
  }

  int SelectedDeviceIdx = atoi(CHIPDeviceStr.c_str());

  bool AnyDeviceType = false;
  ze_device_type_t ZeDeviceType;
  if (!CHIPDeviceTypeStr.compare("gpu")) {
    ZeDeviceType = ZE_DEVICE_TYPE_GPU;
  } else if (!CHIPDeviceTypeStr.compare("fpga")) {
    ZeDeviceType = ZE_DEVICE_TYPE_FPGA;
  } else if (!CHIPDeviceTypeStr.compare("default")) {
    // For 'default' pick all devices of any type.
    AnyDeviceType = true;
  } else {
    CHIPERR_LOG_AND_THROW("CHIP_DEVICE_TYPE must be either gpu or fpga",
                          hipErrorInitializationError);
  }
  int PlatformIdx = std::atoi(CHIPPlatformStr.c_str());
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

  // TODO Allow for multilpe platforms(drivers)
  // TODO Check platform ID is not the same as OpenCL. You can have
  // two OCL platforms but only one level0 driver
  ze_driver_handle_t ZeDriver = ZeDrivers[PlatformIdx];

  assert(ZeDriver != nullptr);
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
  Backend->addContext(ChipL0Ctx);

  // Filter in only devices of selected type and add them to the
  // backend as derivates of CHIPDevice
  auto Dev = ZeDevices[SelectedDeviceIdx];
  ze_device_properties_t DeviceProperties{};
  DeviceProperties.pNext = nullptr;
  DeviceProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;

  Status = zeDeviceGetProperties(Dev, &DeviceProperties);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  if (AnyDeviceType || ZeDeviceType == DeviceProperties.type) {
    CHIPDeviceLevel0 *ChipL0Dev = CHIPDeviceLevel0::create(Dev, ChipL0Ctx, 0);
    ChipL0Ctx->setDevice(ChipL0Dev);
  }

  StaleEventMonitor_ =
      (CHIPStaleEventMonitorLevel0 *)Backend->createStaleEventMonitor_();
  CallbackEventMonitor_ =
      (CHIPCallbackEventMonitorLevel0 *)Backend->createCallbackEventMonitor_();
}

void CHIPBackendLevel0::initializeFromNative(const uintptr_t *NativeHandles,
                                             int NumHandles) {
  logTrace("CHIPBackendLevel0 InitializeNative");
  MinQueuePriority_ = ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH;

  ze_driver_handle_t Drv = (ze_driver_handle_t)NativeHandles[0];
  ze_device_handle_t Dev = (ze_device_handle_t)NativeHandles[1];
  ze_context_handle_t Ctx = (ze_context_handle_t)NativeHandles[2];

  CHIPContextLevel0 *ChipCtx = new CHIPContextLevel0(Drv, Ctx);
  ChipCtx->setZeContextOwnership(false);
  addContext(ChipCtx);

  CHIPDeviceLevel0 *ChipDev = CHIPDeviceLevel0::create(Dev, ChipCtx, 0);
  ChipCtx->setDevice(ChipDev);

  LOCK(Backend->BackendMtx); // CHIPBackendLevel0::StaleEventMonitor
  ChipDev->LegacyDefaultQueue = ChipDev->createQueue(NativeHandles, NumHandles);

  StaleEventMonitor_ =
      (CHIPStaleEventMonitorLevel0 *)Backend->createStaleEventMonitor_();
  CallbackEventMonitor_ =
      (CHIPCallbackEventMonitorLevel0 *)Backend->createCallbackEventMonitor_();
  setActiveDevice(ChipDev);
}

hipEvent_t CHIPBackendLevel0::getHipEvent(void *NativeEvent) {
  ze_event_handle_t E = (ze_event_handle_t)NativeEvent;
  CHIPEventLevel0 *NewEvent =
      new CHIPEventLevel0((CHIPContextLevel0 *)ActiveCtx_, E);
  NewEvent->increaseRefCount("getHipEvent");
  return NewEvent;
}

void *CHIPBackendLevel0::getNativeEvent(hipEvent_t HipEvent) {
  CHIPEventLevel0 *E = (CHIPEventLevel0 *)HipEvent;
  if (!E->isRecordingOrRecorded())
    return nullptr;
  // TODO should we retain here?
  return (void *)E->get("getNativeEvent");
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
  // The application must not call this function from
  // simultaneous threads with the same context handle.
  // Done via destructor should not be called from multiple threads
  if (ownsZeContext) {
    zeContextDestroy(this->ZeCtx);
  }
}

void *CHIPContextLevel0::allocateImpl(size_t Size, size_t Alignment,
                                      hipMemoryType MemTy,
                                      CHIPHostAllocFlags Flags) {

#ifdef MALLOC_SHARED_WORKAROUND
  if (MemTy == hipMemoryType::hipMemoryTypeUnified) {
    MemTy = hipMemoryType::hipMemoryTypeHost;
    logWarn("Using zeMemAllocHost as a workaround instead of zeMemAllocShared");
  }
#endif

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

    return Ptr;
  } else if (MemTy == hipMemoryType::hipMemoryTypeDevice) {
    auto ChipDev = (CHIPDeviceLevel0 *)Backend->getActiveDevice();
    ze_device_handle_t ZeDev = ChipDev->get();

    ze_result_t Status =
        zeMemAllocDevice(ZeCtx, &DmaDesc, Size, Alignment, ZeDev, &Ptr);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                                hipErrorMemoryAllocation);

    return Ptr;
  } else if (MemTy == hipMemoryType::hipMemoryTypeHost) {
    // TODO Check if devices support cross-device sharing?
    ze_result_t Status = zeMemAllocHost(ZeCtx, &HmaDesc, Size, Alignment, &Ptr);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                                hipErrorMemoryAllocation);

    return Ptr;
  }
  CHIPERR_LOG_AND_THROW("Failed to allocate memory", hipErrorMemoryAllocation);
}

// CHIPDeviceLevelZero
// ***********************************************************************
CHIPDeviceLevel0::CHIPDeviceLevel0(ze_device_handle_t ZeDev,
                                   CHIPContextLevel0 *ChipCtx, int Idx)
    : CHIPDevice(ChipCtx, Idx), ZeDev_(ZeDev), ZeCtx_(ChipCtx->get()),
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
  ze_device_module_properties_t DeviceModuleProps;
  DeviceModuleProps.pNext = nullptr;
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

  HipDeviceProps_.clockInstructionRate = ZeDeviceProps_.coreClockRate;
  HipDeviceProps_.concurrentKernels = 1;
  HipDeviceProps_.pciDomainID = 0;
  HipDeviceProps_.pciBusID = 0x10;
  HipDeviceProps_.pciDeviceID = 0x40 + getDeviceId();
  HipDeviceProps_.isMultiGpuBoard = 0;
  HipDeviceProps_.canMapHostMemory = 1;
  HipDeviceProps_.gcnArch = 0;
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
  // in CHIP-SPV.
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
}

CHIPQueue *CHIPDeviceLevel0::createQueue(CHIPQueueFlags Flags, int Priority) {
  CHIPQueueLevel0 *NewQ = new CHIPQueueLevel0(this, Flags, Priority);
  return NewQ;
}

CHIPQueue *CHIPDeviceLevel0::createQueue(const uintptr_t *NativeHandles,
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

CHIPTexture *CHIPDeviceLevel0::createTexture(
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

    CHIPRegionDesc SrcRegion = CHIPRegionDesc::from(*Array);
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
    auto SrcDesc = CHIPRegionDesc::get1DRegion(Width, TexelByteSize);
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
    auto SrcDesc = CHIPRegionDesc::from(*PResDesc);
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
void CHIPModuleLevel0::compile(CHIPDevice *ChipDev) {
  logTrace("CHIPModuleLevel0.compile()");
  consumeSPIRV();

  ze_result_t Status;

  // Create module with global address aware
  std::string CompilerOptions = Backend->getJitFlags();
  ze_module_desc_t ModuleDesc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                                 nullptr,
                                 ZE_MODULE_FORMAT_IL_SPIRV,
                                 IlSize_,
                                 FuncIL_,
                                 CompilerOptions.c_str(),
                                 nullptr};

  CHIPContextLevel0 *ChipCtxLz = (CHIPContextLevel0 *)(ChipDev->getContext());
  CHIPDeviceLevel0 *LzDev = (CHIPDeviceLevel0 *)ChipDev;

  ze_device_handle_t ZeDev = LzDev->get();
  ze_context_handle_t ZeCtx = ChipCtxLz->get();

  ze_module_build_log_handle_t Log;
  auto BuildStatus =
      zeModuleCreate(ZeCtx, ZeDev, &ModuleDesc, &ZeModule_, &Log);
  if (BuildStatus != ZE_RESULT_SUCCESS) {
    size_t LogSize;
    Status = zeModuleBuildLogGetString(Log, &LogSize, nullptr);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
    char LogStr[LogSize];
    Status = zeModuleBuildLogGetString(Log, &LogSize, LogStr);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
    logError("ZE Build Log: {}", std::string(LogStr).c_str());
    // The application must not call this function from
    // simultaneous threads with the same build log handle.
    // Done via this function is only invoked via call_once
    Status = zeModuleBuildLogDestroy(Log);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  }
  CHIPERR_CHECK_LOG_AND_THROW(BuildStatus, ZE_RESULT_SUCCESS, hipErrorTbd);
  logTrace("LZ CREATE MODULE via calling zeModuleCreate {} ",
           resultToString(BuildStatus));
  // if (Status == ZE_RESULT_ERROR_MODULE_BUILD_FAILURE) {
  //  CHIPERR_LOG_AND_THROW("Module failed to JIT: " + std::string(log_str),
  //                        hipErrorUnknown);
  //}

  uint32_t KernelCount = 0;
  Status = zeModuleGetKernelNames(ZeModule_, &KernelCount, nullptr);
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
      // shouldn't be turned into a CHIPKernel
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
        std::make_shared<CHIPArgSpillBuffer>(ChipQueue_->getContext());
    ArgSpillBuffer_->computeAndReserveSpace(*FuncInfo);
  }

  auto ArgVisitor = [&](const SPVFuncInfo::KernelArg &Arg) -> void {
    ze_result_t Status;
    switch (Arg.Kind) {
    default:
      CHIPERR_LOG_AND_THROW("Internal CHIP-SPV error: Unknown argument kind.",
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

void CHIPExecItemLevel0::setKernel(CHIPKernel *Kernel) {
  ChipKernel_ = static_cast<CHIPKernelLevel0 *>(Kernel);
}

CHIPKernel *CHIPExecItemLevel0::getKernel() { return ChipKernel_; }
