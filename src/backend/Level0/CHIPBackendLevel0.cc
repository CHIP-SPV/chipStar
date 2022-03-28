#include "CHIPBackendLevel0.hh"

// CHIPEventLevel0
// ***********************************************************************

ze_event_handle_t CHIPEventLevel0::peek() { return Event_; }
ze_event_handle_t CHIPEventLevel0::get() {
  increaseRefCount();
  return Event_;
}

CHIPEventLevel0::~CHIPEventLevel0() {
  if (Event_)
    zeEventDestroy(Event_);
  if (EventPool_) // Previous event could have deleted this evet
    zeEventPoolDestroy(EventPool_);
  Event_ = nullptr;
  EventPool_ = nullptr;
}

CHIPEventLevel0::CHIPEventLevel0(CHIPContextLevel0 *ChipCtx,
                                 CHIPEventFlags Flags)
    : CHIPEvent((CHIPContext *)(ChipCtx), Flags) {
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

  ze_result_t Status =
      zeEventPoolCreate(ZeCtx->get(), &EventPoolDesc, 0, nullptr, &EventPool_);
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

  Status = zeEventCreate(EventPool_, &EventDesc, &Event_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd,
                              "Level Zero Event_ creation fail! ");
}

void CHIPEventLevel0::takeOver(CHIPEvent *OtherIn) {
  // Take over target queues Event_
  CHIPEventLevel0 *Other = (CHIPEventLevel0 *)OtherIn;
  std::lock_guard<std::mutex> LockThis(this->Mtx);
  std::lock_guard<std::mutex> LockOther(Other->Mtx);
  if (*Refc_ > 1)
    decreaseRefCount();
  this->Event_ = Other->get(); // increases refcount
  this->EventPool_ = Other->EventPool_;
  this->Msg = Other->Msg;
  this->Refc_ = Other->Refc_;
}

// Must use this for now - Level Zero hangs when events are host visible +
// kernel timings are enabled
void CHIPEventLevel0::recordStream(CHIPQueue *ChipQueue) {
  ze_result_t Status;

  if (EventStatus_ == EVENT_STATUS_RECORDED) {
    logTrace("{}: EVENT_STATUS_RECORDED ... Resetting event.");
    ze_result_t Status = zeEventHostReset(Event_);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  }

  if (ChipQueue == nullptr)
    CHIPERR_LOG_AND_THROW("Queue passed in is null", hipErrorTbd);

  CHIPQueueLevel0 *Q = (CHIPQueueLevel0 *)ChipQueue;

  Status = zeCommandListAppendBarrier(Q->getCmdList(), nullptr, 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  Status = zeCommandListAppendWriteGlobalTimestamp(
      Q->getCmdList(), (uint64_t *)(Q->getSharedBufffer()), nullptr, 0,
      nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  Status = zeCommandListAppendBarrier(Q->getCmdList(), nullptr, 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  Status = zeCommandListAppendMemoryCopy(Q->getCmdList(), &Timestamp_,
                                         Q->getSharedBufffer(),
                                         sizeof(uint64_t), Event_, 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  EventStatus_ = EVENT_STATUS_RECORDING;
  if (Msg == "") {
    Msg = "recordStream";
  }
  return;
}

bool CHIPEventLevel0::wait() {
  logTrace("CHIPEventLevel0::wait() msg={}", Msg);

  ze_result_t Status = zeEventHostSynchronize(Event_, UINT64_MAX);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  EventStatus_ = EVENT_STATUS_RECORDED;
  return true;
}

bool CHIPEventLevel0::updateFinishStatus() {
  auto EventStatusOld = getEventStatusStr();

  ze_result_t Status = zeEventQueryStatus(Event_);
  if (Status == ZE_RESULT_SUCCESS)
    EventStatus_ = EVENT_STATUS_RECORDED;

  auto EventStatusNew = getEventStatusStr();
  if (EventStatusNew != EventStatusOld)
    logTrace("CHIPEventLevel0::updateFinishStatus() {}: {} -> {}", Msg,
             EventStatusOld, EventStatusNew);
  return true;
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

unsigned long CHIPEventLevel0::getFinishTime() {
  CHIPContextLevel0 *ChipCtxLz = (CHIPContextLevel0 *)ChipContext_;
  CHIPDeviceLevel0 *ChipDevLz = (CHIPDeviceLevel0 *)ChipCtxLz->getDevices()[0];
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

  if (!this->isRecordingOrRecorded() || !Other->isRecordingOrRecorded())
    return hipErrorInvalidResourceHandle;

  this->updateFinishStatus();
  Other->updateFinishStatus();
  if (!this->isFinished() || !Other->isFinished())
    return hipErrorNotReady;

  unsigned long Started = this->getFinishTime();
  unsigned long Finished = Other->getFinishTime();

  if (Started > Finished) {
    logWarn("End < Start ... Swapping events");
    std::swap(Started, Finished);
  }

  // TODO should this be context or global? Probably context
  uint64_t Elapsed = Finished - Started;

#define NANOSECS 1000000000
  uint64_t MS = (Elapsed / NANOSECS) * 1000;
  uint64_t NS = Elapsed % NANOSECS;
  float FractInMS = ((float)NS) / 1000000.0f;
  auto Ms = (float)MS + FractInMS;

  return Ms;
}

void CHIPEventLevel0::hostSignal() {
  logTrace("CHIPEventLevel0::hostSignal()");
  auto Status = zeEventHostSignal(Event_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  EventStatus_ = EVENT_STATUS_RECORDED;
}

// End CHIPEventLevel0

// CHIPCallbackDataLevel0
// ***********************************************************************

CHIPCallbackDataLevel0::CHIPCallbackDataLevel0(hipStreamCallback_t CallbackF,
                                               void *CallbackArgs,
                                               CHIPQueue *ChipQueue)
    : CHIPCallbackData(CallbackF, CallbackArgs, ChipQueue) {
  std::lock_guard<std::mutex> Lock(Mtx);
  std::lock_guard<std::mutex> LockEventList(Backend->EventsMtx);
  CHIPContext *Ctx = ChipQueue->getContext();

  CpuCallbackComplete = Backend->createCHIPEvent(Ctx);
  CpuCallbackComplete->Msg = "CpuCallbackComplete";
  CpuCallbackComplete->increaseRefCount();

  GpuReady = ChipQueue->enqueueBarrier(nullptr);
  GpuReady->Msg = "GpuReady";
  GpuReady->increaseRefCount();

  std::vector<CHIPEvent *> ChipEvs = {CpuCallbackComplete};
  ChipQueue->enqueueBarrier(&ChipEvs);

  GpuAck = ChipQueue->enqueueMarker();
  GpuAck->Msg = "GpuAck";
  GpuAck->increaseRefCount();
}

// End CHIPCallbackDataLevel0

// CHIPEventMonitorLevel0
// ***********************************************************************

void CHIPCallbackEventMonitorLevel0::monitor() {
  logTrace("CHIPEventMonitorLevel0::monitor()");
  while (true) {
    if (Backend->CallbackStack.size() == 0) {
      pthread_yield();
      continue;
    }
    std::lock_guard<std::mutex> Lock(Backend->CallbackStackMtx);

    // get the callback item
    CHIPCallbackDataLevel0 *CallbackData =
        (CHIPCallbackDataLevel0 *)Backend->CallbackStack.front();

    // Lock the item and members
    std::lock_guard<std::mutex> LockCallbackData(CallbackData->Mtx);
    std::lock_guard<std::mutex> Lock1(CallbackData->GpuReady->Mtx);
    std::lock_guard<std::mutex> Lock2(CallbackData->CpuCallbackComplete->Mtx);
    std::lock_guard<std::mutex> Lock3(CallbackData->GpuAck->Mtx);

    Backend->CallbackStack.pop();

    // Update Status
    CallbackData->GpuReady->updateFinishStatus();
    if (CallbackData->GpuReady->getEventStatus() != EVENT_STATUS_RECORDED) {
      // if not ready, push to the back
      Backend->CallbackStack.push(CallbackData);
      continue;
    }

    CallbackData->execute(hipSuccess);
    CallbackData->CpuCallbackComplete->hostSignal();
    CallbackData->GpuAck->wait();

    delete CallbackData;
    pthread_yield();
  }
}

void CHIPStaleEventMonitorLevel0::monitor() {
  logTrace("CHIPEventMonitorLevel0::monitor()");
  while (true) {
    sleep(1);
    std::vector<CHIPEvent *> EventsToDelete;
    std::lock_guard<std::mutex> AllEventsLock(Backend->EventsMtx);
    for (int i = 0; i < Backend->Events.size(); i++) {
      CHIPEvent *ChipEvent = Backend->Events[i];

      std::lock_guard<std::mutex> CurrentEventLock(Backend->Events[i]->Mtx);

      assert(ChipEvent);
      auto E = (CHIPEventLevel0 *)ChipEvent;

      if (E->Msg.compare("UserEvent") != 0)
        if (E->getCHIPRefc() == 1) { // only do this check for non UserEvents
          E->updateFinishStatus();   // only check if refcount is 1
          if (E->getEventStatus() == EVENT_STATUS_RECORDED) {
            EventsToDelete.push_back(E);
          }
        }
    } // done collecting events to delete

    for (int i = 0; i < EventsToDelete.size(); i++) {
      auto E = EventsToDelete[i];
      auto Found = std::find(Backend->Events.begin(), Backend->Events.end(), E);
      if (Found == Backend->Events.end())
        CHIPERR_LOG_AND_THROW(
            "StaleEventMonitor is trying to destroy an event which is already "
            "removed from backend event list",
            hipErrorTbd);
      Backend->Events.erase(Found);
      delete E;
    }
    pthread_yield();
  } // endless loop
}
// End CHIPEventMonitorLevel0

// CHIPKernelLevelZero
// ***********************************************************************

ze_kernel_handle_t CHIPKernelLevel0::get() { return ZeKernel_; }

CHIPKernelLevel0::CHIPKernelLevel0(ze_kernel_handle_t ZeKernel,
                                   std::string HostFName, OCLFuncInfo *FuncInfo)
    : CHIPKernel(HostFName, FuncInfo), ZeKernel_(ZeKernel) {
  logTrace("CHIPKernelLevel0 constructor via ze_kernel_handle");
}
// End CHIPKernelLevelZero

// CHIPQueueLevelZero
// ***********************************************************************

CHIPEventLevel0 *CHIPQueueLevel0::getLastEvent() {
  return (CHIPEventLevel0 *)LastEvent_;
}

CHIPQueueLevel0::CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev, unsigned int Flags)
    : CHIPQueueLevel0(ChipDev, Flags, 0) {}
CHIPQueueLevel0::CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev)
    : CHIPQueueLevel0(ChipDev, 0, 0) {}

CHIPQueueLevel0::CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev, unsigned int Flags,
                                 int Priority)
    : CHIPQueue(ChipDev, Flags, Priority) {
  ze_result_t Status;
  auto ChipDevLz = ChipDev;
  auto Ctx = ChipDevLz->getContext();
  auto ChipContextLz = (CHIPContextLevel0 *)Ctx;

  ZeCtx_ = ChipContextLz->get();
  ZeDev_ = ChipDevLz->get();

  logTrace("CHIPQueueLevel0 constructor called via CHIPContextLevel0 and "
           "CHIPDeviceLevel0");

  // Discover all command queue groups
  uint32_t CmdqueueGroupCount = 0;
  zeDeviceGetCommandQueueGroupProperties(ZeDev_, &CmdqueueGroupCount, nullptr);
  logTrace("CommandGroups found: {}", CmdqueueGroupCount);

  ze_command_queue_group_properties_t *CmdqueueGroupProperties =
      (ze_command_queue_group_properties_t *)malloc(
          CmdqueueGroupCount * sizeof(ze_command_queue_group_properties_t));
  for (int i = 0; i < CmdqueueGroupCount; i++) {
    CmdqueueGroupProperties[i] = {
        ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES, // stype
        nullptr,                                          // pNext
        0,                                                // flags
        0, // maxMemoryFillPatternSize
        0  // numQueues
    };
  }
  zeDeviceGetCommandQueueGroupProperties(ZeDev_, &CmdqueueGroupCount,
                                         CmdqueueGroupProperties);

  // Find a command queue type that support compute
  uint32_t ComputeQueueGroupOrdinal = CmdqueueGroupCount;
  for (uint32_t i = 0; i < CmdqueueGroupCount; ++i) {
    if (CmdqueueGroupProperties[i].flags &
        ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
      ComputeQueueGroupOrdinal = i;
      logTrace("Found compute command group");
      break;
    }
  }
  ze_command_queue_desc_t CommandQueueDesc = {
      ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
      nullptr,
      ComputeQueueGroupOrdinal,
      0, // index
      0, // flags
      ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
      ZE_COMMAND_QUEUE_PRIORITY_NORMAL};

  // Create a default command queue (in case need to pass it outside of
  Status = zeCommandQueueCreate(ZeCtx_, ZeDev_, &CommandQueueDesc, &ZeCmdQ_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);

  // CHIP-SPV) Create an immediate command list
  Status = zeCommandListCreateImmediate(ZeCtx_, ZeDev_, &CommandQueueDesc,
                                        &ZeCmdListImm_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);
  ChipContext_->addQueue(this);
  ChipDevice_->addQueue(this);

  // Initialize the internal Event_ pool and finish Event_
  ze_event_pool_desc_t EpDesc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr,
                                 ZE_EVENT_POOL_FLAG_HOST_VISIBLE,
                                 1 /* Count */};

  ze_event_desc_t EvDesc = {ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr,
                            0, /* Index */
                            ZE_EVENT_SCOPE_FLAG_HOST, ZE_EVENT_SCOPE_FLAG_HOST};
  uint32_t NumDevices = 1;
  Status = zeEventPoolCreate(ZeCtx_, &EpDesc, NumDevices, &ZeDev_, &EventPool_);

  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd,
                              "zeEventPoolCreate FAILED");

  Status = zeEventCreate(EventPool_, &EvDesc, &FinishEvent_);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd,
                              "zeEventCreate FAILED with return code");

  // Initialize the shared memory buffer
  // TODO This does not record the buffer allocation in device allocation
  // tracker
  SharedBuf_ = ChipContextLz->allocateImpl(32, 8, CHIPMemoryType::Shared);

  // Initialize the uint64_t part as 0
  *(uint64_t *)this->SharedBuf_ = 0;
  /**
   * queues should always have lastEvent. Can't do this in the constuctor
   * because enqueueMarker is virtual and calling overriden virtual methods from
   * constructors is undefined behavior.
   *
   * Also, must call implementation method enqueueMarker_ as opposed to wrapped
   * one (enqueueMarker) because the wrapped method enforces queue semantics
   * which require LastEvent to be initialized.
   *
   */
  std::lock_guard<std::mutex> LockEvents(Backend->EventsMtx);
  auto Ev = enqueueMarker();
  Ev->Msg = "InitialMarker";
}

CHIPEvent *CHIPQueueLevel0::launchImpl(CHIPExecItem *ExecItem) {
  CHIPContextLevel0 *ChipCtxZe = (CHIPContextLevel0 *)ChipContext_;
  CHIPEventLevel0 *Ev = (CHIPEventLevel0 *)Backend->createCHIPEvent(ChipCtxZe);
  Ev->Msg = "launch";

  CHIPKernelLevel0 *ChipKernel = (CHIPKernelLevel0 *)ExecItem->getKernel();
  ze_kernel_handle_t KernelZe = ChipKernel->get();
  logTrace("Launching Kernel {}", ChipKernel->getName());

  ze_result_t Status =
      zeKernelSetGroupSize(KernelZe, ExecItem->getBlock().x,
                           ExecItem->getBlock().y, ExecItem->getBlock().z);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  ExecItem->setupAllArgs();
  auto X = ExecItem->getGrid().x;
  auto Y = ExecItem->getGrid().y;
  auto Z = ExecItem->getGrid().z;
  ze_group_count_t LaunchArgs = {X, Y, Z};
  Status = zeCommandListAppendLaunchKernel(ZeCmdListImm_, KernelZe, &LaunchArgs,
                                           Ev->peek(), 0, nullptr);
  logTrace("Kernel submitted to the queue");
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);
  return Ev;
}

CHIPEvent *CHIPQueueLevel0::memFillAsyncImpl(void *Dst, size_t Size,
                                             const void *Pattern,
                                             size_t PatternSize) {
  CHIPContextLevel0 *ChipCtxZe = (CHIPContextLevel0 *)ChipContext_;
  CHIPEventLevel0 *Ev = (CHIPEventLevel0 *)Backend->createCHIPEvent(ChipCtxZe);
  Ev->Msg = "memFill";
  ze_result_t Status = zeCommandListAppendMemoryFill(
      ZeCmdListImm_, Dst, Pattern, PatternSize, Size, Ev->peek(), 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
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
  ze_result_t Status = zeCommandListAppendMemoryCopyRegion(
      ZeCmdListImm_, Dst, &DstRegion, Dpitch, Dspitch, Src, &SrcRegion, Spitch,
      Sspitch, Ev->peek(), 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  return Ev;
};

// Memory copy to texture object, i.e. image
CHIPEvent *CHIPQueueLevel0::memCopyToTextureImpl(CHIPTexture *TexObj,
                                                 void *Src) {
  CHIPContextLevel0 *ChipCtxZe = (CHIPContextLevel0 *)ChipContext_;
  CHIPEventLevel0 *Ev = (CHIPEventLevel0 *)Backend->createCHIPEvent(ChipCtxZe);
  Ev->Msg = "memCopyToTexture";

  ze_image_handle_t ImageHandle = (ze_image_handle_t)TexObj->Image;
  ze_result_t Status = zeCommandListAppendImageCopyFromMemory(
      ZeCmdListImm_, ImageHandle, Src, 0, Ev->peek(), 0, 0);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  return Ev;
};

void CHIPQueueLevel0::getBackendHandles(unsigned long *NativeInfo, int *Size) {
  logTrace("CHIPQueueLevel0::getBackendHandles");
  *Size = 4;

  // Get queue handler
  NativeInfo[3] = (unsigned long)ZeCmdQ_;

  // Get context handler
  CHIPContextLevel0 *Ctx = (CHIPContextLevel0 *)ChipContext_;
  NativeInfo[2] = (unsigned long)Ctx->get();

  // Get device handler
  CHIPDeviceLevel0 *Dev = (CHIPDeviceLevel0 *)ChipDevice_;
  NativeInfo[1] = (unsigned long)Dev->get();

  // Get driver handler
  NativeInfo[0] = (unsigned long)Ctx->ZeDriver;
}

CHIPEvent *CHIPQueueLevel0::enqueueMarkerImpl() {
  CHIPEventLevel0 *MarkerEvent =
      (CHIPEventLevel0 *)Backend->createCHIPEvent(ChipContext_);

  MarkerEvent->Msg = "marker";
  auto Status =
      zeCommandListAppendSignalEvent(getCmdList(), MarkerEvent->peek());
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  return MarkerEvent;
}

CHIPEvent *
CHIPQueueLevel0::enqueueBarrierImpl(std::vector<CHIPEvent *> *EventsToWaitFor) {
  // Create an event, refc=2, add it to EventList
  CHIPEventLevel0 *EventToSignal =
      (CHIPEventLevel0 *)Backend->createCHIPEvent(ChipContext_);
  EventToSignal->Msg = "barrier";
  size_t NumEventsToWaitFor = 0;
  if (EventsToWaitFor)
    NumEventsToWaitFor = EventsToWaitFor->size();

  ze_event_handle_t *EventHandles = nullptr;
  ze_event_handle_t SignalEventHandle = nullptr;

  if (EventToSignal)
    SignalEventHandle = ((CHIPEventLevel0 *)(EventToSignal))->peek();

  if (NumEventsToWaitFor > 0) {
    EventHandles = new ze_event_handle_t[NumEventsToWaitFor];
    for (int i = 0; i < NumEventsToWaitFor; i++) {
      CHIPEventLevel0 *ChipEventLz = (CHIPEventLevel0 *)(*EventsToWaitFor)[i];
      EventHandles[i] = ChipEventLz->peek();
    }
  } // done gather Event_ handles to wait on

  auto Status = zeCommandListAppendBarrier(getCmdList(), SignalEventHandle,
                                           NumEventsToWaitFor, EventHandles);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  if (EventHandles)
    delete[] EventHandles;
  return EventToSignal;
}

CHIPEvent *CHIPQueueLevel0::memCopyAsyncImpl(void *Dst, const void *Src,
                                             size_t Size) {
  logTrace("CHIPQueueLevel0::memCopyAsync");
  CHIPContextLevel0 *ChipCtxZe = (CHIPContextLevel0 *)ChipContext_;
  CHIPEventLevel0 *Ev = (CHIPEventLevel0 *)Backend->createCHIPEvent(ChipCtxZe);

  Ev->Msg = "memCopy";

  ze_result_t Status;
  Status = zeCommandListAppendMemoryCopy(ZeCmdListImm_, Dst, Src, Size,
                                         Ev->peek(), 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);
  return Ev;
}

void CHIPQueueLevel0::finish() {
  // The finish Event_ that denotes the finish of current command list items
  pthread_yield();
  getLastEvent()->wait();
  return;
}

// End CHIPQueueLevelZero

// CHIPBackendLevel0
// ***********************************************************************

std::string CHIPBackendLevel0::getDefaultJitFlags() {
  return std::string(
      "-cl-std=CL2.0 -cl-take-global-address -cl-match-sincospi");
}

void CHIPBackendLevel0::initializeImpl(std::string CHIPPlatformStr,
                                       std::string CHIPDeviceTypeStr,
                                       std::string CHIPDeviceStr) {
  logTrace("CHIPBackendLevel0 Initialize");
  ze_result_t Status;
  Status = zeInit(0);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);

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
  zeDeviceGet(ZeDriver, &DeviceCount, nullptr);
  ZeDevices.resize(DeviceCount);
  zeDeviceGet(ZeDriver, &DeviceCount, ZeDevices.data());

  const ze_context_desc_t CtxDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr,
                                     0};

  ze_context_handle_t ZeCtx;
  zeContextCreateEx(ZeDriver, &CtxDesc, DeviceCount, ZeDevices.data(), &ZeCtx);
  CHIPContextLevel0 *ChipL0Ctx = new CHIPContextLevel0(ZeDriver, ZeCtx);
  Backend->addContext(ChipL0Ctx);

  // Filter in only devices of selected type and add them to the
  // backend as derivates of CHIPDevice
  for (int i = 0; i < DeviceCount; i++) {
    auto Dev = ZeDevices[i];
    ze_device_properties_t DeviceProperties{};
    DeviceProperties.pNext = nullptr;
    DeviceProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;

    zeDeviceGetProperties(Dev, &DeviceProperties);
    if (AnyDeviceType || ZeDeviceType == DeviceProperties.type) {
      CHIPDeviceLevel0 *ChipL0Dev =
          new CHIPDeviceLevel0(std::move(Dev), ChipL0Ctx, i);
      ChipL0Dev->populateDeviceProperties();
      ChipL0Ctx->addDevice(ChipL0Dev);

      auto Q = ChipL0Dev->addQueue(0, 0);

      Backend->addDevice(ChipL0Dev);
      Backend->addQueue(Q);
      break; // For now don't add more than one device
    }
  } // End adding CHIPDevices

  StaleEventMonitor_ =
      (CHIPStaleEventMonitorLevel0 *)Backend->createStaleEventMonitor();
}

// CHIPContextLevelZero
// ***********************************************************************

void *CHIPContextLevel0::allocateImpl(size_t Size, size_t Alignment,
                                      CHIPMemoryType MemTy) {
  Alignment = 0x1000; // TODO Where/why
  void *Ptr = 0;
  if (MemTy == CHIPMemoryType::Shared) {
    ze_device_mem_alloc_desc_t DmaDesc;
    DmaDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    DmaDesc.pNext = NULL;
    DmaDesc.flags = 0;
    DmaDesc.ordinal = 0;
    ze_host_mem_alloc_desc_t HmaDesc;
    HmaDesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    HmaDesc.pNext = NULL;
    HmaDesc.flags = 0;

    // TODO Check if devices support cross-device sharing?
    // ze_device_handle_t ZeDev = ((CHIPDeviceLevel0 *)getDevices()[0])->get();
    ze_device_handle_t ZeDev = nullptr; // Do not associate allocation

    ze_result_t Status = zeMemAllocShared(ZeCtx, &DmaDesc, &HmaDesc, Size,
                                          Alignment, ZeDev, &Ptr);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                                hipErrorMemoryAllocation);

    logTrace("LZ MEMORY ALLOCATE via calling zeMemAllocShared {} ", Status);

    return Ptr;
  } else if (MemTy == CHIPMemoryType::Device) {
    ze_device_mem_alloc_desc_t DmaDesc;
    DmaDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    DmaDesc.pNext = NULL;
    DmaDesc.flags = 0;
    DmaDesc.ordinal = 0;

    // TODO Select proper device
    ze_device_handle_t ZeDev = ((CHIPDeviceLevel0 *)getDevices()[0])->get();

    ze_result_t Status =
        zeMemAllocDevice(ZeCtx, &DmaDesc, Size, Alignment, ZeDev, &Ptr);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                                hipErrorMemoryAllocation);

    return Ptr;
  } else if (MemTy == CHIPMemoryType::Host) {
    // TODO
    ze_device_mem_alloc_desc_t DmaDesc;
    DmaDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    DmaDesc.pNext = NULL;
    DmaDesc.flags = 0;
    DmaDesc.ordinal = 0;
    ze_host_mem_alloc_desc_t HmaDesc;
    HmaDesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    HmaDesc.pNext = NULL;
    HmaDesc.flags = 0;

    // TODO Check if devices support cross-device sharing?
    // ze_device_handle_t ZeDev = ((CHIPDeviceLevel0 *)getDevices()[0])->get();
    ze_device_handle_t ZeDev = nullptr; // Do not associate allocation

    ze_result_t Status = zeMemAllocHost(ZeCtx, &HmaDesc, Size, Alignment, &Ptr);

    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                                hipErrorMemoryAllocation);
    logTrace("LZ MEMORY ALLOCATE via calling zeMemAllocShared {} ", Status);

    return Ptr;
  }
  CHIPERR_LOG_AND_THROW("Failed to allocate memory", hipErrorMemoryAllocation);
}

// CHIPDeviceLevelZero
// ***********************************************************************
CHIPDeviceLevel0::CHIPDeviceLevel0(ze_device_handle_t *ZeDev,
                                   CHIPContextLevel0 *ChipCtx, int Idx)
    : CHIPDevice(ChipCtx), ZeDev_(*ZeDev), ZeCtx_(ChipCtx->get()) {
  assert(Ctx_ != nullptr);
  Idx_ = Idx;
}
CHIPDeviceLevel0::CHIPDeviceLevel0(ze_device_handle_t &&ZeDev,
                                   CHIPContextLevel0 *ChipCtx, int Idx)
    : CHIPDevice(ChipCtx), ZeDev_(ZeDev), ZeCtx_(ChipCtx->get()) {
  assert(Ctx_ != nullptr);
  Idx_ = Idx;
}

void CHIPDeviceLevel0::reset() { UNIMPLEMENTED(); }

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

  // Copy device name
  if (255 < ZE_MAX_DEVICE_NAME) {
    strncpy(HipDeviceProps_.name, HipDeviceProps_.name, 255);
    HipDeviceProps_.name[255] = 0;
  } else {
    strncpy(HipDeviceProps_.name, HipDeviceProps_.name, ZE_MAX_DEVICE_NAME);
    HipDeviceProps_.name[ZE_MAX_DEVICE_NAME - 1] = 0;
  }

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
      ZeDeviceProps_.numEUsPerSubslice *
      ZeDeviceProps_.numSlices; // DeviceComputeProps.maxTotalGroupSize;
  //??? Dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  HipDeviceProps_.l2CacheSize = DeviceCacheProps.cacheSize;
  // Dev.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();

  // not actually correct
  HipDeviceProps_.totalConstMem = DeviceMemProps.totalSize;
  // ??? Dev.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();

  // as per gen architecture doc
  HipDeviceProps_.regsPerBlock = 4096;

  HipDeviceProps_.warpSize =
      DeviceComputeProps.subGroupSizes[DeviceComputeProps.numSubGroupSizes - 1];

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
}

CHIPQueue *CHIPDeviceLevel0::addQueueImpl(unsigned int Flags, int Priority) {
  CHIPQueueLevel0 *NewQ = new CHIPQueueLevel0(this, Flags, Priority);
  ChipQueues_.push_back(NewQ);
  return NewQ;
}

CHIPTexture *CHIPDeviceLevel0::createTexture(
    const hipResourceDesc *PResDesc, const hipTextureDesc *PTexDesc,
    const struct hipResourceViewDesc *PResViewDesc) {
  ze_image_handle_t ImageHandle;
  ze_sampler_handle_t SamplerHandle;
  auto Image =
      CHIPTextureLevel0::createImage(this, PResDesc, PTexDesc, PResViewDesc);
  auto Sampler =
      CHIPTextureLevel0::createSampler(this, PResDesc, PTexDesc, PResViewDesc);

  CHIPTextureLevel0 *ChipTexture =
      new CHIPTextureLevel0((intptr_t)Image, (intptr_t)Sampler);

  auto Q = (CHIPQueueLevel0 *)getActiveQueue();
  // Check if need to copy data in
  if (PResDesc->res.array.array != nullptr) {
    hipArray *HipArr = PResDesc->res.array.array;
    Q->memCopyToTexture(ChipTexture, (unsigned char *)HipArr->data);
  }

  return ChipTexture;
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

  CHIPDeviceLevel0 *ChipDevLz = (CHIPDeviceLevel0 *)ChipDev;
  CHIPContextLevel0 *ChipCtxLz = (CHIPContextLevel0 *)(ChipDev->getContext());

  ze_device_handle_t ZeDev = ((CHIPDeviceLevel0 *)ChipDev)->get();
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
    logTrace("ZE Build Log: {}", std::string(LogStr).c_str());
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
  for (int i = 0; i < KernelCount; i++) {
    std::string HostFName = KernelNames[i];
    logTrace("Registering kernel {}", HostFName);

    auto *FuncInfo = findFunctionInfo(HostFName);
    if (!FuncInfo) {
      // TODO: __syncthreads() gets turned into Intel_Symbol_Table_Void_Program
      // This is a call to OCML so it shouldn't be turned into a CHIPKernel
      continue;
      // CHIPERR_LOG_AND_THROW("Failed to find kernel in OpenCLFunctionInfoMap",
      //                      hipErrorInitializationError);
    }

    // Create kernel
    ze_kernel_handle_t ZeKernel;
    ze_kernel_desc_t KernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr,
                                   0, // flags
                                   HostFName.c_str()};
    Status = zeKernelCreate(ZeModule_, &KernelDesc, &ZeKernel);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
    logTrace("LZ KERNEL CREATION via calling zeKernelCreate {} ", Status);
    CHIPKernelLevel0 *ChipZeKernel =
        new CHIPKernelLevel0(ZeKernel, HostFName, FuncInfo);
    addKernel(ChipZeKernel);
  }
}

void CHIPExecItem::setupAllArgs() {
  CHIPKernelLevel0 *Kernel = (CHIPKernelLevel0 *)ChipKernel_;

  OCLFuncInfo *FuncInfo = ChipKernel_->getFuncInfo();

  size_t NumLocals = 0;
  int LastArgIdx = -1;

  for (size_t i = 0; i < FuncInfo->ArgTypeInfo.size(); ++i) {
    if (FuncInfo->ArgTypeInfo[i].Space == OCLSpace::Local) {
      ++NumLocals;
    }
  }
  // there can only be one dynamic shared mem variable, per cuda spec
  assert(NumLocals <= 1);

  // Argument processing for the new HIP launch API.
  if (ArgsPointer_) {
    for (size_t i = 0, ArgIdx = 0; i < FuncInfo->ArgTypeInfo.size();
         ++i, ++ArgIdx) {
      OCLArgTypeInfo &ArgTypeInfo = FuncInfo->ArgTypeInfo[i];

      if (ArgTypeInfo.Type == OCLType::Image) {
        CHIPTextureLevel0 *TexObj =
            (CHIPTextureLevel0 *)(*((unsigned long *)(ArgsPointer_[1])));

        // Set image part
        logTrace("setImageArg {} size {}\n", ArgIdx, ArgTypeInfo.Size);
        ze_result_t Status = zeKernelSetArgumentValue(
            Kernel->get(), ArgIdx, ArgTypeInfo.Size, &(TexObj->Image));
        CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

        // Set sampler part
        ArgIdx++;

        logTrace("setImageArg {} size {}\n", ArgIdx, ArgTypeInfo.Size);
        Status = zeKernelSetArgumentValue(Kernel->get(), ArgIdx,
                                          ArgTypeInfo.Size, &(TexObj->Sampler));
        CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
      } else {
        logTrace("setArg {} size {} addr {}\n", ArgIdx, ArgTypeInfo.Size,
                 ArgsPointer_[i]);
        ze_result_t Status = zeKernelSetArgumentValue(
            Kernel->get(), ArgIdx, ArgTypeInfo.Size, ArgsPointer_[i]);
        CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd,
                                    "zeKernelSetArgumentValue failed");
      }
    }
  } else {
    // Argument processing for the old HIP launch API.
    if ((OffsetSizes_.size() + NumLocals) != FuncInfo->ArgTypeInfo.size()) {
      CHIPERR_LOG_AND_THROW("Some arguments are still unset", hipErrorTbd);
    }

    if (OffsetSizes_.size() == 0)
      return;

    std::sort(OffsetSizes_.begin(), OffsetSizes_.end());
    if ((std::get<0>(OffsetSizes_[0]) != 0) ||
        (std::get<1>(OffsetSizes_[0]) == 0)) {
      CHIPERR_LOG_AND_THROW("Invalid offset/size", hipErrorTbd);
    }

    // check args are set
    if (OffsetSizes_.size() > 1) {
      for (size_t i = 1; i < OffsetSizes_.size(); ++i) {
        if ((std::get<0>(OffsetSizes_[i]) == 0) ||
            (std::get<1>(OffsetSizes_[i]) == 0) ||
            ((std::get<0>(OffsetSizes_[i - 1]) +
              std::get<1>(OffsetSizes_[i - 1])) >
             std::get<0>(OffsetSizes_[i]))) {
          CHIPERR_LOG_AND_THROW("Invalid offset/size", hipErrorTbd);
        }
      }
    }

    const unsigned char *Start = ArgData_.data();
    void *Ptr;
    int Err;
    for (size_t i = 0; i < OffsetSizes_.size(); ++i) {
      OCLArgTypeInfo &ArgTypeInfo = FuncInfo->ArgTypeInfo[i];
      logTrace("ARG {}: OS[0]: {} OS[1]: {} \n      TYPE {} SPAC {} SIZE {}\n",
               i, std::get<0>(OffsetSizes_[i]), std::get<1>(OffsetSizes_[i]),
               (unsigned)ArgTypeInfo.Type, (unsigned)ArgTypeInfo.Space,
               ArgTypeInfo.Size);

      if (ArgTypeInfo.Type == OCLType::Pointer) {
        // TODO: sync with ExecItem's solution
        assert(ArgTypeInfo.Size == sizeof(void *));
        assert(std::get<1>(OffsetSizes_[i]) == ArgTypeInfo.Size);
        size_t Size = std::get<1>(OffsetSizes_[i]);
        size_t Offset = std::get<0>(OffsetSizes_[i]);
        const void *Value = (void *)(Start + Offset);
        logTrace("setArg SVM {} to {}\n", i, Ptr);
        ze_result_t Status =
            zeKernelSetArgumentValue(Kernel->get(), i, Size, Value);

        CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd,
                                    "zeKernelSetArgumentValue failed");

        logTrace("LZ SET ARGUMENT VALUE via calling zeKernelSetArgumentValue "
                 "{} ",
                 Status);
      } else {
        size_t Size = std::get<1>(OffsetSizes_[i]);
        size_t Offset = std::get<0>(OffsetSizes_[i]);
        const void *Value = (void *)(Start + Offset);
        logTrace("setArg {} size {} offs {}\n", i, Size, Offset);
        ze_result_t Status =
            zeKernelSetArgumentValue(Kernel->get(), i, Size, Value);

        CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd,
                                    "zeKernelSetArgumentValue failed");

        logTrace("LZ SET ARGUMENT VALUE via calling zeKernelSetArgumentValue "
                 "{} ",
                 Status);
      }
    }
  }

  // Setup the kernel argument's value related to dynamically sized share
  // memory
  if (NumLocals == 1) {
    ze_result_t Status = zeKernelSetArgumentValue(
        Kernel->get(), FuncInfo->ArgTypeInfo.size() - 1, SharedMem_, nullptr);
    logTrace("LZ set dynamically sized share memory related argument via "
             "calling "
             "zeKernelSetArgumentValue {} ",
             Status);
  }

  return;
}

ze_image_handle_t *
CHIPTextureLevel0::createImage(CHIPDeviceLevel0 *ChipDev,
                               const hipResourceDesc *PResDesc,
                               const hipTextureDesc *PTexDesc,
                               const struct hipResourceViewDesc *PResViewDesc) {
  if (!PResDesc)
    CHIPERR_LOG_AND_THROW("Resource descriptor is null", hipErrorTbd);
  if (PResDesc->resType != hipResourceTypeArray) {
    CHIPERR_LOG_AND_THROW("only support hipArray as image storage",
                          hipErrorTbd);
  }

  hipArray *HipArr = PResDesc->res.array.array;
  if (!HipArr)
    CHIPERR_LOG_AND_THROW("hipResourceViewDesc result array is null",
                          hipErrorTbd);
  hipChannelFormatDesc ChannelDesc = HipArr->desc;

  ze_image_format_layout_t FormatLayout = ZE_IMAGE_FORMAT_LAYOUT_32;
  if (ChannelDesc.x == 8) {
    FormatLayout = ZE_IMAGE_FORMAT_LAYOUT_8;
  } else if (ChannelDesc.x == 16) {
    FormatLayout = ZE_IMAGE_FORMAT_LAYOUT_16;
  } else if (ChannelDesc.x == 32) {
    FormatLayout = ZE_IMAGE_FORMAT_LAYOUT_32;
  } else {
    CHIPERR_LOG_AND_THROW("hipChannelFormatDesc value is out of the scope",
                          hipErrorTbd);
  }

  ze_image_format_type_t FormatType = ZE_IMAGE_FORMAT_TYPE_FLOAT;
  if (ChannelDesc.f == hipChannelFormatKindSigned) {
    FormatType = ZE_IMAGE_FORMAT_TYPE_SINT;
  } else if (ChannelDesc.f == hipChannelFormatKindUnsigned) {
    FormatType = ZE_IMAGE_FORMAT_TYPE_UINT;
  } else if (ChannelDesc.f == hipChannelFormatKindFloat) {
    FormatType = ZE_IMAGE_FORMAT_TYPE_FLOAT;
  } else if (ChannelDesc.f == hipChannelFormatKindNone) {
    FormatType = ZE_IMAGE_FORMAT_TYPE_FORCE_UINT32;
  } else {
    CHIPERR_LOG_AND_THROW("hipChannelFormatDesc value is out of the scope",
                          hipErrorTbd);
  }

  ze_image_format_t Format = {FormatLayout,
                              FormatType,
                              ZE_IMAGE_FORMAT_SWIZZLE_R,
                              ZE_IMAGE_FORMAT_SWIZZLE_0,
                              ZE_IMAGE_FORMAT_SWIZZLE_0,
                              ZE_IMAGE_FORMAT_SWIZZLE_1};

  ze_image_type_t ImageType = ZE_IMAGE_TYPE_2D;

  ze_image_desc_t ImageDesc = {ZE_STRUCTURE_TYPE_IMAGE_DESC, nullptr,
                               0, // read-only
                               ImageType, Format,
                               // 128, 128, 0, 0, 0
                               HipArr->width, HipArr->height, 0, 0, 0};

  // Create LZ image handle
  CHIPContextLevel0 *ChipCtxLz = (CHIPContextLevel0 *)ChipDev->getContext();
  ze_image_handle_t *Image = new ze_image_handle_t();
  ze_result_t Status =
      zeImageCreate(ChipCtxLz->get(), ChipDev->get(), &ImageDesc, Image);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  return Image;
}

ze_sampler_handle_t *CHIPTextureLevel0::createSampler(
    CHIPDeviceLevel0 *ChipDev, const hipResourceDesc *PResDesc,
    const hipTextureDesc *PTexDesc,
    const struct hipResourceViewDesc *PResViewDesc) {
  // Identify the address mode
  ze_sampler_address_mode_t AddressMode = ZE_SAMPLER_ADDRESS_MODE_NONE;
  if (PTexDesc->addressMode[0] == hipAddressModeWrap)
    AddressMode = ZE_SAMPLER_ADDRESS_MODE_NONE;
  else if (PTexDesc->addressMode[0] == hipAddressModeClamp)
    AddressMode = ZE_SAMPLER_ADDRESS_MODE_CLAMP;
  else if (PTexDesc->addressMode[0] == hipAddressModeMirror)
    AddressMode = ZE_SAMPLER_ADDRESS_MODE_MIRROR;
  else if (PTexDesc->addressMode[0] == hipAddressModeBorder)
    AddressMode = ZE_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;

  // Identify the filter mode
  ze_sampler_filter_mode_t FilterMode = ZE_SAMPLER_FILTER_MODE_NEAREST;
  if (PTexDesc->filterMode == hipFilterModePoint)
    FilterMode = ZE_SAMPLER_FILTER_MODE_NEAREST;
  else if (PTexDesc->filterMode == hipFilterModeLinear)
    FilterMode = ZE_SAMPLER_FILTER_MODE_LINEAR;

  // Identify the normalization
  ze_bool_t IsNormalized = 0;
  if (PTexDesc->normalizedCoords == 0)
    IsNormalized = 0;
  else
    IsNormalized = 1;

  ze_sampler_desc_t SamplerDesc = {ZE_STRUCTURE_TYPE_SAMPLER_DESC, nullptr,
                                   AddressMode, FilterMode, IsNormalized};

  // Create LZ samler handle
  CHIPContextLevel0 *ChipCtxLz = (CHIPContextLevel0 *)ChipDev->getContext();
  ze_sampler_handle_t *Sampler = new ze_sampler_handle_t();
  ze_result_t Status =
      zeSamplerCreate(ChipCtxLz->get(), ChipDev->get(), &SamplerDesc, Sampler);
  CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

  return Sampler;
}
