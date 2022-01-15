#include "Level0Backend.hh"

// CHIPEventLevel0
// ***********************************************************************

CHIPEventLevel0::~CHIPEventLevel0() {
  zeEventDestroy(event);
  zeEventPoolDestroy(event_pool);
  event = nullptr;
  event_pool = nullptr;
}

CHIPEventLevel0::CHIPEventLevel0(CHIPContextLevel0* chip_ctx_,
                                 CHIPEventFlags flags_)
    : CHIPEvent((CHIPContext*)(chip_ctx_), flags_) {
  CHIPContextLevel0* ze_ctx = (CHIPContextLevel0*)chip_context;

  unsigned int pool_flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
  // if (!flags.isDisableTiming())
  //   pool_flags = pool_flags | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;

  ze_event_pool_desc_t eventPoolDesc = {
      ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr,
      pool_flags,  // event in pool are visible to host
      1            // count
  };

  ze_result_t status =
      zeEventPoolCreate(ze_ctx->get(), &eventPoolDesc, 0, nullptr, &event_pool);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd,
                              "Level Zero event pool creation fail! ");

  ze_event_desc_t eventDesc = {
      ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr,
      0,                         // index
      ZE_EVENT_SCOPE_FLAG_HOST,  // ensure memory/cache coherency required on
                                 // signal
      ZE_EVENT_SCOPE_FLAG_HOST   // ensure memory coherency across device and
                                 // Host after event completes
  };

  status = zeEventCreate(event_pool, &eventDesc, &event);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd,
                              "Level Zero event creation fail! ");
}

void CHIPEventLevel0::takeOver(CHIPEvent* other_) {
  // Take over target queues event
  CHIPEventLevel0* other = (CHIPEventLevel0*)other_;
  if (*refc > 1) decreaseRefCount();
  this->event = other->get();  // increases refcount
  this->event_pool = other->event_pool;
  this->msg = other->msg;
  this->refc = other->refc;
}

// Must use this for now - Level Zero hangs when events are host visible +
// kernel timings are enabled
void CHIPEventLevel0::recordStream(CHIPQueue* chip_queue_) {
  std::lock_guard<std::mutex> Lock(mtx);
  ze_result_t status;
  if (event_status == EVENT_STATUS_RECORDED) {
    ze_result_t status = zeEventHostReset(event);
    CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd);
  }

  if (chip_queue_ == nullptr)
    CHIPERR_LOG_AND_THROW("Queue passed in is null", hipErrorTbd);

  CHIPQueueLevel0* q = (CHIPQueueLevel0*)chip_queue_;

  status = zeCommandListAppendBarrier(q->getCmdList(), nullptr, 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd);

  status = zeCommandListAppendWriteGlobalTimestamp(
      q->getCmdList(), (uint64_t*)(q->getSharedBufffer()), nullptr, 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd);

  status = zeCommandListAppendBarrier(q->getCmdList(), nullptr, 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd);

  status = zeCommandListAppendMemoryCopy(q->getCmdList(), &timestamp,
                                         q->getSharedBufffer(),
                                         sizeof(uint64_t), event, 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd);

  event_status = EVENT_STATUS_RECORDING;
  return;
}

bool CHIPEventLevel0::wait() {
  std::lock_guard<std::mutex> Lock(mtx);
  if (event_status != EVENT_STATUS_RECORDING) return false;

  ze_result_t status = zeEventHostSynchronize(event, UINT64_MAX);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd);

  event_status = EVENT_STATUS_RECORDED;
  return true;
}

bool CHIPEventLevel0::updateFinishStatus() {
  std::lock_guard<std::mutex> Lock(mtx);
  if (event_status != EVENT_STATUS_RECORDING) return false;

  ze_result_t status = zeEventQueryStatus(event);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd);
  if (status == ZE_RESULT_SUCCESS) event_status = EVENT_STATUS_RECORDED;

  return true;
}

/** This Doesn't work right now due to Level Zero Backend hanging?
  unsinged long CHIPEventLevel0::getFinishTime() {
    std::lock_guard<std::mutex> Lock(mtx);
    ze_kernel_timestamp_result_t res{};
    auto status = zeEventQueryKernelTimestamp(event, &res);
    CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd);

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
  std::lock_guard<std::mutex> Lock(mtx);
  CHIPContextLevel0* chip_ctx_lz = (CHIPContextLevel0*)chip_context;
  CHIPDeviceLevel0* chip_dev_lz =
      (CHIPDeviceLevel0*)chip_ctx_lz->getDevices()[0];
  auto props = chip_dev_lz->getDeviceProps();

  uint64_t timerResolution = props->timerResolution;
  uint32_t timestampValidBits = props->timestampValidBits;

  uint32_t t = (timestamp & (((uint64_t)1 << timestampValidBits) - 1));
  t = t * timerResolution;

  return t;
}

float CHIPEventLevel0::getElapsedTime(CHIPEvent* other_) {
  /**
   * Modified HIPLZ Implementation
   * https://github.com/intel/pti-gpu/blob/master/chapters/device_activity_tracing/LevelZero.md
   */
  logTrace("CHIPEventLevel0::getElapsedTime()");
  CHIPEventLevel0* other = (CHIPEventLevel0*)other_;
  // std::lock_guard<std::mutex> Lock(ContextMutex);

  if (!this->isRecordingOrRecorded() || !other->isRecordingOrRecorded())
    return hipErrorInvalidResourceHandle;

  this->updateFinishStatus();
  other->updateFinishStatus();
  if (!this->isFinished() || !other->isFinished()) return hipErrorNotReady;

  unsigned long Started = this->getFinishTime();
  unsigned long Finished = other->getFinishTime();

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
  auto ms = (float)MS + FractInMS;

  return ms;
}

void CHIPEventLevel0::hostSignal() {
  logTrace("CHIPEventLevel0::hostSignal()");
  auto status = zeEventHostSignal(event);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd);

  event_status = EVENT_STATUS_RECORDED;
}

void CHIPEventLevel0::barrier(CHIPQueue* chip_queue_) {
  CHIPQueueLevel0* chip_queue = (CHIPQueueLevel0*)chip_queue_;
  ze_result_t status =
      zeCommandListAppendBarrier(chip_queue->getCmdList(), nullptr, 1, &event);

  event_status = EVENT_STATUS_RECORDING;
}

// End CHIPEventLevel0

// CHIPCallbackDataLevel0
// ***********************************************************************

CHIPCallbackDataLevel0::CHIPCallbackDataLevel0(hipStreamCallback_t callback_f_,
                                               void* callback_args_,
                                               CHIPQueue* chip_queue_)
    : CHIPCallbackData(callback_f_, callback_args_, chip_queue_) {}

void CHIPCallbackDataLevel0::setup() {
  // create a pool
  // create events from pool
  // assign events to parent events
  UNIMPLEMENTED();
}
// End CHIPCallbackDataLevel0

// CHIPEventMonitorLevel0
// ***********************************************************************

CHIPEventMonitorLevel0::CHIPEventMonitorLevel0() : CHIPEventMonitor() {
  logTrace("CHIPEventMonitorLevel0::CHIPEventMonitorLevel0()");
};

void CHIPEventMonitorLevel0::monitor() {
  logTrace("CHIPEventMonitorLevel0::monitor()");
  CHIPEventMonitor::monitor();
}
// End CHIPEventMonitorLevel0

// CHIPKernelLevelZero
// ***********************************************************************

CHIPKernelLevel0::CHIPKernelLevel0(ze_kernel_handle_t ze_kernel_,
                                   std::string host_f_name_,
                                   OCLFuncInfo* func_info_)
    : CHIPKernel(host_f_name_, func_info_) {
  ze_kernel = ze_kernel_;
  logTrace("CHIPKernelLevel0 constructor via ze_kernel_handle");
}
// End CHIPKernelLevelZero

// CHIPQueueLevelZero
// ***********************************************************************

CHIPEventLevel0* CHIPQueueLevel0::getLastEvent() {
  return (CHIPEventLevel0*)LastEvent;
}

CHIPQueueLevel0::CHIPQueueLevel0(CHIPDeviceLevel0* chip_dev_)
    : CHIPQueue(chip_dev_) {
  ze_result_t status;
  auto chip_dev_lz = chip_dev_;
  auto ctx = chip_dev_lz->getContext();
  auto chip_context_lz = (CHIPContextLevel0*)ctx;

  ze_ctx = chip_context_lz->get();
  ze_dev = chip_dev_lz->get();

  logTrace(
      "CHIPQueueLevel0 constructor called via CHIPContextLevel0 and "
      "CHIPDeviceLevel0");

  // Discover all command queue groups
  uint32_t cmdqueueGroupCount = 0;
  zeDeviceGetCommandQueueGroupProperties(ze_dev, &cmdqueueGroupCount, nullptr);
  logTrace("CommandGroups found: {}", cmdqueueGroupCount);

  ze_command_queue_group_properties_t* cmdqueueGroupProperties =
      (ze_command_queue_group_properties_t*)malloc(
          cmdqueueGroupCount * sizeof(ze_command_queue_group_properties_t));
  zeDeviceGetCommandQueueGroupProperties(ze_dev, &cmdqueueGroupCount,
                                         cmdqueueGroupProperties);

  // Find a command queue type that support compute
  uint32_t computeQueueGroupOrdinal = cmdqueueGroupCount;
  for (uint32_t i = 0; i < cmdqueueGroupCount; ++i) {
    if (cmdqueueGroupProperties[i].flags &
        ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
      computeQueueGroupOrdinal = i;
      logTrace("Found compute command group");
      break;
    }
  }
  ze_command_queue_desc_t commandQueueDesc = {
      ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
      nullptr,
      computeQueueGroupOrdinal,
      0,  // index
      0,  // flags
      ZE_COMMAND_QUEUE_MODE_DEFAULT,
      ZE_COMMAND_QUEUE_PRIORITY_NORMAL};

  // Create a default command queue (in case need to pass it outside of
  status = zeCommandQueueCreate(ze_ctx, ze_dev, &commandQueueDesc, &ze_cmd_q);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);

  // CHIP-SPV) Create an immediate command list
  status = zeCommandListCreateImmediate(ze_ctx, ze_dev, &commandQueueDesc,
                                        &ze_cmd_list_imm);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);
  chip_context->addQueue(this);
  chip_device->addQueue(this);

  // Initialize the internal event pool and finish event
  ze_event_pool_desc_t ep_desc = {};
  ep_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
  ep_desc.count = 1;
  ep_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
  ze_event_desc_t ev_desc = {};
  ev_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
  ev_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
  ev_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
  status = zeEventPoolCreate(ze_ctx, &ep_desc, 1, &ze_dev, &event_pool);

  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd,
                              "zeEventPoolCreate FAILED");

  status = zeEventCreate(event_pool, &ev_desc, &finish_event);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd,
                              "zeEventCreate FAILED with return code");

  // Initialize the shared memory buffer
  // TODO This does not record the buffer allocation in device allocation
  // tracker
  shared_buf = chip_context_lz->allocate_(32, 8, CHIPMemoryType::Shared);

  // Initialize the uint64_t part as 0
  *(uint64_t*)this->shared_buf = 0;
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
  setLastEvent(enqueueMarkerImpl());
}

CHIPEvent* CHIPQueueLevel0::launchImpl(CHIPExecItem* exec_item) {
  CHIPContextLevel0* chip_ctx_ze = (CHIPContextLevel0*)chip_context;
  CHIPEventLevel0* ev = new CHIPEventLevel0(chip_ctx_ze);
  ev->msg = "launch";

  CHIPKernelLevel0* chip_kernel = (CHIPKernelLevel0*)exec_item->getKernel();
  ze_kernel_handle_t kernel_ze = chip_kernel->get();
  logTrace("Launching Kernel {}", chip_kernel->getName());

  ze_result_t status =
      zeKernelSetGroupSize(kernel_ze, exec_item->getBlock().x,
                           exec_item->getBlock().y, exec_item->getBlock().z);

  exec_item->setupAllArgs();
  auto x = exec_item->getGrid().x;
  auto y = exec_item->getGrid().y;
  auto z = exec_item->getGrid().z;
  ze_group_count_t launchArgs = {x, y, z};
  status = zeCommandListAppendLaunchKernel(ze_cmd_list_imm, kernel_ze,
                                           &launchArgs, ev->peek(), 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);
  return ev;
}

CHIPEvent* CHIPQueueLevel0::memFillAsyncImpl(void* dst, size_t size,
                                             const void* pattern,
                                             size_t pattern_size) {
  CHIPContextLevel0* chip_ctx_ze = (CHIPContextLevel0*)chip_context;
  CHIPEventLevel0* ev = new CHIPEventLevel0(chip_ctx_ze);
  ev->msg = "memFill";
  ze_result_t status =
      zeCommandListAppendMemoryFill(ze_cmd_list_imm, dst, pattern, pattern_size,
                                    size, ev->peek(), 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd);
  return ev;
};

CHIPEvent* CHIPQueueLevel0::memCopy2DAsyncImpl(void* dst, size_t dpitch,
                                               const void* src, size_t spitch,
                                               size_t width, size_t height) {
  return memCopy3DAsyncImpl(dst, dpitch, 0, src, spitch, 0, width, height, 0);
};

CHIPEvent* CHIPQueueLevel0::memCopy3DAsyncImpl(void* dst, size_t dpitch,
                                               size_t dspitch, const void* src,
                                               size_t spitch, size_t sspitch,
                                               size_t width, size_t height,
                                               size_t depth) {
  CHIPContextLevel0* chip_ctx_ze = (CHIPContextLevel0*)chip_context;
  CHIPEventLevel0* ev = new CHIPEventLevel0(chip_ctx_ze);
  ev->msg = "memCopy3DAsync";

  ze_copy_region_t dstRegion;
  dstRegion.originX = 0;
  dstRegion.originY = 0;
  dstRegion.originZ = 0;
  dstRegion.width = width;
  dstRegion.height = height;
  dstRegion.depth = depth;
  ze_copy_region_t srcRegion;
  srcRegion.originX = 0;
  srcRegion.originY = 0;
  srcRegion.originZ = 0;
  srcRegion.width = width;
  srcRegion.height = height;
  srcRegion.depth = depth;
  ze_result_t status = zeCommandListAppendMemoryCopyRegion(
      ze_cmd_list_imm, dst, &dstRegion, dpitch, dspitch, src, &srcRegion,
      spitch, sspitch, ev->peek(), 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd);
  return ev;
};

// Memory copy to texture object, i.e. image
CHIPEvent* CHIPQueueLevel0::memCopyToTextureImpl(CHIPTexture* texObj,
                                                 void* src) {
  CHIPContextLevel0* chip_ctx_ze = (CHIPContextLevel0*)chip_context;
  CHIPEventLevel0* ev = new CHIPEventLevel0(chip_ctx_ze);
  ev->msg = "memCopyToTexture";

  ze_image_handle_t imageHandle = (ze_image_handle_t)texObj->image;
  ze_result_t status = zeCommandListAppendImageCopyFromMemory(
      ze_cmd_list_imm, imageHandle, src, 0, ev->peek(), 0, 0);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd);
  return ev;
};

void CHIPQueueLevel0::getBackendHandles(unsigned long* nativeInfo, int* size) {
  logTrace("CHIPQueueLevel0::getBackendHandles");
  *size = 4;

  // Get queue handler
  nativeInfo[3] = (unsigned long)ze_cmd_q;

  // Get context handler
  CHIPContextLevel0* ctx = (CHIPContextLevel0*)chip_context;
  nativeInfo[2] = (unsigned long)ctx->get();

  // Get device handler
  CHIPDeviceLevel0* dev = (CHIPDeviceLevel0*)chip_device;
  nativeInfo[1] = (unsigned long)dev->get();

  // Get driver handler
  nativeInfo[0] = (unsigned long)ctx->ze_driver;
}

CHIPEvent* CHIPQueueLevel0::enqueueMarkerImpl() {
  CHIPEventLevel0* marker_event =
      new CHIPEventLevel0((CHIPContextLevel0*)chip_context);
  marker_event->msg = "marker";
  auto status =
      zeCommandListAppendSignalEvent(getCmdList(), marker_event->peek());
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd);
  return marker_event;
}

CHIPEvent* CHIPQueueLevel0::enqueueBarrierImpl(
    std::vector<CHIPEvent*>* eventsToWaitFor) {
  CHIPEventLevel0* eventToSignal =
      new CHIPEventLevel0((CHIPContextLevel0*)chip_context);
  eventToSignal->msg = "barrier";
  size_t numEventsToWaitFor = 0;
  if (eventsToWaitFor) numEventsToWaitFor = eventsToWaitFor->size();

  ze_event_handle_t* event_handles = nullptr;
  ze_event_handle_t signal_event_handle = nullptr;

  if (eventToSignal)
    signal_event_handle = ((CHIPEventLevel0*)(eventToSignal))->peek();

  if (numEventsToWaitFor > 0) {
    event_handles = new ze_event_handle_t[numEventsToWaitFor];
    for (int i = 0; i < numEventsToWaitFor; i++) {
      CHIPEventLevel0* chip_event_lz = (CHIPEventLevel0*)(*eventsToWaitFor)[i];
      event_handles[i] = chip_event_lz->peek();
    }
  }  // done gather event handles to wait on

  auto status = zeCommandListAppendBarrier(getCmdList(), signal_event_handle,
                                           numEventsToWaitFor, event_handles);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd);

  if (event_handles) delete event_handles;
  return eventToSignal;
}

CHIPEvent* CHIPQueueLevel0::memCopyAsyncImpl(void* dst, const void* src,
                                             size_t size) {
  logTrace("CHIPQueueLevel0::memCopyAsync");
  CHIPContextLevel0* chip_ctx_ze = (CHIPContextLevel0*)chip_context;
  CHIPEventLevel0* ev = new CHIPEventLevel0(chip_ctx_ze);
  ev->msg = "memCopy";

  ze_result_t status;
  status = zeCommandListAppendMemoryCopy(ze_cmd_list_imm, dst, src, size,
                                         ev->peek(), 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);
  return ev;
}

void CHIPQueueLevel0::finish() {
  // The finish event that denotes the finish of current command list items
  auto status =
      zeCommandListAppendBarrier(ze_cmd_list_imm, finish_event, 0, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(
      status, ZE_RESULT_SUCCESS, hipErrorTbd,
      "zeCommandListAppendBarrier FAILED with return code");

  status = zeEventHostSynchronize(finish_event, UINT64_MAX);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd,
                              "zeEventHostSynchronize FAILED with return code");

  status = zeEventHostReset(finish_event);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd,
                              "zeEventHostReset FAILED with return code");
  return;
}

// End CHIPQueueLevelZero

// CHIPBackendLevel0
// ***********************************************************************

std::string CHIPBackendLevel0::getDefaultJitFlags() {
  return std::string(
      "-cl-std=CL2.0 -cl-take-global-address -cl-match-sincospi");
}

void CHIPBackendLevel0::initialize_(std::string CHIPPlatformStr,
                                    std::string CHIPDeviceTypeStr,
                                    std::string CHIPDeviceStr) {
  logTrace("CHIPBackendLevel0 Initialize");
  ze_result_t status;
  status = zeInit(0);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);

  bool any_device_type = false;
  ze_device_type_t ze_device_type;
  if (!CHIPDeviceTypeStr.compare("gpu")) {
    ze_device_type = ZE_DEVICE_TYPE_GPU;
  } else if (!CHIPDeviceTypeStr.compare("fpga")) {
    ze_device_type = ZE_DEVICE_TYPE_FPGA;
  } else if (!CHIPDeviceTypeStr.compare("default")) {
    // For 'default' pick all devices of any type.
    any_device_type = true;
  } else {
    CHIPERR_LOG_AND_THROW("CHIP_DEVICE_TYPE must be either gpu or fpga",
                          hipErrorInitializationError);
  }
  int platform_idx = std::atoi(CHIPPlatformStr.c_str());
  std::vector<ze_driver_handle_t> ze_drivers;
  std::vector<ze_device_handle_t> ze_devices;

  // Get number of drivers
  uint32_t driverCount = 0, deviceCount = 0;
  status = zeDriverGet(&driverCount, nullptr);
  logTrace("Found Level0 Drivers: {}", driverCount);
  // Resize and fill ze_driver vector with drivers
  ze_drivers.resize(driverCount);
  status = zeDriverGet(&driverCount, ze_drivers.data());

  // TODO Allow for multilpe platforms(drivers)
  // TODO Check platform ID is not the same as OpenCL. You can have
  // two OCL platforms but only one level0 driver
  ze_driver_handle_t ze_driver = ze_drivers[platform_idx];

  assert(ze_driver != nullptr);
  // Load devices to device vector
  zeDeviceGet(ze_driver, &deviceCount, nullptr);
  ze_devices.resize(deviceCount);
  zeDeviceGet(ze_driver, &deviceCount, ze_devices.data());

  const ze_context_desc_t ctxDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr,
                                     0};

  ze_context_handle_t ze_ctx;
  zeContextCreateEx(ze_driver, &ctxDesc, deviceCount, ze_devices.data(),
                    &ze_ctx);
  CHIPContextLevel0* chip_l0_ctx = new CHIPContextLevel0(ze_driver, ze_ctx);
  Backend->addContext(chip_l0_ctx);

  // Filter in only devices of selected type and add them to the
  // backend as derivates of CHIPDevice
  for (int i = 0; i < deviceCount; i++) {
    auto dev = ze_devices[i];
    ze_device_properties_t device_properties;
    zeDeviceGetProperties(dev, &device_properties);
    if (any_device_type || ze_device_type == device_properties.type) {
      CHIPDeviceLevel0* chip_l0_dev =
          new CHIPDeviceLevel0(std::move(dev), chip_l0_ctx);
      chip_l0_dev->populateDeviceProperties();
      chip_l0_ctx->addDevice(chip_l0_dev);

      CHIPQueueLevel0* q = new CHIPQueueLevel0(chip_l0_dev);
      // chip_l0_dev->addQueue(q);
      Backend->addDevice(chip_l0_dev);
      Backend->addQueue(q);
      break;  // For now don't add more than one device
    }
  }  // End adding CHIPDevices
}

// CHIPContextLevelZero
// ***********************************************************************

void* CHIPContextLevel0::allocate_(size_t size, size_t alignment,
                                   CHIPMemoryType memTy) {
  alignment = 0x1000;  // TODO Where/why
  void* ptr = 0;
  if (memTy == CHIPMemoryType::Shared) {
    ze_device_mem_alloc_desc_t dmaDesc;
    dmaDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    dmaDesc.pNext = NULL;
    dmaDesc.flags = 0;
    dmaDesc.ordinal = 0;
    ze_host_mem_alloc_desc_t hmaDesc;
    hmaDesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    hmaDesc.pNext = NULL;
    hmaDesc.flags = 0;

    // TODO Check if devices support cross-device sharing?
    ze_device_handle_t ze_dev = ((CHIPDeviceLevel0*)getDevices()[0])->get();
    ze_dev = nullptr;  // Do not associate allocation

    ze_result_t status = zeMemAllocShared(ze_ctx, &dmaDesc, &hmaDesc, size,
                                          alignment, ze_dev, &ptr);
    CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS,
                                hipErrorMemoryAllocation);

    logTrace("LZ MEMORY ALLOCATE via calling zeMemAllocShared {} ", status);

    return ptr;
  } else if (memTy == CHIPMemoryType::Device) {
    ze_device_mem_alloc_desc_t dmaDesc;
    dmaDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    dmaDesc.pNext = NULL;
    dmaDesc.flags = 0;
    dmaDesc.ordinal = 0;

    // TODO Select proper device
    ze_device_handle_t ze_dev = ((CHIPDeviceLevel0*)getDevices()[0])->get();

    ze_result_t status =
        zeMemAllocDevice(ze_ctx, &dmaDesc, size, alignment, ze_dev, &ptr);
    CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS,
                                hipErrorMemoryAllocation);

    return ptr;
  } else if (memTy == CHIPMemoryType::Host) {
    // TODO
    ze_device_mem_alloc_desc_t dmaDesc;
    dmaDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    dmaDesc.pNext = NULL;
    dmaDesc.flags = 0;
    dmaDesc.ordinal = 0;
    ze_host_mem_alloc_desc_t hmaDesc;
    hmaDesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    hmaDesc.pNext = NULL;
    hmaDesc.flags = 0;

    // TODO Check if devices support cross-device sharing?
    ze_device_handle_t ze_dev = ((CHIPDeviceLevel0*)getDevices()[0])->get();
    ze_dev = nullptr;  // Do not associate allocation

    ze_result_t status = zeMemAllocShared(ze_ctx, &dmaDesc, &hmaDesc, size,
                                          alignment, ze_dev, &ptr);

    CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS,
                                hipErrorMemoryAllocation);
    logTrace("LZ MEMORY ALLOCATE via calling zeMemAllocShared {} ", status);

    return ptr;
  }
  CHIPERR_LOG_AND_THROW("Failed to allocate memory", hipErrorMemoryAllocation);
}

// CHIPDeviceLevelZero
// ***********************************************************************
CHIPDeviceLevel0::CHIPDeviceLevel0(ze_device_handle_t* ze_dev_,
                                   CHIPContextLevel0* chip_ctx_)
    : CHIPDevice(chip_ctx_), ze_dev(*ze_dev_), ze_ctx(chip_ctx_->get()) {
  assert(ctx != nullptr);
}
CHIPDeviceLevel0::CHIPDeviceLevel0(ze_device_handle_t&& ze_dev_,
                                   CHIPContextLevel0* chip_ctx_)
    : CHIPDevice(chip_ctx_), ze_dev(ze_dev_), ze_ctx(chip_ctx_->get()) {
  assert(ctx != nullptr);
}

void CHIPDeviceLevel0::reset() { UNIMPLEMENTED(); }

void CHIPDeviceLevel0::populateDeviceProperties_() {
  ze_result_t status = ZE_RESULT_SUCCESS;

  // Initialize members used as input for zeDeviceGet*Properties() calls.
  ze_device_props.pNext = nullptr;
  ze_device_memory_properties_t device_mem_props;
  device_mem_props.pNext = nullptr;
  ze_device_compute_properties_t device_compute_props;
  device_compute_props.pNext = nullptr;
  ze_device_cache_properties_t device_cache_props;
  device_cache_props.pNext = nullptr;
  ze_device_module_properties_t device_module_props;
  device_module_props.pNext = nullptr;

  // Query device properties
  status = zeDeviceGetProperties(ze_dev, &ze_device_props);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);

  // Query device memory properties
  uint32_t count = 1;
  status = zeDeviceGetMemoryProperties(ze_dev, &count, &device_mem_props);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);

  // Query device computation properties
  status = zeDeviceGetComputeProperties(ze_dev, &device_compute_props);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);

  // Query device cache properties
  count = 1;
  status = zeDeviceGetCacheProperties(ze_dev, &count, &device_cache_props);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);

  // Query device module properties
  status = zeDeviceGetModuleProperties(ze_dev, &device_module_props);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS,
                              hipErrorInitializationError);

  // Copy device name
  if (255 < ZE_MAX_DEVICE_NAME) {
    strncpy(hip_device_props.name, hip_device_props.name, 255);
    hip_device_props.name[255] = 0;
  } else {
    strncpy(hip_device_props.name, hip_device_props.name, ZE_MAX_DEVICE_NAME);
    hip_device_props.name[ZE_MAX_DEVICE_NAME - 1] = 0;
  }

  // Get total device memory
  hip_device_props.totalGlobalMem = device_mem_props.totalSize;

  hip_device_props.sharedMemPerBlock =
      device_compute_props.maxSharedLocalMemory;
  //??? Dev.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>(&err);

  hip_device_props.maxThreadsPerBlock = device_compute_props.maxTotalGroupSize;
  //??? Dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&err);

  hip_device_props.maxThreadsDim[0] = device_compute_props.maxGroupSizeX;
  hip_device_props.maxThreadsDim[1] = device_compute_props.maxGroupSizeY;
  hip_device_props.maxThreadsDim[2] = device_compute_props.maxGroupSizeZ;

  // Maximum configured clock frequency of the device in MHz.
  hip_device_props.clockRate =
      1000 * ze_device_props.coreClockRate;  // deviceMemoryProps.maxClockRate;
  // Dev.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();

  hip_device_props.multiProcessorCount =
      ze_device_props.numEUsPerSubslice *
      ze_device_props.numSlices;  // device_compute_props.maxTotalGroupSize;
  //??? Dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  hip_device_props.l2CacheSize = device_cache_props.cacheSize;
  // Dev.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();

  // not actually correct
  hip_device_props.totalConstMem = device_mem_props.totalSize;
  // ??? Dev.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();

  // as per gen architecture doc
  hip_device_props.regsPerBlock = 4096;

  hip_device_props.warpSize =
      device_compute_props
          .subGroupSizes[device_compute_props.numSubGroupSizes - 1];

  // Replicate from OpenCL implementation

  // HIP and LZ uses int and uint32_t, respectively, for storing the
  // group count. Clamp the group count to INT_MAX to avoid 2^31+ size
  // being interpreted as negative number.
  constexpr unsigned int_max = std::numeric_limits<int>::max();
  hip_device_props.maxGridSize[0] =
      std::min(device_compute_props.maxGroupCountX, int_max);
  hip_device_props.maxGridSize[1] =
      std::min(device_compute_props.maxGroupCountY, int_max);
  hip_device_props.maxGridSize[2] =
      std::min(device_compute_props.maxGroupCountZ, int_max);
  hip_device_props.memoryClockRate = device_mem_props.maxClockRate;
  hip_device_props.memoryBusWidth = device_mem_props.maxBusWidth;
  hip_device_props.major = 2;
  hip_device_props.minor = 0;

  hip_device_props.maxThreadsPerMultiProcessor =
      ze_device_props.numEUsPerSubslice *
      ze_device_props.numThreadsPerEU;  //  10;

  hip_device_props.computeMode = hipComputeModeDefault;
  hip_device_props.arch = {};

  hip_device_props.arch.hasGlobalInt32Atomics = 1;
  hip_device_props.arch.hasSharedInt32Atomics = 1;

  hip_device_props.arch.hasGlobalInt64Atomics =
      (device_module_props.flags & ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS) ? 1 : 0;
  hip_device_props.arch.hasSharedInt64Atomics =
      (device_module_props.flags & ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS) ? 1 : 0;

  hip_device_props.arch.hasDoubles =
      (device_module_props.flags & ZE_DEVICE_MODULE_FLAG_FP64) ? 1 : 0;

  hip_device_props.clockInstructionRate = ze_device_props.coreClockRate;
  hip_device_props.concurrentKernels = 1;
  hip_device_props.pciDomainID = 0;
  hip_device_props.pciBusID = 0x10 + getDeviceId();
  hip_device_props.pciDeviceID = 0x40 + getDeviceId();
  hip_device_props.isMultiGpuBoard = 0;
  hip_device_props.canMapHostMemory = 1;
  hip_device_props.gcnArch = 0;
  hip_device_props.integrated =
      (ze_device_props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) ? 1 : 0;
  hip_device_props.maxSharedMemoryPerMultiProcessor =
      device_compute_props.maxSharedLocalMemory;
}

CHIPQueue* CHIPDeviceLevel0::addQueue_(unsigned int flags, int priority) {
  CHIPQueueLevel0* new_q = new CHIPQueueLevel0(this);
  chip_queues.push_back(new_q);
  return new_q;
}

CHIPTexture* CHIPDeviceLevel0::createTexture(
    const hipResourceDesc* pResDesc, const hipTextureDesc* pTexDesc,
    const struct hipResourceViewDesc* pResViewDesc) {
  ze_image_handle_t imageHandle;
  ze_sampler_handle_t samplerHandle;
  auto image =
      CHIPTextureLevel0::createImage(this, pResDesc, pTexDesc, pResViewDesc);
  auto sampler =
      CHIPTextureLevel0::createSampler(this, pResDesc, pTexDesc, pResViewDesc);

  CHIPTextureLevel0* chip_texture =
      new CHIPTextureLevel0((intptr_t)image, (intptr_t)sampler);

  auto q = (CHIPQueueLevel0*)getActiveQueue();
  // Check if need to copy data in
  if (pResDesc->res.array.array != nullptr) {
    hipArray* hipArr = pResDesc->res.array.array;
    q->memCopyToTexture(chip_texture, (unsigned char*)hipArr->data);
  }

  return chip_texture;
}

// Other
// ***********************************************************************
std::string resultToString(ze_result_t status) {
  switch (status) {
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
void CHIPModuleLevel0::compile(CHIPDevice* chip_dev) {
  logTrace("CHIPModuleLevel0.compile()");
  consumeSPIRV();
  ze_result_t status;

  // Create module with global address aware
  std::string compilerOptions = Backend->getJitFlags();
  ze_module_desc_t moduleDesc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                                 nullptr,
                                 ZE_MODULE_FORMAT_IL_SPIRV,
                                 ilSize,
                                 funcIL,
                                 compilerOptions.c_str(),
                                 nullptr};

  CHIPDeviceLevel0* chip_dev_lz = (CHIPDeviceLevel0*)chip_dev;
  CHIPContextLevel0* chip_ctx_lz = (CHIPContextLevel0*)(chip_dev->getContext());

  ze_device_handle_t ze_dev = ((CHIPDeviceLevel0*)chip_dev)->get();
  ze_context_handle_t ze_ctx = chip_ctx_lz->get();

  ze_module_build_log_handle_t log;
  auto build_status =
      zeModuleCreate(ze_ctx, ze_dev, &moduleDesc, &ze_module, &log);
  if (build_status != ZE_RESULT_SUCCESS) {
    size_t log_size;
    status = zeModuleBuildLogGetString(log, &log_size, nullptr);
    CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd);
    char log_str[log_size];
    status = zeModuleBuildLogGetString(log, &log_size, log_str);
    CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd);
    logTrace("ZE Build Log: {}", std::string(log_str).c_str());
  }
  CHIPERR_CHECK_LOG_AND_THROW(build_status, ZE_RESULT_SUCCESS, hipErrorTbd);
  logTrace("LZ CREATE MODULE via calling zeModuleCreate {} ",
           resultToString(build_status));
  // if (status == ZE_RESULT_ERROR_MODULE_BUILD_FAILURE) {
  //  CHIPERR_LOG_AND_THROW("Module failed to JIT: " + std::string(log_str),
  //                        hipErrorUnknown);
  //}

  uint32_t kernel_count = 0;
  status = zeModuleGetKernelNames(ze_module, &kernel_count, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd);
  logTrace("Found {} kernels in this module.", kernel_count);

  const char* kernel_names[kernel_count];
  status = zeModuleGetKernelNames(ze_module, &kernel_count, kernel_names);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd);
  for (auto& kernel : kernel_names) logTrace("Kernel {}", kernel);
  for (int i = 0; i < kernel_count; i++) {
    std::string host_f_name = kernel_names[i];
    logTrace("Registering kernel {}", host_f_name);
    int found_func_info = func_infos.count(host_f_name);
    if (found_func_info == 0) {
      // TODO: __syncthreads() gets turned into Intel_Symbol_Table_Void_Program
      // This is a call to OCML so it shouldn't be turned into a CHIPKernel
      continue;
      // CHIPERR_LOG_AND_THROW("Failed to find kernel in OpenCLFunctionInfoMap",
      //                      hipErrorInitializationError);
    }
    auto func_info = func_infos[host_f_name];
    // Create kernel
    ze_kernel_handle_t ze_kernel;
    ze_kernel_desc_t kernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr,
                                   0,  // flags
                                   host_f_name.c_str()};
    status = zeKernelCreate(ze_module, &kernelDesc, &ze_kernel);
    CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd);
    logTrace("LZ KERNEL CREATION via calling zeKernelCreate {} ", status);
    CHIPKernelLevel0* chip_ze_kernel =
        new CHIPKernelLevel0(ze_kernel, host_f_name, func_info);
    addKernel(chip_ze_kernel);
  }
}

void CHIPExecItem::setupAllArgs() {
  CHIPKernelLevel0* kernel = (CHIPKernelLevel0*)chip_kernel;

  OCLFuncInfo* FuncInfo = chip_kernel->getFuncInfo();

  size_t NumLocals = 0;
  int LastArgIdx = -1;

  for (size_t i = 0; i < FuncInfo->ArgTypeInfo.size(); ++i) {
    if (FuncInfo->ArgTypeInfo[i].space == OCLSpace::Local) {
      ++NumLocals;
    }
  }
  // there can only be one dynamic shared mem variable, per cuda spec
  assert(NumLocals <= 1);

  // Argument processing for the new HIP launch API.
  if (ArgsPointer) {
    for (size_t i = 0, argIdx = 0; i < FuncInfo->ArgTypeInfo.size();
         ++i, ++argIdx) {
      OCLArgTypeInfo& ai = FuncInfo->ArgTypeInfo[i];

      if (ai.type == OCLType::Image) {
        CHIPTextureLevel0* texObj =
            (CHIPTextureLevel0*)(*((unsigned long*)(ArgsPointer[1])));

        // Set image part
        logTrace("setImageArg {} size {}\n", argIdx, ai.size);
        ze_result_t status = zeKernelSetArgumentValue(
            kernel->get(), argIdx, ai.size, &(texObj->image));
        CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd);

        // Set sampler part
        argIdx++;

        logTrace("setImageArg {} size {}\n", argIdx, ai.size);
        status = zeKernelSetArgumentValue(kernel->get(), argIdx, ai.size,
                                          &(texObj->sampler));
        CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd);
      } else {
        logTrace("setArg {} size {} addr {}\n", argIdx, ai.size,
                 ArgsPointer[i]);
        ze_result_t status = zeKernelSetArgumentValue(kernel->get(), argIdx,
                                                      ai.size, ArgsPointer[i]);
        CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd,
                                    "zeKernelSetArgumentValue failed");
      }
    }
  } else {
    // Argument processing for the old HIP launch API.
    if ((offset_sizes.size() + NumLocals) != FuncInfo->ArgTypeInfo.size()) {
      CHIPERR_LOG_AND_THROW("Some arguments are still unset", hipErrorTbd);
    }

    if (offset_sizes.size() == 0) return;

    std::sort(offset_sizes.begin(), offset_sizes.end());
    if ((std::get<0>(offset_sizes[0]) != 0) ||
        (std::get<1>(offset_sizes[0]) == 0)) {
      CHIPERR_LOG_AND_THROW("Invalid offset/size", hipErrorTbd);
    }

    // check args are set
    if (offset_sizes.size() > 1) {
      for (size_t i = 1; i < offset_sizes.size(); ++i) {
        if ((std::get<0>(offset_sizes[i]) == 0) ||
            (std::get<1>(offset_sizes[i]) == 0) ||
            ((std::get<0>(offset_sizes[i - 1]) +
              std::get<1>(offset_sizes[i - 1])) >
             std::get<0>(offset_sizes[i]))) {
          CHIPERR_LOG_AND_THROW("Invalid offset/size", hipErrorTbd);
        }
      }
    }

    const unsigned char* start = arg_data.data();
    void* p;
    int err;
    for (size_t i = 0; i < offset_sizes.size(); ++i) {
      OCLArgTypeInfo& ai = FuncInfo->ArgTypeInfo[i];
      logTrace("ARG {}: OS[0]: {} OS[1]: {} \n      TYPE {} SPAC {} SIZE {}\n",
               i, std::get<0>(offset_sizes[i]), std::get<1>(offset_sizes[i]),
               (unsigned)ai.type, (unsigned)ai.space, ai.size);

      if (ai.type == OCLType::Pointer) {
        // TODO: sync with ExecItem's solution
        assert(ai.size == sizeof(void*));
        assert(std::get<1>(offset_sizes[i]) == ai.size);
        size_t size = std::get<1>(offset_sizes[i]);
        size_t offs = std::get<0>(offset_sizes[i]);
        const void* value = (void*)(start + offs);
        logTrace("setArg SVM {} to {}\n", i, p);
        ze_result_t status =
            zeKernelSetArgumentValue(kernel->get(), i, size, value);

        CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd,
                                    "zeKernelSetArgumentValue failed");

        logTrace(
            "LZ SET ARGUMENT VALUE via calling zeKernelSetArgumentValue "
            "{} ",
            status);
      } else {
        size_t size = std::get<1>(offset_sizes[i]);
        size_t offs = std::get<0>(offset_sizes[i]);
        const void* value = (void*)(start + offs);
        logTrace("setArg {} size {} offs {}\n", i, size, offs);
        ze_result_t status =
            zeKernelSetArgumentValue(kernel->get(), i, size, value);

        CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd,
                                    "zeKernelSetArgumentValue failed");

        logTrace(
            "LZ SET ARGUMENT VALUE via calling zeKernelSetArgumentValue "
            "{} ",
            status);
      }
    }
  }

  // Setup the kernel argument's value related to dynamically sized share
  // memory
  if (NumLocals == 1) {
    ze_result_t status = zeKernelSetArgumentValue(
        kernel->get(), FuncInfo->ArgTypeInfo.size() - 1, shared_mem, nullptr);
    logTrace(
        "LZ set dynamically sized share memory related argument via "
        "calling "
        "zeKernelSetArgumentValue {} ",
        status);
  }

  return;
}

ze_image_handle_t* CHIPTextureLevel0::createImage(
    CHIPDeviceLevel0* chip_dev, const hipResourceDesc* pResDesc,
    const hipTextureDesc* pTexDesc,
    const struct hipResourceViewDesc* pResViewDesc) {
  if (!pResDesc)
    CHIPERR_LOG_AND_THROW("Resource descriptor is null", hipErrorTbd);
  if (pResDesc->resType != hipResourceTypeArray) {
    CHIPERR_LOG_AND_THROW("only support hipArray as image storage",
                          hipErrorTbd);
  }

  hipArray* hipArr = pResDesc->res.array.array;
  if (!hipArr)
    CHIPERR_LOG_AND_THROW("hipResourceViewDesc result array is null",
                          hipErrorTbd);
  hipChannelFormatDesc channelDesc = hipArr->desc;

  ze_image_format_layout_t format_layout = ZE_IMAGE_FORMAT_LAYOUT_32;
  if (channelDesc.x == 8) {
    format_layout = ZE_IMAGE_FORMAT_LAYOUT_8;
  } else if (channelDesc.x == 16) {
    format_layout = ZE_IMAGE_FORMAT_LAYOUT_16;
  } else if (channelDesc.x == 32) {
    format_layout = ZE_IMAGE_FORMAT_LAYOUT_32;
  } else {
    CHIPERR_LOG_AND_THROW("hipChannelFormatDesc value is out of the scope",
                          hipErrorTbd);
  }

  ze_image_format_type_t format_type = ZE_IMAGE_FORMAT_TYPE_FLOAT;
  if (channelDesc.f == hipChannelFormatKindSigned) {
    format_type = ZE_IMAGE_FORMAT_TYPE_SINT;
  } else if (channelDesc.f == hipChannelFormatKindUnsigned) {
    format_type = ZE_IMAGE_FORMAT_TYPE_UINT;
  } else if (channelDesc.f == hipChannelFormatKindFloat) {
    format_type = ZE_IMAGE_FORMAT_TYPE_FLOAT;
  } else if (channelDesc.f == hipChannelFormatKindNone) {
    format_type = ZE_IMAGE_FORMAT_TYPE_FORCE_UINT32;
  } else {
    CHIPERR_LOG_AND_THROW("hipChannelFormatDesc value is out of the scope",
                          hipErrorTbd);
  }

  ze_image_format_t format = {format_layout,
                              format_type,
                              ZE_IMAGE_FORMAT_SWIZZLE_R,
                              ZE_IMAGE_FORMAT_SWIZZLE_0,
                              ZE_IMAGE_FORMAT_SWIZZLE_0,
                              ZE_IMAGE_FORMAT_SWIZZLE_1};

  ze_image_type_t image_type = ZE_IMAGE_TYPE_2D;

  ze_image_desc_t imageDesc = {ZE_STRUCTURE_TYPE_IMAGE_DESC, nullptr,
                               0,  // read-only
                               image_type, format,
                               // 128, 128, 0, 0, 0
                               hipArr->width, hipArr->height, 0, 0, 0};

  // Create LZ image handle
  CHIPContextLevel0* chip_ctx_lz = (CHIPContextLevel0*)chip_dev->getContext();
  ze_image_handle_t* image = new ze_image_handle_t();
  ze_result_t status =
      zeImageCreate(chip_ctx_lz->get(), chip_dev->get(), &imageDesc, image);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd);

  return image;
}

ze_sampler_handle_t* CHIPTextureLevel0::createSampler(
    CHIPDeviceLevel0* chip_dev, const hipResourceDesc* pResDesc,
    const hipTextureDesc* pTexDesc,
    const struct hipResourceViewDesc* pResViewDesc) {
  // Identify the address mode
  ze_sampler_address_mode_t addressMode = ZE_SAMPLER_ADDRESS_MODE_NONE;
  if (pTexDesc->addressMode[0] == hipAddressModeWrap)
    addressMode = ZE_SAMPLER_ADDRESS_MODE_NONE;
  else if (pTexDesc->addressMode[0] == hipAddressModeClamp)
    addressMode = ZE_SAMPLER_ADDRESS_MODE_CLAMP;
  else if (pTexDesc->addressMode[0] == hipAddressModeMirror)
    addressMode = ZE_SAMPLER_ADDRESS_MODE_MIRROR;
  else if (pTexDesc->addressMode[0] == hipAddressModeBorder)
    addressMode = ZE_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;

  // Identify the filter mode
  ze_sampler_filter_mode_t filterMode = ZE_SAMPLER_FILTER_MODE_NEAREST;
  if (pTexDesc->filterMode == hipFilterModePoint)
    filterMode = ZE_SAMPLER_FILTER_MODE_NEAREST;
  else if (pTexDesc->filterMode == hipFilterModeLinear)
    filterMode = ZE_SAMPLER_FILTER_MODE_LINEAR;

  // Identify the normalization
  ze_bool_t isNormalized = 0;
  if (pTexDesc->normalizedCoords == 0)
    isNormalized = 0;
  else
    isNormalized = 1;

  ze_sampler_desc_t samplerDesc = {ZE_STRUCTURE_TYPE_SAMPLER_DESC, nullptr,
                                   addressMode, filterMode, isNormalized};

  // Create LZ samler handle
  CHIPContextLevel0* chip_ctx_lz = (CHIPContextLevel0*)chip_dev->getContext();
  ze_sampler_handle_t* sampler = new ze_sampler_handle_t();
  ze_result_t status = zeSamplerCreate(chip_ctx_lz->get(), chip_dev->get(),
                                       &samplerDesc, sampler);
  CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd);

  return sampler;
}
