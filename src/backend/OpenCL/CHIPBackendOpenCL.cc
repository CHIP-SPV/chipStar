#include "CHIPBackendOpenCL.hh"
// CHIPCallbackDataLevel0
// ************************************************************************

CHIPCallbackDataOpenCL::CHIPCallbackDataOpenCL(hipStreamCallback_t CallbackF,
                                               void *CallbackArgs,
                                               CHIPQueue *ChipQueue)
    : ChipQueue((CHIPQueueOpenCL *)ChipQueue) {
  if (CallbackArgs != nullptr)
    CallbackArgs = CallbackArgs;
  if (CallbackF == nullptr)
    CHIPERR_LOG_AND_THROW("", hipErrorTbd);
  CallbackF = CallbackF;
}

// CHIPEventMonitorOpenCL
// ************************************************************************
CHIPEventMonitorOpenCL::CHIPEventMonitorOpenCL() : CHIPEventMonitor(){};

void CHIPEventMonitorOpenCL::monitor() {
  logTrace("CHIPEventMonitorOpenCL::monitor()");
  CHIPEventMonitor::monitor();
}

// CHIPDeviceOpenCL
// ************************************************************************

cl::Device *CHIPDeviceOpenCL::get() { return ClDevice; }
CHIPModuleOpenCL *CHIPDeviceOpenCL::addModule(std::string *ModuleStr) {
  CHIPModuleOpenCL *Module = new CHIPModuleOpenCL(ModuleStr);
  ChipModules.insert(std::make_pair(ModuleStr, Module));
  return Module;
}
CHIPTexture *
CHIPDeviceOpenCL::createTexture(const hipResourceDesc *ResDesc,
                                const hipTextureDesc *TexDesc,
                                const struct hipResourceViewDesc *ResViewDesc) {
  UNIMPLEMENTED(nullptr);
}
void CHIPDeviceOpenCL::destroyTexture(CHIPTexture *ChipTexture) {
  UNIMPLEMENTED();
}

CHIPDeviceOpenCL::CHIPDeviceOpenCL(CHIPContextOpenCL *ChipCtx,
                                   cl::Device *DevIn, int Idx)
    : CHIPDevice(ChipCtx), ClDevice(DevIn), ClContext(ChipCtx->get()) {
  logTrace("CHIPDeviceOpenCL initialized via OpenCL device pointer and context "
           "pointer");

  ChipCtx->addDevice(this);
}

void CHIPDeviceOpenCL::populateDevicePropertiesImpl() {
  logTrace("CHIPDeviceOpenCL->populate_device_properties()");
  cl_int Err;
  std::string Temp;

  assert(ClDevice != nullptr);
  Temp = ClDevice->getInfo<CL_DEVICE_NAME>();
  strncpy(HipDeviceProps_.name, Temp.c_str(), 255);
  HipDeviceProps_.name[255] = 0;

  HipDeviceProps_.totalGlobalMem =
      ClDevice->getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>(&Err);

  HipDeviceProps_.sharedMemPerBlock =
      ClDevice->getInfo<CL_DEVICE_LOCAL_MEM_SIZE>(&Err);

  HipDeviceProps_.maxThreadsPerBlock =
      ClDevice->getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&Err);

  std::vector<size_t> Wi = ClDevice->getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

  HipDeviceProps_.maxThreadsDim[0] = Wi[0];
  HipDeviceProps_.maxThreadsDim[1] = Wi[1];
  HipDeviceProps_.maxThreadsDim[2] = Wi[2];

  // Maximum configured clock frequency of the device in MHz.
  HipDeviceProps_.clockRate =
      1000 * ClDevice->getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();

  HipDeviceProps_.multiProcessorCount =
      ClDevice->getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  HipDeviceProps_.l2CacheSize =
      ClDevice->getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();

  // not actually correct
  HipDeviceProps_.totalConstMem =
      ClDevice->getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();

  // totally made up
  HipDeviceProps_.regsPerBlock = 64;

  // The minimum subgroup size on an intel GPU
  if (ClDevice->getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) {
    std::vector<uint> Sg = ClDevice->getInfo<CL_DEVICE_SUB_GROUP_SIZES_INTEL>();
    if (Sg.begin() != Sg.end())
      HipDeviceProps_.warpSize = *std::min_element(Sg.begin(), Sg.end());
  }
  HipDeviceProps_.maxGridSize[0] = HipDeviceProps_.maxGridSize[1] =
      HipDeviceProps_.maxGridSize[2] = 65536;
  HipDeviceProps_.memoryClockRate = 1000;
  HipDeviceProps_.memoryBusWidth = 256;
  HipDeviceProps_.major = 2;
  HipDeviceProps_.minor = 0;

  HipDeviceProps_.maxThreadsPerMultiProcessor = 10;

  HipDeviceProps_.computeMode = 0;
  HipDeviceProps_.arch = {};

  Temp = ClDevice->getInfo<CL_DEVICE_EXTENSIONS>();
  if (Temp.find("cl_khr_global_int32_base_atomics") != std::string::npos)
    HipDeviceProps_.arch.hasGlobalInt32Atomics = 1;
  else
    HipDeviceProps_.arch.hasGlobalInt32Atomics = 0;

  if (Temp.find("cl_khr_local_int32_base_atomics") != std::string::npos)
    HipDeviceProps_.arch.hasSharedInt32Atomics = 1;
  else
    HipDeviceProps_.arch.hasSharedInt32Atomics = 0;

  if (Temp.find("cl_khr_int64_base_atomics") != std::string::npos) {
    HipDeviceProps_.arch.hasGlobalInt64Atomics = 1;
    HipDeviceProps_.arch.hasSharedInt64Atomics = 1;
  } else {
    HipDeviceProps_.arch.hasGlobalInt64Atomics = 1;
    HipDeviceProps_.arch.hasSharedInt64Atomics = 1;
  }

  if (Temp.find("cl_khr_fp64") != std::string::npos)
    HipDeviceProps_.arch.hasDoubles = 1;
  else
    HipDeviceProps_.arch.hasDoubles = 0;

  HipDeviceProps_.clockInstructionRate = 2465;
  HipDeviceProps_.concurrentKernels = 1;
  HipDeviceProps_.pciDomainID = 0;
  HipDeviceProps_.pciBusID = 0x10;
  HipDeviceProps_.pciDeviceID = 0x40 + getDeviceId();
  HipDeviceProps_.isMultiGpuBoard = 0;
  HipDeviceProps_.canMapHostMemory = 1;
  HipDeviceProps_.gcnArch = 0;
  HipDeviceProps_.integrated = 0;
  HipDeviceProps_.maxSharedMemoryPerMultiProcessor = 0;
}

void CHIPDeviceOpenCL::reset() { UNIMPLEMENTED(); }
// CHIPEventOpenCL
// ************************************************************************

CHIPEventOpenCL::CHIPEventOpenCL(CHIPContextOpenCL *ChipContext,
                                 cl_event ClEvent, CHIPEventFlags Flags)
    : CHIPEvent((CHIPContext *)(ChipContext), Flags), ClEvent(ClEvent) {
  clRetainEvent(ClEvent);
}

CHIPEventOpenCL::CHIPEventOpenCL(CHIPContextOpenCL *ChipContext,
                                 CHIPEventFlags Flags)
    : CHIPEvent((CHIPContext *)(ChipContext), Flags), ClEvent(nullptr) {}

cl_event CHIPEventOpenCL::peek() { return ClEvent; }
cl_event CHIPEventOpenCL::get() {
  increaseRefCount();
  return ClEvent;
}

uint64_t CHIPEventOpenCL::getFinishTime() {
  int Status;
  uint64_t Ret;
  Status = clGetEventProfilingInfo(ClEvent, CL_PROFILING_COMMAND_END,
                                   sizeof(Ret), &Ret, NULL);

  if (Status != CL_SUCCESS) {
    int UpdatedStatus;
    auto Status = clGetEventInfo(ClEvent, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                 sizeof(int), &EventStatus_, NULL);
    CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
  }
  // CHIPERR_CHECK_LOG_AND_THROW(status, CL_SUCCESS, hipErrorTbd,
  //                             "Failed to query event for profiling info.");
  return Ret;
}

size_t *CHIPEventOpenCL::getRefCount() {
  cl_uint RefCount;
  int Status = ::clGetEventInfo(this->peek(), CL_EVENT_REFERENCE_COUNT, 4,
                                &RefCount, NULL);
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
  // logTrace("CHIPEventOpenCL::getRefCount() CHIP refc: {} OCL refc: {}",
  // refc,
  //         refcount);
  return Refc_;
}

CHIPEventOpenCL::~CHIPEventOpenCL() { ClEvent = nullptr; }
void CHIPEventOpenCL::decreaseRefCount() {
  logTrace("CHIPEventOpenCL::decreaseRefCount() msg={}", Msg.c_str());
  auto R = getRefCount();
  logTrace("CHIP Refc: {}->{} OpenCL Refc: {}->{}", *Refc_, *Refc_ - 1, *R,
           *R - 1);
  (*Refc_)--;
  clReleaseEvent(ClEvent);
  // Destructor to be called by event monitor once backend is done using it
  // if (*refc == 1) delete this;
}
void CHIPEventOpenCL::increaseRefCount() {
  logTrace("CHIPEventOpenCL::increaseRefCount() msg={}", Msg.c_str());
  auto R = getRefCount();
  logTrace("CHIP Refc: {}->{} OpenCL Refc: {}->{}", *Refc_, *Refc_ + 1, *R,
           *R + 1);
  (*Refc_)++;
  clRetainEvent(ClEvent);
}

CHIPEventOpenCL *CHIPBackendOpenCL::createCHIPEvent(CHIPContext *ChipCtx,
                                                    CHIPEventFlags Flags,
                                                    bool UserEvent) {
  return new CHIPEventOpenCL((CHIPContextOpenCL *)ChipCtx, Flags);
}

void CHIPEventOpenCL::takeOver(CHIPEvent *OtherIn) {
  if (*Refc_ > 1)
    decreaseRefCount();
  auto *Other = (CHIPEventOpenCL *)OtherIn;
  this->ClEvent = Other->get(); // increases refcount
  this->Refc_ = Other->getRefCount();
  this->Msg = Other->Msg;
}
// void CHIPEventOpenCL::recordStream(CHIPQueue *chip_queue_) {
//   logTrace("CHIPEventOpenCL::recordStream()");
//   /**
//    * each CHIPQueue keeps track of the status of the last enqueue command.
//    This
//    * is done by creating a CHIPEvent and associating it with the newly
//    submitted
//    * command. Each CHIPQueue has a LastEvent field.
//    *
//    * Recording is done by taking ownership of the target queues' LastEvent,
//    * incrementing that event's refcount.
//    */
//
//   auto chip_queue = (CHIPQueueOpenCL *)chip_queue_;
//   auto last_chip_event = (CHIPEventOpenCL *)chip_queue->getLastEvent();

//   // If this event was used previously, clear it
//   // can be >1 because recordEvent can be called >1 on the same event
//   bool fresh_event = true;
//   if (ev != nullptr) {
//     fresh_event = false;
//     decreaseRefCount();
//   }

//   // if no previous event, create a marker event - we always need 2 events to
//   // measure differences
//   assert(chip_queue->getLastEvent() != nullptr);

//   // Take over target queues event
//   this->ev = chip_queue->getLastEvent()->get();
//   this->refc = chip_queue->getLastEvent()->getRefCount();
//   this->msg = chip_queue->getLastEvent()->msg;
//   // if (fresh_event) assert(this->refc  3);

//   event_status = EVENT_STATUS_RECORDING;

//   /**
//    * There's nothing preventing you from calling hipRecordStream multiple
//    times
//    * in a row on the same event. In such case, after the first call, this
//    events
//    * clEvent field is no longer null and the event's refcount has been
//    * incremented.
//    *
//    * From HIP API: If hipEventRecord() has been previously called on this
//    * event, then this call will overwrite any existing state in event.
//    *
//    * hipEventCreate(myEvent); < clEvent is nullptr
//    * hipMemCopy(..., Q1)
//    * Q1.LastEvent = Q1_MemCopyEvent_0.refcount = 1
//    *
//    * hipStreamRecord(myEvent, Q1);
//    * clEvent== Q1_MemCopyEvent_0, refcount 1->2
//    *
//    * hipMemCopy(..., Q1)
//    * Q1.LastEvent = Q1_MemCopyEvent_1.refcount = 1
//    * Q1_MemCopyEvent_0.refcount 2->1
//    *
//    * hipStreamRecord(myEvent, Q1);
//    * Q1_MemCopyEvent_0.refcount 1->0
//    * clEvent==Q1_MemCopyEvent_1, refcount 1->2
//    */
// }

bool CHIPEventOpenCL::wait() {
  logTrace("CHIPEventOpenCL::wait()");

  if (EventStatus_ != EVENT_STATUS_RECORDING) {
    logWarn("Called wait() on an event that isn't active.");
    return false;
  }

  auto Status = clWaitForEvents(1, &ClEvent);

  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
  return true;
}

bool CHIPEventOpenCL::updateFinishStatus() {
  logTrace("CHIPEventOpenCL::updateFinishStatus()");
  if (EventStatus_ != EVENT_STATUS_RECORDING)
    return false;

  int UpdatedStatus;
  auto Status = clGetEventInfo(ClEvent, CL_EVENT_COMMAND_EXECUTION_STATUS,
                               sizeof(int), &UpdatedStatus, NULL);
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);

  if (UpdatedStatus <= CL_COMPLETE)
    EventStatus_ = EVENT_STATUS_RECORDED;

  return true;
}

float CHIPEventOpenCL::getElapsedTime(CHIPEvent *OtherIn) {
  // Why do I need to lock the context mutex?
  // Can I lock the mutex of this and the other event?
  //

  CHIPEventOpenCL *Other = (CHIPEventOpenCL *)OtherIn;

  if (this->getContext() != Other->getContext())
    CHIPERR_LOG_AND_THROW(
        "Attempted to get elapsed time between two events that are not part of "
        "the same context",
        hipErrorTbd);

  this->updateFinishStatus();
  Other->updateFinishStatus();

  if (!this->isRecordingOrRecorded() || !Other->isRecordingOrRecorded())
    CHIPERR_LOG_AND_THROW("one of the events isn't/hasn't recorded",
                          hipErrorTbd);

  if (!this->isFinished() || !Other->isFinished())
    CHIPERR_LOG_AND_THROW("one of the events hasn't finished",
                          hipErrorNotReady);

  uint64_t Started = this->getFinishTime();
  uint64_t Finished = Other->getFinishTime();

  logTrace("EventElapsedTime: STARTED {} / {} FINISHED {} / {} \n",
           (void *)this, Started, (void *)Other, Finished);

  // apparently fails for Intel NEO, god knows why
  // assert(Finished >= Started);
  uint64_t Elapsed;
  const uint64_t NANOSECS = 1000000000;
  if (Finished < Started) {
    logWarn("Finished < Started\n");
    Elapsed = Started - Finished;
  } else
    Elapsed = Finished - Started;
  uint64_t MS = (Elapsed / NANOSECS) * 1000;
  uint64_t NS = Elapsed % NANOSECS;
  float FractInMS = ((float)NS) / 1000000.0f;
  return (float)MS + FractInMS;
}

void CHIPEventOpenCL::hostSignal() { UNIMPLEMENTED(); }

// CHIPModuleOpenCL
//*************************************************************************

CHIPModuleOpenCL::CHIPModuleOpenCL(std::string *ModuleStr)
    : CHIPModule(ModuleStr){};

cl::Program &CHIPModuleOpenCL::get() { return Program_; }

void CHIPModuleOpenCL::compile(CHIPDevice *ChipDev) {

  // TODO make compile_ which calls consumeSPIRV()
  logTrace("CHIPModuleOpenCL::compile()");
  consumeSPIRV();
  CHIPDeviceOpenCL *ChipDevOcl = (CHIPDeviceOpenCL *)ChipDev;
  CHIPContextOpenCL *ChipCtxOcl =
      (CHIPContextOpenCL *)(ChipDevOcl->getContext());

  int Err;
  std::vector<char> BinaryVec(Src_.begin(), Src_.end());
  auto Program = cl::Program(*(ChipCtxOcl->get()), BinaryVec, false, &Err);
  CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorInitializationError);

  //   for (CHIPDevice *chip_dev : chip_devices) {
  std::string Name = ChipDevOcl->getName();
  Err = Program.build(Backend->getJitFlags().c_str());
  auto ErrBuild = Err;

  std::string Log =
      Program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*ChipDevOcl->ClDevice, &Err);
  logTrace("Program BUILD LOG for device #{}:{}:\n{}\n",
           ChipDevOcl->getDeviceId(), Name, Log);
  CHIPERR_CHECK_LOG_AND_THROW(ErrBuild, CL_SUCCESS,
                              hipErrorInitializationError);
  CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorInitializationError);

  logTrace("Program BUILD LOG for device #{}:{}:\n{}\n",
           ChipDevOcl->getDeviceId(), Name, Log);

  std::vector<cl::Kernel> Kernels;
  Err = Program.createKernels(&Kernels);
  CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorInitializationError);

  logTrace("Kernels in CHIPModuleOpenCL: {} \n", Kernels.size());
  for (int KernelIdx = 0; KernelIdx < Kernels.size(); KernelIdx++) {
    auto Kernel = Kernels[KernelIdx];
    std::string HostFName = Kernel.getInfo<CL_KERNEL_FUNCTION_NAME>(&Err);
    CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorInitializationError,
                                "Failed to fetch OpenCL kernel name");
    int FoundFuncInfo = FuncInfos_.count(HostFName);
    if (FoundFuncInfo == 0) {
      continue; // TODO
      // CHIPERR_LOG_AND_THROW("Failed to find kernel in OpenCLFunctionInfoMap",
      //                      hipErrorInitializationError);
    }
    auto FuncInfo = FuncInfos_[HostFName];
    CHIPKernelOpenCL *ChipKernel =
        new CHIPKernelOpenCL(std::move(Kernel), HostFName, FuncInfo);
    addKernel(ChipKernel);
  }
}

CHIPQueue *CHIPDeviceOpenCL::addQueueImpl(unsigned int Flags, int Priority) {
  CHIPQueueOpenCL *NewQ = new CHIPQueueOpenCL(this);
  ChipQueues_.push_back(NewQ);
  return NewQ;
}

// CHIPKernelOpenCL
//*************************************************************************

OCLFuncInfo *CHIPKernelOpenCL::getFuncInfo() const { return FuncInfo_; }
std::string CHIPKernelOpenCL::getName() { return Name_; }
cl::Kernel CHIPKernelOpenCL::get() const { return OclKernel_; }
size_t CHIPKernelOpenCL::getTotalArgSize() const { return TotalArgSize_; };

CHIPKernelOpenCL::CHIPKernelOpenCL(const cl::Kernel &&ClKernel,
                                   std::string HostFName, OCLFuncInfo *FuncInfo)
    : CHIPKernel(HostFName, FuncInfo) /*, ocl_kernel(cl_kernel_)*/ {
  OclKernel_ = ClKernel;
  int Err = 0;
  // TODO attributes
  cl_uint NumArgs = OclKernel_.getInfo<CL_KERNEL_NUM_ARGS>(&Err);
  CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorTbd,
                              "Failed to get num args for kernel");
  assert(FuncInfo_->ArgTypeInfo.size() == NumArgs);

  if (NumArgs > 0) {
    logTrace("Kernel {} numArgs: {} \n", Name_, NumArgs);
    logTrace("  RET_TYPE: {} {} {}\n", FuncInfo_->RetTypeInfo.Size,
             (unsigned)FuncInfo_->RetTypeInfo.Space,
             (unsigned)FuncInfo_->RetTypeInfo.Type);
    for (auto &Argty : FuncInfo_->ArgTypeInfo) {
      logTrace("  ARG: SIZE {} SPACE {} TYPE {}\n", Argty.Size,
               (unsigned)Argty.Space, (unsigned)Argty.Type);
      TotalArgSize_ += Argty.Size;
    }
  }
}

// CHIPContextOpenCL
//*************************************************************************

void CHIPContextOpenCL::freeImpl(void *Ptr) { UNIMPLEMENTED(); };

cl::Context *CHIPContextOpenCL::get() { return ClContext; }
CHIPContextOpenCL::CHIPContextOpenCL(cl::Context *CtxIn) {
  logTrace("CHIPContextOpenCL Initialized via OpenCL Context pointer.");
  ClContext = CtxIn;
}

void *CHIPContextOpenCL::allocateImpl(size_t Size, size_t Alignment,
                                      CHIPMemoryType MemType) {
  void *Retval;

  Retval = SvmMemory.allocate(*ClContext, Size);
  return Retval;
}

// CHIPQueueOpenCL
//*************************************************************************
struct HipStreamCallbackData {
  hipStream_t Stream;
  hipError_t Status;
  void *UserData;
  hipStreamCallback_t Callback;
};

void CL_CALLBACK pfn_notify(cl_event Event, cl_int CommandExecStatus,
                            void *UserData) {
  HipStreamCallbackData *Cbo = (HipStreamCallbackData *)(UserData);
  if (Cbo == nullptr)
    return;
  if (Cbo->Callback == nullptr)
    return;
  Cbo->Callback(Cbo->Stream, Cbo->Status, Cbo->UserData);
  delete Cbo;
}

cl::CommandQueue *CHIPQueueOpenCL::get() { return ClQueue_; }

bool CHIPQueueOpenCL::addCallback(hipStreamCallback_t Callback,
                                  void *UserData) {
  logTrace("CHIPQueueOpenCL::addCallback()");
  auto Ev = getLastEvent();
  if (Ev == nullptr) {
    Callback(this, hipSuccess, UserData);
    return true;
  }

  HipStreamCallbackData *Cb =
      new HipStreamCallbackData{this, hipSuccess, UserData, Callback};

  auto Status = clSetEventCallback(Ev->peek(), CL_COMPLETE, pfn_notify, Cb);
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);

  // enqueue barrier with no dependencies (all further enqueues will wait for
  // this one to finish)

  enqueueBarrier(nullptr);
  return true;
};

CHIPEvent *CHIPQueueOpenCL::enqueueMarkerImpl() {
  cl::Event MarkerEvent;
  auto Status = this->get()->enqueueMarkerWithWaitList(nullptr, &MarkerEvent);
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
  CHIPEventOpenCL *CHIPMarkerEvent =
      new CHIPEventOpenCL((CHIPContextOpenCL *)ChipContext_, MarkerEvent.get());
  CHIPMarkerEvent->Msg = "marker";
  return CHIPMarkerEvent;
}

CHIPEventOpenCL *CHIPQueueOpenCL::getLastEvent() {
  return (CHIPEventOpenCL *)LastEvent_;
}

CHIPEvent *CHIPQueueOpenCL::launchImpl(CHIPExecItem *ExecItem) {
  //
  logTrace("CHIPQueueOpenCL->launch()");
  CHIPExecItemOpenCL *ChipOclExecItem = (CHIPExecItemOpenCL *)ExecItem;
  CHIPKernelOpenCL *Kernel = (CHIPKernelOpenCL *)ChipOclExecItem->getKernel();
  assert(Kernel != nullptr);
  logTrace("Launching Kernel {}", Kernel->getName());

  ChipOclExecItem->setupAllArgs(Kernel);

  dim3 GridDim = ChipOclExecItem->getGrid();
  dim3 BlockDim = ChipOclExecItem->getBlock();

  const cl::NDRange Global(GridDim.x * BlockDim.x, GridDim.y * BlockDim.y,
                           GridDim.z * BlockDim.z);
  const cl::NDRange Local(BlockDim.x, BlockDim.y, BlockDim.z);

  cl::Event Ev;
  int Err = ClQueue_->enqueueNDRangeKernel(Kernel->get(), cl::NullRange, Global,
                                           Local, nullptr, &Ev);

  CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorTbd);
  hipError_t Retval = hipSuccess;

  CHIPEventOpenCL *E =
      new CHIPEventOpenCL((CHIPContextOpenCL *)ChipContext_, Ev.get());
  // clRetainEvent(ev.get());
  // updateLastEvent(e);
  return E;
}

CHIPQueueOpenCL::CHIPQueueOpenCL(CHIPDevice *ChipDevice)
    : CHIPQueue(ChipDevice) {
  ClContext_ = ((CHIPContextOpenCL *)ChipContext_)->get();
  ClDevice_ = ((CHIPDeviceOpenCL *)ChipDevice_)->get();

  cl_int Status;
  ClQueue_ = new cl::CommandQueue(*ClContext_, *ClDevice_,
                                  CL_QUEUE_PROFILING_ENABLE, &Status);
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorInitializationError);
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
  ChipDevice->addQueue(this);
}

CHIPQueueOpenCL::~CHIPQueueOpenCL() {
  delete ClContext_;
  delete ClDevice_;
}

CHIPEvent *CHIPQueueOpenCL::memCopyAsyncImpl(void *Dst, const void *Src,
                                             size_t Size) {
  logTrace("clSVMmemcpy {} -> {} / {} B\n", Src, Dst, Size);
  cl_event Ev = nullptr;
  int Retval = ::clEnqueueSVMMemcpy(ClQueue_->get(), CL_FALSE, Dst, Src, Size,
                                    0, nullptr, &Ev);
  CHIPERR_CHECK_LOG_AND_THROW(Retval, CL_SUCCESS, hipErrorRuntimeMemory);
  CHIPEventOpenCL *E =
      new CHIPEventOpenCL((CHIPContextOpenCL *)ChipContext_, Ev);
  return E;
}

void CHIPQueueOpenCL::finish() {
  auto Status = ClQueue_->finish();
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
}

CHIPEvent *CHIPQueueOpenCL::memFillAsyncImpl(void *Dst, size_t Size,
                                             const void *Pattern,
                                             size_t PatternSize) {
  logTrace("clSVMmemfill {} / {} B\n", Dst, Size);
  cl_event Ev = nullptr;
  int Retval = ::clEnqueueSVMMemFill(ClQueue_->get(), Dst, Pattern, PatternSize,
                                     Size, 0, nullptr, &Ev);
  CHIPERR_CHECK_LOG_AND_THROW(Retval, CL_SUCCESS, hipErrorRuntimeMemory);
  CHIPEventOpenCL *E =
      new CHIPEventOpenCL((CHIPContextOpenCL *)ChipContext_, Ev);
  return E;
};

CHIPEvent *CHIPQueueOpenCL::memCopy2DAsyncImpl(void *Dst, size_t Dpitch,
                                               const void *Src, size_t Spitch,
                                               size_t Width, size_t Height) {
  UNIMPLEMENTED(nullptr);
};

CHIPEvent *CHIPQueueOpenCL::memCopy3DAsyncImpl(void *Dst, size_t Dpitch,
                                               size_t Dspitch, const void *Src,
                                               size_t Spitch, size_t Sspitch,
                                               size_t Width, size_t Height,
                                               size_t Depth) {
  UNIMPLEMENTED(nullptr);
};

// Memory copy to texture object, i.e. image
CHIPEvent *CHIPQueueOpenCL::memCopyToTextureImpl(CHIPTexture *TexObj,
                                                 void *Src) {
  UNIMPLEMENTED(nullptr);
};

void CHIPQueueOpenCL::getBackendHandles(unsigned long *NativeInfo, int *Size) {
  UNIMPLEMENTED();
}
CHIPEvent *CHIPQueueOpenCL::memPrefetchImpl(const void *Ptr, size_t Count) {
  UNIMPLEMENTED(nullptr);
}

CHIPEvent *
CHIPQueueOpenCL::enqueueBarrierImpl(std::vector<CHIPEvent *> *EventsToWaitFor) {
  cl::Event MarkerEvent;
  int status = ClQueue_->enqueueMarkerWithWaitList(nullptr, &MarkerEvent);
  CHIPERR_CHECK_LOG_AND_THROW(status, CL_SUCCESS, hipErrorTbd);

  cl::vector<cl::Event> Events = {};
  // if (EventsToWaitFor)
  //   for (auto E : *EventsToWaitFor) {
  //     auto Ee = (CHIPEventOpenCL *)E;
  //     Events.push_back(cl::Event(Ee->peek()));
  //   }
  Events.push_back(MarkerEvent);

  cl::Event Barrier;
  auto Status = ClQueue_->enqueueBarrierWithWaitList(&Events, &Barrier);
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);

  CHIPEventOpenCL *NewEvent =
      new CHIPEventOpenCL((CHIPContextOpenCL *)ChipContext_, Barrier.get());

  return NewEvent;
}

static int setLocalSize(size_t Shared, OCLFuncInfo *FuncInfo,
                        cl_kernel Kernel) {
  logTrace("setLocalSize");
  int Err = CL_SUCCESS;

  if (Shared > 0) {
    logTrace("setLocalMemSize to {}\n", Shared);
    size_t LastArgIdx = FuncInfo->ArgTypeInfo.size() - 1;
    if (FuncInfo->ArgTypeInfo[LastArgIdx].Space != OCLSpace::Local) {
      // this can happen if for example the llvm optimizes away
      // the dynamic local variable
      logWarn("Can't set the dynamic local size, "
              "because the kernel doesn't use any local memory.\n");
    } else {
      Err = ::clSetKernelArg(Kernel, LastArgIdx, Shared, nullptr);
      CHIPERR_CHECK_LOG_AND_THROW(
          Err, CL_SUCCESS, hipErrorTbd,
          "clSetKernelArg() failed to set dynamic local size");
    }
  }

  return Err;
}

// CHIPExecItemOpenCL
//*************************************************************************

cl::Kernel *CHIPExecItemOpenCL::get() { return ClKernel_; }

int CHIPExecItemOpenCL::setupAllArgs(CHIPKernelOpenCL *Kernel) {
  OCLFuncInfo *FuncInfo = Kernel->getFuncInfo();
  size_t NumLocals = 0;
  for (size_t i = 0; i < FuncInfo->ArgTypeInfo.size(); ++i) {
    if (FuncInfo->ArgTypeInfo[i].Space == OCLSpace::Local)
      ++NumLocals;
  }
  // there can only be one dynamic shared mem variable, per cuda spec
  assert(NumLocals <= 1);
  int Err = 0;

  if (ArgsPointer_) {
    logTrace("Setting up arguments NEW HIP API");
    for (cl_uint i = 0; i < FuncInfo->ArgTypeInfo.size(); ++i) {
      OCLArgTypeInfo &Ai = FuncInfo->ArgTypeInfo[i];
      if (Ai.Type == OCLType::Pointer && Ai.Space != OCLSpace::Local) {
        logTrace("clSetKernelArgSVMPointer {} SIZE {} to {}\n", i, Ai.Size,
                 ArgsPointer_[i]);
        assert(Ai.Size == sizeof(void *));
        const void *Argval = *(void **)ArgsPointer_[i];
        Err = ::clSetKernelArgSVMPointer(Kernel->get().get(), i, Argval);

        CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorTbd,
                                    "clSetKernelArgSVMPointer failed");
      } else {
        logTrace("clSetKernelArg {} SIZE {} to {}\n", i, Ai.Size,
                 ArgsPointer_[i]);
        Err =
            ::clSetKernelArg(Kernel->get().get(), i, Ai.Size, ArgsPointer_[i]);
        CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorTbd,
                                    "clSetKernelArg failed");
      }
    }
  } else {
    logTrace("Setting up arguments OLD HIP API");

    if ((OffsetSizes_.size() + NumLocals) != FuncInfo->ArgTypeInfo.size()) {
      CHIPERR_LOG_AND_THROW("Some arguments are still unset", hipErrorTbd);
    }

    if (OffsetSizes_.size() == 0)
      return CL_SUCCESS;

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
    void *P;
    int Err;
    for (cl_uint i = 0; i < OffsetSizes_.size(); ++i) {
      OCLArgTypeInfo &Ai = FuncInfo->ArgTypeInfo[i];
      logTrace("ARG {}: OS[0]: {} OS[1]: {} \n      TYPE {} SPAC {} SIZE {}\n",
               i, std::get<0>(OffsetSizes_[i]), std::get<1>(OffsetSizes_[i]),
               (unsigned)Ai.Type, (unsigned)Ai.Space, Ai.Size);

      if (Ai.Type == OCLType::Pointer) {
        // TODO other than global AS ?
        assert(Ai.Size == sizeof(void *));
        assert(std::get<1>(OffsetSizes_[i]) == Ai.Size);
        P = *(void **)(Start + std::get<0>(OffsetSizes_[i]));
        logTrace("setArg SVM {} to {}\n", i, P);
        Err = ::clSetKernelArgSVMPointer(Kernel->get().get(), i, P);
        CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorTbd,
                                    "clSetKernelArgSVMPointer failed");
      } else {
        size_t Size = std::get<1>(OffsetSizes_[i]);
        size_t Offs = std::get<0>(OffsetSizes_[i]);
        void *Value = (void *)(Start + Offs);
        logTrace("setArg {} size {} offs {}\n", i, Size, Offs);
        Err = ::clSetKernelArg(Kernel->get().get(), i, Size, Value);
        CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorTbd,
                                    "clSetKernelArg failed");
      }
    }
  }

  return setLocalSize(SharedMem_, FuncInfo, Kernel->get().get());
}
// CHIPBackendOpenCL
//*************************************************************************

CHIPTexture *CHIPBackendOpenCL::createCHIPTexture(intptr_t Image,
                                                  intptr_t Sampler) {
  UNIMPLEMENTED(nullptr);
  // return new CHIPTextureOpenCL();
}

CHIPQueue *CHIPBackendOpenCL::createCHIPQueue(CHIPDevice *ChipDev) {
  CHIPDeviceOpenCL *ChipDevCl = (CHIPDeviceOpenCL *)ChipDev;
  return new CHIPQueueOpenCL(ChipDevCl);
}

CHIPCallbackData *
CHIPBackendOpenCL::createCallbackData(hipStreamCallback_t Callback,
                                      void *UserData, CHIPQueue *ChipQueue) {
  UNIMPLEMENTED(nullptr);
}

CHIPEventMonitor *CHIPBackendOpenCL::createCallbackEventMonitor() {
  auto Evm = new CHIPEventMonitorOpenCL();
  Evm->start();
  return Evm;
}

CHIPEventMonitor *CHIPBackendOpenCL::createStaleEventMonitor() {
  UNIMPLEMENTED(nullptr);
}

std::string CHIPBackendOpenCL::getDefaultJitFlags() {
  return std::string("-x spir -cl-kernel-arg-info");
}

void CHIPBackendOpenCL::initializeImpl(std::string CHIPPlatformStr,
                                       std::string CHIPDeviceTypeStr,
                                       std::string CHIPDeviceStr) {
  logTrace("CHIPBackendOpenCL Initialize");

  // transform device type string into CL
  cl_bitfield SelectedDevType = 0;
  if (CHIPDeviceTypeStr == "all")
    SelectedDevType = CL_DEVICE_TYPE_ALL;
  else if (CHIPDeviceTypeStr == "cpu")
    SelectedDevType = CL_DEVICE_TYPE_CPU;
  else if (CHIPDeviceTypeStr == "gpu")
    SelectedDevType = CL_DEVICE_TYPE_GPU;
  else if (CHIPDeviceTypeStr == "default")
    SelectedDevType = CL_DEVICE_TYPE_DEFAULT;
  else if (CHIPDeviceTypeStr == "accel")
    SelectedDevType = CL_DEVICE_TYPE_ACCELERATOR;
  else
    throw InvalidDeviceType("Unknown value provided for CHIP_DEVICE_TYPE\n");
  std::cout << "Using Devices of type " << CHIPDeviceTypeStr << "\n";

  std::vector<cl::Platform> Platforms;
  cl_int Err = cl::Platform::get(&Platforms);
  CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorInitializationError);
  std::cout << "\nFound " << Platforms.size() << " OpenCL platforms:\n";
  for (int i = 0; i < Platforms.size(); i++) {
    std::cout << i << ". " << Platforms[i].getInfo<CL_PLATFORM_NAME>() << "\n";
  }

  std::cout << "OpenCL Devices of type " << CHIPDeviceTypeStr
            << " with SPIR-V_1 support:\n";
  std::vector<cl::Device> Devices;
  for (auto Plat : Platforms) {
    std::vector<cl::Device> Dev;
    Err = Plat.getDevices(SelectedDevType, &Dev);
    CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorInitializationError);
    for (auto D : Dev) {
      std::string Ver = D.getInfo<CL_DEVICE_IL_VERSION>(&Err);
      if ((Err == CL_SUCCESS) && (Ver.rfind("SPIR-V_1.", 0) == 0)) {
        std::cout << D.getInfo<CL_DEVICE_NAME>() << "\n";
        Devices.push_back(D);
      }
    }
  }

  // Create context which has devices
  // Create queues that have devices each of which has an associated context
  // TODO Change this to spirv_enabled_devices
  cl::Context *Ctx = new cl::Context(Devices);
  CHIPContextOpenCL *ChipContext = new CHIPContextOpenCL(Ctx);
  Backend->addContext(ChipContext);
  for (int i = 0; i < Devices.size(); i++) {
    cl::Device *Dev = new cl::Device(Devices[i]);
    CHIPDeviceOpenCL *ChipDev = new CHIPDeviceOpenCL(ChipContext, Dev, i);
    logTrace("CHIPDeviceOpenCL {}",
             ChipDev->ClDevice->getInfo<CL_DEVICE_NAME>());
    ChipDev->populateDeviceProperties();
    Backend->addDevice(ChipDev);
    CHIPQueueOpenCL *Queue = new CHIPQueueOpenCL(ChipDev);
    // chip_dev->addQueue(queue);
    ChipContext->addQueue(Queue);
    Backend->addQueue(Queue);
  }
  std::cout << "OpenCL Context Initialized.\n";
};

// Other
//*************************************************************************

std::string resultToString(int Status) {
  switch (Status) {
  case CL_SUCCESS:
    return "CL_SUCCESS";
  case CL_DEVICE_NOT_FOUND:
    return "CL_DEVICE_NOT_FOUND";
  case CL_DEVICE_NOT_AVAILABLE:
    return "CL_DEVICE_NOT_AVAILABLE";
  case CL_COMPILER_NOT_AVAILABLE:
    return "CL_COMPILER_NOT_AVAILABLE";
  case CL_MEM_OBJECT_ALLOCATION_FAILURE:
    return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case CL_OUT_OF_RESOURCES:
    return "CL_OUT_OF_RESOURCES";
  case CL_OUT_OF_HOST_MEMORY:
    return "CL_OUT_OF_HOST_MEMORY";
  case CL_PROFILING_INFO_NOT_AVAILABLE:
    return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case CL_MEM_COPY_OVERLAP:
    return "CL_MEM_COPY_OVERLAP";
  case CL_IMAGE_FORMAT_MISMATCH:
    return "CL_IMAGE_FORMAT_MISMATCH";
  case CL_IMAGE_FORMAT_NOT_SUPPORTED:
    return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case CL_BUILD_PROGRAM_FAILURE:
    return "CL_BUILD_PROGRAM_FAILURE";
  case CL_MAP_FAILURE:
    return "CL_MAP_FAILURE";
#ifdef CL_VERSION_1_1
  case CL_MISALIGNED_SUB_BUFFER_OFFSET:
    return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
#endif
#ifdef CL_VERSION_1_2
  case CL_COMPILE_PROGRAM_FAILURE:
    return "CL_COMPILE_PROGRAM_FAILURE";
  case CL_LINKER_NOT_AVAILABLE:
    return "CL_LINKER_NOT_AVAILABLE";
  case CL_LINK_PROGRAM_FAILURE:
    return "CL_LINK_PROGRAM_FAILURE";
  case CL_DEVICE_PARTITION_FAILED:
    return "CL_DEVICE_PARTITION_FAILED";
  case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
    return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
#endif
  case (CL_INVALID_VALUE):
    return "CL_INVALID_VALUE";
  case (CL_INVALID_DEVICE_TYPE):
    return "CL_INVALID_DEVICE_TYPE";
  case (CL_INVALID_PLATFORM):
    return "CL_INVALID_PLATFORM";
  case (CL_INVALID_DEVICE):
    return "CL_INVALID_DEVICE";
  case (CL_INVALID_CONTEXT):
    return "CL_INVALID_CONTEXT";
  case (CL_INVALID_QUEUE_PROPERTIES):
    return "CL_INVALID_QUEUE_PROPERTIES";
  case (CL_INVALID_COMMAND_QUEUE):
    return "CL_INVALID_COMMAND_QUEUE";
  case (CL_INVALID_HOST_PTR):
    return "CL_INVALID_HOST_PTR";
  case (CL_INVALID_MEM_OBJECT):
    return "CL_INVALID_MEM_OBJECT";
  case (CL_INVALID_IMAGE_FORMAT_DESCRIPTOR):
    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case (CL_INVALID_IMAGE_SIZE):
    return "CL_INVALID_IMAGE_SIZE";
  case (CL_INVALID_SAMPLER):
    return "CL_INVALID_SAMPLER";
  case (CL_INVALID_BINARY):
    return "CL_INVALID_BINARY";
  case (CL_INVALID_BUILD_OPTIONS):
    return "CL_INVALID_BUILD_OPTIONS";
  case (CL_INVALID_PROGRAM):
    return "CL_INVALID_PROGRAM";
  case (CL_INVALID_PROGRAM_EXECUTABLE):
    return "CL_INVALID_PROGRAM_EXECUTABLE";
  case (CL_INVALID_KERNEL_NAME):
    return "CL_INVALID_KERNEL_NAME";
  case (CL_INVALID_KERNEL_DEFINITION):
    return "CL_INVALID_KERNEL_DEFINITION";
  case (CL_INVALID_KERNEL):
    return "CL_INVALID_KERNEL";
  case (CL_INVALID_ARG_INDEX):
    return "CL_INVALID_ARG_INDEX";
  case (CL_INVALID_ARG_VALUE):
    return "CL_INVALID_ARG_VALUE";
  case (CL_INVALID_ARG_SIZE):
    return "CL_INVALID_ARG_SIZE";
  case (CL_INVALID_KERNEL_ARGS):
    return "CL_INVALID_KERNEL_ARGS";
  case (CL_INVALID_WORK_DIMENSION):
    return "CL_INVALID_WORK_DIMENSION";
  case (CL_INVALID_WORK_GROUP_SIZE):
    return "CL_INVALID_WORK_GROUP_SIZE";
  case (CL_INVALID_WORK_ITEM_SIZE):
    return "CL_INVALID_WORK_ITEM_SIZE";
  case (CL_INVALID_GLOBAL_OFFSET):
    return "CL_INVALID_GLOBAL_OFFSET";
  case (CL_INVALID_EVENT_WAIT_LIST):
    return "CL_INVALID_EVENT_WAIT_LIST";
  case (CL_INVALID_EVENT):
    return "CL_INVALID_EVENT";
  case (CL_INVALID_OPERATION):
    return "CL_INVALID_OPERATION";
  case (CL_INVALID_GL_OBJECT):
    return "CL_INVALID_GL_OBJECT";
  case (CL_INVALID_BUFFER_SIZE):
    return "CL_INVALID_BUFFER_SIZE";
  case (CL_INVALID_MIP_LEVEL):
    return "CL_INVALID_MIP_LEVEL";
  case (CL_INVALID_GLOBAL_WORK_SIZE):
    return "CL_INVALID_GLOBAL_WORK_SIZE";
#ifdef CL_VERSION_1_1
  case (CL_INVALID_PROPERTY):
    return "CL_INVALID_PROPERTY";
#endif
#ifdef CL_VERSION_1_2
  case (CL_INVALID_IMAGE_DESCRIPTOR):
    return "CL_INVALID_IMAGE_DESCRIPTOR";
  case (CL_INVALID_COMPILER_OPTIONS):
    return "CL_INVALID_COMPILER_OPTIONS";
  case (CL_INVALID_LINKER_OPTIONS):
    return "CL_INVALID_LINKER_OPTIONS";
  case (CL_INVALID_DEVICE_PARTITION_COUNT):
    return "CL_INVALID_DEVICE_PARTITION_COUNT";
#endif
#ifdef CL_VERSION_2_0
  case (CL_INVALID_PIPE_SIZE):
    return "CL_INVALID_PIPE_SIZE";
  case (CL_INVALID_DEVICE_QUEUE):
    return "CL_INVALID_DEVICE_QUEUE";
#endif
#ifdef CL_VERSION_2_2
  case (CL_INVALID_SPEC_ID):
    return "CL_INVALID_SPEC_ID";
  case (CL_MAX_SIZE_RESTRICTION_EXCEEDED):
    return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
#endif
  default:
    return "CL_UNKNOWN_ERROR";
  }
}
