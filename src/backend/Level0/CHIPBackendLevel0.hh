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

#ifndef CHIP_BACKEND_LEVEL0_H
#define CHIP_BACKEND_LEVEL0_H

#define L0_DEFAULT_QUEUE_PRIORITY ZE_COMMAND_QUEUE_PRIORITY_NORMAL

#include "../../CHIPBackend.hh"
#include "ze_api.h"
#include "../src/common.hh"

std::string resultToString(ze_result_t Status);

// fw declares
class CHIPBackendLevel0;
class CHIPContextLevel0;
class CHIPDeviceLevel0;
class CHIPModuleLevel0;
class CHIPTextureLevel0;
class CHIPEventLevel0;
class CHIPQueueLevel0;
class LZCommandList;
class LZEventPool;
class CHIPExecItemLevel0;
class CHIPKernelLevel0;
class EventMonitorLevel0;

class CHIPExecItemLevel0 : public chipstar::ExecItem {
  CHIPKernelLevel0 *ChipKernel_ = nullptr;

public:
  CHIPExecItemLevel0(const CHIPExecItemLevel0 &Other)
      : CHIPExecItemLevel0(Other.GridDim_, Other.BlockDim_, Other.SharedMem_,
                           Other.ChipQueue_) {
    ChipKernel_ = Other.ChipKernel_;
    this->ArgsSetup = Other.ArgsSetup;
    this->Args_ = Other.Args_;
  }

  CHIPExecItemLevel0(dim3 GirdDim, dim3 BlockDim, size_t SharedMem,
                     hipStream_t ChipQueue)
      : ExecItem(GirdDim, BlockDim, SharedMem, ChipQueue) {}

  virtual ~CHIPExecItemLevel0() override {}

  virtual void setupAllArgs() override;
  virtual chipstar::ExecItem *clone() const override {
    auto NewExecItem = new CHIPExecItemLevel0(*this);
    return NewExecItem;
  }

  void setKernel(chipstar::Kernel *Kernel) override;
  chipstar::Kernel *getKernel() override;
};

class CHIPEventLevel0 : public chipstar::Event {
public:
  using ActionFn = std::function<void()>;

private:
  // Used for resolving device counter overflow
  uint64_t HostTimestamp_ = 0, DeviceTimestamp_ = 0;
  friend class CHIPEventLevel0;
  // The handler of event_pool and event
  ze_event_handle_t Event_;
  ze_event_pool_handle_t EventPoolHandle_;

  // The timestamp value
  uint64_t Timestamp_;

  std::vector<ActionFn> Actions_;

public:
  uint32_t getValidTimestampBits();
  uint64_t getHostTimestamp() { return HostTimestamp_; }
  unsigned int EventPoolIndex;
  LZEventPool *EventPool;
  CHIPEventLevel0()
      : CHIPEventLevel0((CHIPContextLevel0 *)Backend->getActiveContext()) {}
  CHIPEventLevel0(CHIPContextLevel0 *ChipCtx,
                  chipstar::EventFlags Flags = chipstar::EventFlags());
  CHIPEventLevel0(CHIPContextLevel0 *ChipCtx, ze_event_handle_t NativeEvent);
  CHIPEventLevel0(CHIPContextLevel0 *ChipCtx, LZEventPool *EventPool,
                  unsigned int PoolIndex, chipstar::EventFlags Flags);
  virtual ~CHIPEventLevel0() override;

  void recordStream(chipstar::Queue *ChipQueue) override;

  virtual bool wait() override;

  virtual bool updateFinishStatus(bool ThrowErrorIfNotReady = true) override;

  unsigned long getFinishTime();

  virtual float getElapsedTime(chipstar::Event *Other) override;

  virtual void hostSignal() override;

  void reset();

  ze_event_handle_t peek();

  /// Bind an action which is promised to be executed when the event is
  /// finished.
  void addAction(ActionFn Action) { Actions_.emplace_back(Action); }

  /// Execute the actions. The event must be finished.
  void doActions() {
    assert(isFinished() && "chipstar::Event must be finished first!");
    for (auto &Action : Actions_)
      Action();
    Actions_.clear();
  }
};

class CHIPCallbackDataLevel0 : public chipstar::CallbackData {
private:
  // ze_event_pool_handle_t ZeEventPool_;

public:
  std::mutex CallbackDataMtx;
  CHIPCallbackDataLevel0(hipStreamCallback_t CallbackF, void *CallbackArgs,
                         chipstar::Queue *ChipQueue);

  virtual ~CHIPCallbackDataLevel0() override {}
};

class CHIPCallbackEventMonitorLevel0 : public chipstar::EventMonitor {
public:

  ~CHIPCallbackEventMonitorLevel0() {
    logTrace("CHIPCallbackEventMonitorLevel0 DEST");
    join();
  };
  // void monitor() override;

  void monitorCallbacks();

  CHIPCallbackEventMonitorLevel0() {
    registerFunction(
        [&]() {
          CHIPCallbackDataLevel0 *CbData;
          while (true) {
            usleep(20000);
            LOCK(EventMonitorMtx); // chipstar::EventMonitor::Stop
            {

              if (Stop) {
                logTrace(
                    "CHIPCallbackEventMonitorLevel0 out of callbacks. Exiting "
                    "thread");
                if (Backend->CallbackQueue.size())
                  logError(
                      "Callback thread exiting while there are still active "
                      "callbacks in the queue");
                pthread_exit(0);
              }

              LOCK(Backend->CallbackQueueMtx); // Backend::CallbackQueue

              if ((Backend->CallbackQueue.size() == 0))
                continue;

              // get the callback item
              CbData = (CHIPCallbackDataLevel0 *)Backend->CallbackQueue.front();

              // Lock the item and members
              assert(CbData);
              LOCK( // Backend::CallbackQueue
                  CbData->CallbackDataMtx);
              Backend->CallbackQueue.pop();

              // Update Status
              logTrace(
                  "CHIPCallbackEventMonitorLevel0::monitor() checking event "
                  "status for {}",
                  static_cast<void *>(CbData->GpuReady.get()));
              CbData->GpuReady->updateFinishStatus(false);
              if (CbData->GpuReady->getEventStatus() != EVENT_STATUS_RECORDED) {
                // if not ready, push to the back
                Backend->CallbackQueue.push(CbData);
                continue;
              }
            }

            CbData->execute(hipSuccess);
            CbData->CpuCallbackComplete->hostSignal();
            CbData->GpuAck->wait();

            delete CbData;
            pthread_yield();
          }
        },
        100);
  }
  // void monitor() override;
};

class EventMonitorLevel0 : public chipstar::EventMonitor {
  // variable for storing the how much time has passed since trying to exit
  // the monitor loop
  int TimeSinceStopRequested_ = 0;
  int LastPrint_ = 0;

  size_t EventPoolSize_ = 1;
  std::vector<LZEventPool *> EventPools_;


public:
  std::shared_ptr<CHIPEventLevel0> getEventFromPool();
  CHIPContextLevel0 *ParentCtx_;
  LZEventPool *getPoolWithAvailableEvent();

  EventMonitorLevel0(CHIPContextLevel0 *ParentCtx) noexcept;
  ~EventMonitorLevel0() {
    logTrace("EventMonitorLevel0 DEST");

      // delete all event pools
  for (LZEventPool *Pool : EventPools_) {
    delete Pool;
  }
  EventPools_.clear();


    join();
  };
};

class LZEventPool {
private:
  CHIPContextLevel0 *Ctx_;
  ze_event_pool_handle_t EventPool_;
  unsigned int Size_;
  std::stack<std::shared_ptr<CHIPEventLevel0>> Events_;

public:
  std::mutex EventPoolMtx;
  LZEventPool(CHIPContextLevel0 *Ctx, unsigned int Size);
  ~LZEventPool();
  bool EventAvailable() { return Events_.size() > 0; }
  ze_event_pool_handle_t get() { return EventPool_; }

  void returnEvent(std::shared_ptr<CHIPEventLevel0> Event);

  std::shared_ptr<CHIPEventLevel0> getEvent();
};

enum LevelZeroQueueType {
  Unknown = 0,
  Compute,
  Copy,
};

class CHIPQueueLevel0 : public chipstar::Queue {
protected:
  ze_context_handle_t ZeCtx_;
  ze_device_handle_t ZeDev_;

  // The shared memory buffer
  void *SharedBuf_;

  // In case of interop queue may or may not be owned by chipStar
  // Ownership indicator helps during teardown
  bool zeCmdQOwnership_{true};
  /**
   * @brief Command queue handle
   * chipStar Uses the immediate command list for all its operations. However,
   * if you wish to call SYCL from HIP using the Level Zero backend then you
   * need pointers to the command queue as well. This is that command queue.
   * Current implementation does nothing with it.
   */
  ze_command_queue_group_properties_t QueueProperties_;
  ze_command_queue_desc_t QueueDescriptor_;
  ze_command_list_desc_t CommandListDesc_;
  ze_command_queue_handle_t ZeCmdQ_;
  ze_command_list_handle_t ZeCmdList_;

  void initializeCmdListImm();

public:
  std::vector<ze_event_handle_t>
  addDependenciesQueueSync(std::shared_ptr<chipstar::Event> TargetEvent);
  std::mutex CmdListMtx;
  ze_command_list_handle_t getCmdList();
  size_t getMaxMemoryFillPatternSize() {
    return QueueProperties_.maxMemoryFillPatternSize;
  }
  LevelZeroQueueType QueueType = LevelZeroQueueType::Unknown;
  CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev);
  CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev, chipstar::QueueFlags Flags);
  CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev, chipstar::QueueFlags Flags,
                  int Priority);
  CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev, chipstar::QueueFlags Flags,
                  int Priority, LevelZeroQueueType TheQueueType);

  CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev, ze_command_queue_handle_t ZeQue);
  virtual ~CHIPQueueLevel0() override;

  virtual void addCallback(hipStreamCallback_t Callback,
                           void *UserData) override;

  virtual std::shared_ptr<chipstar::Event>
  launchImpl(chipstar::ExecItem *ExecItem) override;

  virtual void finish() override;

  virtual std::shared_ptr<chipstar::Event>
  memCopyAsyncImpl(void *Dst, const void *Src, size_t Size) override;

  /**
   * @brief Execute a given command list
   *
   * @param CommandList a handle to either a compute or copy command list
   */
  void executeCommandList(ze_command_list_handle_t CommandList,
                          std::shared_ptr<chipstar::Event> Event = nullptr);
  void executeCommandListReg(ze_command_list_handle_t CommandList);
  void executeCommandListImm(std::shared_ptr<chipstar::Event> Event);

  ze_command_queue_handle_t getCmdQueue() { return ZeCmdQ_; }
  void *getSharedBufffer() { return SharedBuf_; };

  virtual std::shared_ptr<chipstar::Event>
  memFillAsyncImpl(void *Dst, size_t Size, const void *Pattern,
                   size_t PatternSize) override;

  virtual std::shared_ptr<chipstar::Event>
  memCopy2DAsyncImpl(void *Dst, size_t Dpitch, const void *Src, size_t Spitch,
                     size_t Width, size_t Height) override;

  virtual std::shared_ptr<chipstar::Event>
  memCopy3DAsyncImpl(void *Dst, size_t Dpitch, size_t Dspitch, const void *Src,
                     size_t Spitch, size_t Sspitch, size_t Width, size_t Height,
                     size_t Depth) override;

  virtual std::shared_ptr<chipstar::Event>
  memCopyToImage(ze_image_handle_t TexStorage, const void *Src,
                 const chipstar::RegionDesc &SrcRegion);

  virtual hipError_t getBackendHandles(uintptr_t *NativeInfo,
                                       int *NumHandles) override;

  virtual std::shared_ptr<chipstar::Event> enqueueMarkerImpl() override;
  std::shared_ptr<chipstar::Event> enqueueMarkerImplReg();

  virtual std::shared_ptr<chipstar::Event> enqueueBarrierImpl(
      const std::vector<std::shared_ptr<chipstar::Event>> &EventsToWaitFor)
      override;

  std::shared_ptr<chipstar::Event> enqueueBarrierImplReg(
      const std::vector<std::shared_ptr<chipstar::Event>> &EventsToWaitFor);

  virtual std::shared_ptr<chipstar::Event>
  memPrefetchImpl(const void *Ptr, size_t Count) override {
    UNIMPLEMENTED(nullptr);
  }

  void setCmdQueueOwnership(bool isOwnedByChip) {
    zeCmdQOwnership_ = isOwnedByChip;
  }
}; // end CHIPQueueLevel0

class CHIPContextLevel0 : public chipstar::Context {
  OpenCLFunctionInfoMap FuncInfos_;

public:
  std::shared_ptr<CHIPEventLevel0> getEventFromPool();

  bool ownsZeContext = true;
  void setZeContextOwnership(bool keepOwnership) {
    ownsZeContext = keepOwnership;
  }
  ze_context_handle_t ZeCtx;
  ze_driver_handle_t ZeDriver;
  CHIPContextLevel0(ze_driver_handle_t ZeDriver, ze_context_handle_t ZeCtx)
      : ZeCtx(ZeCtx), ZeDriver(ZeDriver) {
  }
  virtual ~CHIPContextLevel0() override;

  void *allocateImpl(
      size_t Size, size_t Alignment, hipMemoryType MemTy,
      chipstar::HostAllocFlags Flags = chipstar::HostAllocFlags()) override;

  bool isAllocatedPtrMappedToVM(void *Ptr) override { return false; } // TODO
  void freeImpl(void *Ptr) override;
  ze_context_handle_t &get() { return ZeCtx; }

}; // CHIPContextLevel0

class CHIPModuleLevel0 : public chipstar::Module {
  ze_module_handle_t ZeModule_ = nullptr;

public:
  CHIPModuleLevel0(const SPVModule &Src) : Module(Src) {}

  virtual ~CHIPModuleLevel0() {
    logTrace("destroy CHIPModuleLevel0 {}", (void *)this);
    for (auto *K : ChipKernels_) // Kernels must be destroyed before the module.
      delete K;
    ChipKernels_.clear();
    if (ZeModule_) {
      // The application must not call this function from
      // simultaneous threads with the same module handle.
      // Done via destructor should not be called from multiple threads
      auto Result = zeModuleDestroy(ZeModule_);
      assert(Result == ZE_RESULT_SUCCESS && "Double free?");
    }
  }

  /**
   * @brief Compile this module.
   * Extracts kernels, sets the ze_module
   *
   * @param chip_dev device for which to compile this module for
   */
  virtual void compile(chipstar::Device *ChipDev) override;
  /**
   * @brief return the raw module handle
   *
   * @return ze_module_handle_t
   */
  ze_module_handle_t get() { return ZeModule_; }
};

class CHIPKernelLevel0 : public chipstar::Kernel {
protected:
  ze_kernel_handle_t ZeKernel_;
  size_t MaxDynamicLocalSize_;
  size_t MaxWorkGroupSize_;
  size_t StaticLocalSize_;
  size_t PrivateSize_;

  CHIPModuleLevel0 *Module;
  CHIPDeviceLevel0 *Device;

public:
  CHIPKernelLevel0();

  virtual ~CHIPKernelLevel0() {
    logTrace("destroy CHIPKernelLevel0 {}", (void *)this);
    // The application must not call this function from
    // simultaneous threads with the same kernel handle.
    // Done via destructor should not be called from multiple threads
    auto Result = zeKernelDestroy(ZeKernel_);
    assert(Result == ZE_RESULT_SUCCESS && "Double free?");
  }

  CHIPKernelLevel0(ze_kernel_handle_t ZeKernel, CHIPDeviceLevel0 *Dev,
                   std::string FuncName, SPVFuncInfo *FuncInfo,
                   CHIPModuleLevel0 *Parent);
  ze_kernel_handle_t get();

  CHIPModuleLevel0 *getModule() override { return Module; }
  const CHIPModuleLevel0 *getModule() const override { return Module; }
  virtual hipError_t getAttributes(hipFuncAttributes *Attr) override;
};

// The struct that accomodate the L0/Hip texture object's content
class CHIPTextureLevel0 : public chipstar::Texture {
  ze_image_handle_t Image;
  ze_sampler_handle_t Sampler;

public:
  CHIPTextureLevel0(const hipResourceDesc &ResDesc, ze_image_handle_t TheImage,
                    ze_sampler_handle_t TheSampler)
      : chipstar::Texture(ResDesc), Image(TheImage), Sampler(TheSampler) {}

  virtual ~CHIPTextureLevel0() {
    destroyImage(Image);
    destroySampler(Sampler);
  }

  ze_image_handle_t getImage() const { return Image; }
  ze_sampler_handle_t getSampler() const { return Sampler; }

  // Destroy the LZ image object
  static void destroyImage(ze_image_handle_t Handle) {
    // The application must not call this function from
    // simultaneous threads with the same image handle.
    // Done via destructor should not be called from multiple threads
    ze_result_t Status = zeImageDestroy(Handle);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  }

  // Destroy the LZ sampler object
  static void destroySampler(ze_sampler_handle_t Handle) {
    // The application must not call this function
    // from simultaneous threads with the same sampler handle.
    // Done via destructor should not be called from multiple threads
    ze_result_t Status = zeSamplerDestroy(Handle);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  }
};

class CHIPDeviceLevel0 : public chipstar::Device {
  ze_device_handle_t ZeDev_;
  ze_context_handle_t ZeCtx_;

  ze_command_queue_group_properties_t CopyQueueProperties_;
  ze_command_queue_group_properties_t ComputeQueueProperties_;
  bool CopyQueueAvailable_ = false;
  int CopyQueueGroupOrdinal_ = -1;
  int ComputeQueueGroupOrdinal_ = -1;
  // Queues need ot be created on separate queue group indices in order to be
  // independent from one another. Use this variable to do round-robin
  // distribution across queues every time you create a queue.
  std::mutex NextQueueIndexMtx_;
  unsigned int NextCopyQueueIndex_ = 0;
  unsigned int NextComputeQueueIndex_ = 0;

  ze_command_list_desc_t CommandListComputeDesc_;
  ze_command_list_desc_t CommandListCopyDesc_;

  ze_command_list_handle_t ZeCmdListComputeImm_;
  ze_command_list_handle_t ZeCmdListCopyImm_;
  void initializeQueueGroupProperties();

  void initializeCopyQueue_();

  // The handle of device properties
  ze_device_properties_t ZeDeviceProps_;

  CHIPDeviceLevel0(ze_device_handle_t ZeDev, CHIPContextLevel0 *ChipCtx,
                   int Idx);

  ze_command_queue_desc_t getQueueDesc_(int Priority);

public:
  virtual CHIPContextLevel0 *createContext() override { return nullptr; }
  bool copyQueueIsAvailable() { return CopyQueueAvailable_; }
  ze_command_list_desc_t getCommandListComputeDesc() {
    return CommandListComputeDesc_;
  }
  ze_command_list_desc_t getCommandListCopyDesc() {
    return CommandListCopyDesc_;
  }
  ze_command_queue_group_properties_t getComputeQueueProps() {
    return ComputeQueueProperties_;
  }
  ze_command_queue_group_properties_t getCopyQueueProps() {
    return CopyQueueProperties_;
  }
  ze_command_queue_desc_t
  getNextComputeQueueDesc(int Priority = L0_DEFAULT_QUEUE_PRIORITY);
  ze_command_queue_desc_t
  getNextCopyQueueDesc(int Priority = L0_DEFAULT_QUEUE_PRIORITY);

  static CHIPDeviceLevel0 *create(ze_device_handle_t ZeDev,
                                  CHIPContextLevel0 *ChipCtx, int Idx);

  virtual void populateDevicePropertiesImpl() override;
  ze_device_handle_t &get() { return ZeDev_; }

  virtual void resetImpl() override;

  virtual chipstar::Queue *createQueue(chipstar::QueueFlags Flags,
                                       int Priority) override;
  virtual chipstar::Queue *createQueue(const uintptr_t *NativeHandles,
                                       int NumHandles) override;

  ze_device_properties_t *getDeviceProps() { return &(this->ZeDeviceProps_); };
  bool hasOnDemandPaging() const {
    return (ZeDeviceProps_.flags & ZE_DEVICE_PROPERTY_FLAG_ONDEMANDPAGING);
  }

  ze_image_handle_t allocateImage(unsigned int TextureType,
                                  hipChannelFormatDesc Format,
                                  bool NormalizeToFloat, size_t Width,
                                  size_t Height = 0, size_t Depth = 0);

  virtual chipstar::Texture *
  createTexture(const hipResourceDesc *PResDesc, const hipTextureDesc *PTexDesc,
                const struct hipResourceViewDesc *PResViewDesc) override;

  virtual void destroyTexture(chipstar::Texture *TextureObject) override {
    logTrace("CHIPDeviceLevel0::destroyTexture");
    delete TextureObject;
  }

  CHIPModuleLevel0 *compile(const SPVModule &Src) override;
};

class CHIPBackendLevel0 : public chipstar::Backend {
  bool useImmCmdLists_ = true; // default to immediate command lists
  int collectEventsTimeout_ = 30;


public:
  EventMonitorLevel0 *EventMonitor_ = nullptr;
  int getCollectEventsTimeout() { return collectEventsTimeout_; }
  void setCollectEventsTimeout() {
    auto str = readEnvVar("CHIP_L0_COLLECT_EVENTS_TIMEOUT", true);
    // assert that str can be converted to an int
    if (str.empty())
      str = "30";
    assert(isConvertibleToInt(str) && "CHIP_L0_COLLECT_EVENTS_TIMEOUT "
                                      "must be an integer or empty "
                                      "(default: 30) measured in seconds");
    collectEventsTimeout_ = std::stoi(str);
    logDebug("CHIP_L0_COLLECT_EVENTS_TIMEOUT = {}", collectEventsTimeout_);
  }

  bool getUseImmCmdLists() { return useImmCmdLists_; }
  void setUseImmCmdLists(std::string_view DeviceName) {
    // Immediate command lists seem to not work on some Intel iGPUs
    // https://en.wikipedia.org/wiki/List_of_Intel_graphics_processing_units
    std::vector<std::string> IgpuDevices = {"UHD", "HD", "Iris"};
    // check if the device name contains any of the strings in IgpuDevices
    bool IsIgpu = std::any_of(IgpuDevices.begin(), IgpuDevices.end(),
                              [&](const std::string &S) {
                                return DeviceName.find(S) != std::string::npos;
                              });
    auto str = readEnvVar("CHIP_L0_IMM_CMD_LISTS", true);
    // assert that str is either 0, 1, on, off or empty
    assert((str.empty() || str == "0" || str == "1" || str == "on" ||
            str == "off") &&
           "CHIP_L0_IMM_CMD_LISTS must be 0, 1, on, off or "
           "empty (default: on)");
    useImmCmdLists_ = (str == "1" || str == "on" || str.empty());
    if (IsIgpu && useImmCmdLists_) {
      logWarn("Immediate command lists are not supported on this device. "
              "Some tests likely to fail.");
    }
    logDebug("CHIP_L0_IMM_CMD_LISTS = {}", useImmCmdLists_ ? "true" : "false");
  }

  virtual chipstar::ExecItem *createExecItem(dim3 GirdDim, dim3 BlockDim,
                                             size_t SharedMem,
                                             hipStream_t ChipQueue) override;

  virtual void uninitialize() override;
  std::mutex CommandListsMtx;

  std::map<CHIPEventLevel0 *, ze_command_list_handle_t> EventCommandListMap;

  virtual void initializeImpl(std::string CHIPPlatformStr,
                              std::string CHIPDeviceTypeStr,
                              std::string CHIPDeviceStr) override;

  virtual void initializeFromNative(const uintptr_t *NativeHandles,
                                    int NumHandles) override;

  virtual std::string getDefaultJitFlags() override;

  virtual int ReqNumHandles() override { return 4; }

  virtual chipstar::Queue *createCHIPQueue(chipstar::Device *ChipDev) override {
    CHIPDeviceLevel0 *ChipDevLz = (CHIPDeviceLevel0 *)ChipDev;
    auto Q = new CHIPQueueLevel0(ChipDevLz);

    return Q;
  }

  /**
   * @brief Check if the given event is associated with a command list and if
   * so, destroy it. This operation will lock the CommandListsMtx.
   *
   * @param ChipEvent
   */
  void destroyAssocCmdList(CHIPEventLevel0 *ChipEvent) {

    // Check if this event is associated with a CommandList
    bool CommandListFound =
        static_cast<CHIPBackendLevel0 *>(::Backend)->EventCommandListMap.count(
            ChipEvent);
    if (CommandListFound) {
      logTrace("Erase cmdlist assoc w/ event: {}", (void *)this);
      auto CommandList = static_cast<CHIPBackendLevel0 *>(::Backend)
                             ->EventCommandListMap[ChipEvent];
      static_cast<CHIPBackendLevel0 *>(::Backend)->EventCommandListMap.erase(
          ChipEvent);

      // The application must not call this function
      // from simultaneous threads with the same command list handle.
      // Done via this is the only thread that calls it
      auto Status = zeCommandListDestroy(CommandList);
      CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
    }
  }

  virtual std::shared_ptr<chipstar::Event>
  createCHIPEvent(chipstar::Context *ChipCtx,
                  chipstar::EventFlags Flags = chipstar::EventFlags(),
                  bool UserEvent = false) override;

  virtual chipstar::CallbackData *
  createCallbackData(hipStreamCallback_t Callback, void *UserData,
                     chipstar::Queue *ChipQueue) override {
    return new CHIPCallbackDataLevel0(Callback, UserData, ChipQueue);
  }

  virtual chipstar::EventMonitor *createCallbackEventMonitor_() override {
    auto Evm = new CHIPCallbackEventMonitorLevel0();
    Evm->start();
    return Evm;
  }

  chipstar::EventMonitor *createEventMonitor_(CHIPContextLevel0 *ParentCtx)  {
    auto Evm = new EventMonitorLevel0(ParentCtx);
    Evm->start();
    return Evm;
  }

  virtual hipEvent_t getHipEvent(void *NativeEvent) override;
  virtual void *getNativeEvent(hipEvent_t HipEvent) override;

}; // CHIPBackendLevel0

#endif
