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

#ifndef CHIP_BACKEND_LEVEL0_H
#define CHIP_BACKEND_LEVEL0_H

// TODO: Should this be a cmake parameter? env? What is max size?
#define EVENT_POOL_SIZE 1000
#define L0_DEFAULT_QUEUE_PRIORITY ZE_COMMAND_QUEUE_PRIORITY_NORMAL

#include "../../CHIPBackend.hh"
#include "../include/ze_api.h"
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

class CHIPExecItemLevel0 : public CHIPExecItem {
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
      : CHIPExecItem(GirdDim, BlockDim, SharedMem, ChipQueue) {}

  virtual ~CHIPExecItemLevel0() override {}

  virtual void setupAllArgs() override;
  virtual CHIPExecItem *clone() const override {
    auto NewExecItem = new CHIPExecItemLevel0(*this);
    return NewExecItem;
  }

  void setKernel(CHIPKernel *Kernel) override;
  CHIPKernel *getKernel() override;
};

class CHIPEventLevel0 : public CHIPEvent {
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

  // reference count
  size_t Refc_ = 1;

  std::vector<ActionFn> Actions_;

public:
  uint32_t getValidTimestampBits();
  uint64_t getHostTimestamp() { return HostTimestamp_; }
  unsigned int EventPoolIndex;
  LZEventPool *EventPool;
  CHIPEventLevel0()
      : CHIPEventLevel0((CHIPContextLevel0 *)Backend->getActiveContext()) {}
  CHIPEventLevel0(CHIPContextLevel0 *ChipCtx,
                  CHIPEventFlags Flags = CHIPEventFlags());
  CHIPEventLevel0(CHIPContextLevel0 *ChipCtx, ze_event_handle_t NativeEvent);
  CHIPEventLevel0(CHIPContextLevel0 *ChipCtx, LZEventPool *EventPool,
                  unsigned int PoolIndex, CHIPEventFlags Flags);
  virtual ~CHIPEventLevel0() override;

  void recordStream(CHIPQueue *ChipQueue) override;

  virtual bool wait() override;

  virtual bool updateFinishStatus(bool ThrowErrorIfNotReady = true) override;

  unsigned long getFinishTime();

  virtual float getElapsedTime(CHIPEvent *Other) override;

  virtual void hostSignal() override;

  void reset();

  ze_event_handle_t peek();
  ze_event_handle_t get(std::string Msg);
  virtual size_t getCHIPRefc() override;

  virtual void decreaseRefCount(std::string Reason) override;
  virtual void increaseRefCount(std::string Reason) override;

  /// Bind an action which is promised to be executed when the event is
  /// finished.
  void addAction(ActionFn Action) { Actions_.emplace_back(Action); }

  /// Execute the actions. The event must be finished.
  void doActions() {
    assert(isFinished() && "Event must be finished first!");
    for (auto &Action : Actions_)
      Action();
    Actions_.clear();
  }
};

class CHIPCallbackDataLevel0 : public CHIPCallbackData {
private:
  // ze_event_pool_handle_t ZeEventPool_;

public:
  std::mutex CallbackDataMtx;
  CHIPCallbackDataLevel0(hipStreamCallback_t CallbackF, void *CallbackArgs,
                         CHIPQueue *ChipQueue);

  virtual ~CHIPCallbackDataLevel0() override {
    delete GpuReady;
    delete CpuCallbackComplete;
    delete GpuAck;
  }
};

class CHIPCallbackEventMonitorLevel0 : public CHIPEventMonitor {
public:
  ~CHIPCallbackEventMonitorLevel0() {
    logTrace("CHIPCallbackEventMonitorLevel0 DEST");
    join();
  };
  virtual void monitor() override;
};

class CHIPStaleEventMonitorLevel0 : public CHIPEventMonitor {
public:
  ~CHIPStaleEventMonitorLevel0() {
    logTrace("CHIPStaleEventMonitorLevel0 DEST");
    join();
  };
  virtual void monitor() override;
};

class LZEventPool {
private:
  CHIPContextLevel0 *Ctx_;
  ze_event_pool_handle_t EventPool_;
  unsigned int Size_;
  std::stack<int> FreeSlots_;
  std::vector<CHIPEventLevel0 *> Events_;

  int getFreeSlot();

public:
  std::mutex EventPoolMtx;
  LZEventPool(CHIPContextLevel0 *Ctx, unsigned int Size);
  ~LZEventPool();
  ze_event_pool_handle_t get() { return EventPool_; }

  void returnSlot(int Slot);

  CHIPEventLevel0 *getEvent();
};

enum LevelZeroQueueType {
  Unknown = 0,
  Compute,
  Copy,
};

class CHIPQueueLevel0 : public CHIPQueue {
protected:
  ze_context_handle_t ZeCtx_;
  ze_device_handle_t ZeDev_;

  // The shared memory buffer
  void *SharedBuf_;

  // In case of interop queue may or may not be owned by CHIP-SPV
  // Ownership indicator helps during teardown
  bool zeCmdQOwnership_{true};
  /**
   * @brief Command queue handle
   * CHIP-SPV Uses the immediate command list for all its operations. However,
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
  ze_command_list_handle_t getCmdList();
  size_t getMaxMemoryFillPatternSize() {
    return QueueProperties_.maxMemoryFillPatternSize;
  }
  LevelZeroQueueType QueueType = LevelZeroQueueType::Unknown;
  CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev);
  CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev, CHIPQueueFlags Flags);
  CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev, CHIPQueueFlags Flags,
                  int Priority);
  CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev, CHIPQueueFlags Flags, int Priority,
                  LevelZeroQueueType TheQueueType);

  CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev, ze_command_queue_handle_t ZeQue);
  virtual ~CHIPQueueLevel0() override;

  virtual void addCallback(hipStreamCallback_t Callback,
                           void *UserData) override;

  virtual CHIPEventLevel0 *getLastEvent() override;

  virtual CHIPEvent *launchImpl(CHIPExecItem *ExecItem) override;

  virtual void finish() override;

  virtual CHIPEvent *memCopyAsyncImpl(void *Dst, const void *Src,
                                      size_t Size) override;

  /**
   * @brief Execute a given command list
   *
   * @param CommandList a handle to either a compute or copy command list
   */
  void executeCommandList(ze_command_list_handle_t CommandList);

  ze_command_queue_handle_t getCmdQueue() { return ZeCmdQ_; }
  void *getSharedBufffer() { return SharedBuf_; };

  virtual CHIPEvent *memFillAsyncImpl(void *Dst, size_t Size,
                                      const void *Pattern,
                                      size_t PatternSize) override;

  virtual CHIPEvent *memCopy2DAsyncImpl(void *Dst, size_t Dpitch,
                                        const void *Src, size_t Spitch,
                                        size_t Width, size_t Height) override;

  virtual CHIPEvent *memCopy3DAsyncImpl(void *Dst, size_t Dpitch,
                                        size_t Dspitch, const void *Src,
                                        size_t Spitch, size_t Sspitch,
                                        size_t Width, size_t Height,
                                        size_t Depth) override;

  virtual CHIPEvent *memCopyToImage(ze_image_handle_t TexStorage,
                                    const void *Src,
                                    const CHIPRegionDesc &SrcRegion);

  virtual hipError_t getBackendHandles(uintptr_t *NativeInfo,
                                       int *NumHandles) override;

  virtual CHIPEvent *enqueueMarkerImpl() override;

  virtual CHIPEvent *
  enqueueBarrierImpl(std::vector<CHIPEvent *> *EventsToWaitFor) override;

  virtual CHIPEvent *memPrefetchImpl(const void *Ptr, size_t Count) override {
    UNIMPLEMENTED(nullptr);
  }

  void setCmdQueueOwnership(bool isOwnedByChip) {
    zeCmdQOwnership_ = isOwnedByChip;
  }
}; // end CHIPQueueLevel0

class CHIPContextLevel0 : public CHIPContext {
  OpenCLFunctionInfoMap FuncInfos_;
  std::vector<LZEventPool *> EventPools_;

public:
  CHIPEventLevel0 *getEventFromPool() {

    // go through all pools and try to get an allocated event
    for (size_t i = 0; i < EventPools_.size(); i++) {
      CHIPEventLevel0 *Event = EventPools_[i]->getEvent();
      if (Event)
        return Event;
    }

    // no events available, create new pool, get event from there and return
    logTrace("No available events found in {} event pools. Creating a new "
             "event pool",
             EventPools_.size());
    auto NewEventPool = new LZEventPool(this, EVENT_POOL_SIZE);
    auto Event = NewEventPool->getEvent();
    EventPools_.push_back(NewEventPool);
    return Event;
  }

  bool ownsZeContext = true;
  void setZeContextOwnership(bool keepOwnership) {
    ownsZeContext = keepOwnership;
  }
  ze_context_handle_t ZeCtx;
  ze_driver_handle_t ZeDriver;
  CHIPContextLevel0(ze_driver_handle_t ZeDriver, ze_context_handle_t &&ZeCtx)
      : ZeCtx(ZeCtx), ZeDriver(ZeDriver) {}
  CHIPContextLevel0(ze_driver_handle_t ZeDriver, ze_context_handle_t ZeCtx)
      : ZeCtx(ZeCtx), ZeDriver(ZeDriver) {}
  virtual ~CHIPContextLevel0() override;

  void *allocateImpl(size_t Size, size_t Alignment, hipMemoryType MemTy,
                     CHIPHostAllocFlags Flags = CHIPHostAllocFlags()) override;

  bool isAllocatedPtrMappedToVM(void *Ptr) override { return false; } // TODO
  void freeImpl(void *Ptr) override;
  ze_context_handle_t &get() { return ZeCtx; }

}; // CHIPContextLevel0

class CHIPModuleLevel0 : public CHIPModule {
  ze_module_handle_t ZeModule_ = nullptr;

public:
  CHIPModuleLevel0(const SPVModule &Src) : CHIPModule(Src) {}

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
  virtual void compile(CHIPDevice *ChipDev) override;
  /**
   * @brief return the raw module handle
   *
   * @return ze_module_handle_t
   */
  ze_module_handle_t get() { return ZeModule_; }
};

class CHIPKernelLevel0 : public CHIPKernel {
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
class CHIPTextureLevel0 : public CHIPTexture {
  ze_image_handle_t Image;
  ze_sampler_handle_t Sampler;

public:
  CHIPTextureLevel0(const hipResourceDesc &ResDesc, ze_image_handle_t TheImage,
                    ze_sampler_handle_t TheSampler)
      : CHIPTexture(ResDesc), Image(TheImage), Sampler(TheSampler) {}

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

class CHIPDeviceLevel0 : public CHIPDevice {
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

  virtual CHIPQueue *createQueue(CHIPQueueFlags Flags, int Priority) override;
  virtual CHIPQueue *createQueue(const uintptr_t *NativeHandles,
                                 int NumHandles) override;

  ze_device_properties_t *getDeviceProps() { return &(this->ZeDeviceProps_); };
  bool hasOnDemandPaging() const {
    return (ZeDeviceProps_.flags & ZE_DEVICE_PROPERTY_FLAG_ONDEMANDPAGING);
  }

  ze_image_handle_t allocateImage(unsigned int TextureType,
                                  hipChannelFormatDesc Format,
                                  bool NormalizeToFloat, size_t Width,
                                  size_t Height = 0, size_t Depth = 0);

  virtual CHIPTexture *
  createTexture(const hipResourceDesc *PResDesc, const hipTextureDesc *PTexDesc,
                const struct hipResourceViewDesc *PResViewDesc) override;

  virtual void destroyTexture(CHIPTexture *TextureObject) override {
    logTrace("CHIPDeviceLevel0::destroyTexture");
    delete TextureObject;
  }

  CHIPModuleLevel0 *compile(const SPVModule &Src) override;
};

class CHIPBackendLevel0 : public CHIPBackend {

public:
  virtual CHIPExecItem *createCHIPExecItem(dim3 GirdDim, dim3 BlockDim,
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

  virtual CHIPQueue *createCHIPQueue(CHIPDevice *ChipDev) override {
    CHIPDeviceLevel0 *ChipDevLz = (CHIPDeviceLevel0 *)ChipDev;
    auto Q = new CHIPQueueLevel0(ChipDevLz);

    return Q;
  }

  virtual CHIPEventLevel0 *
  createCHIPEvent(CHIPContext *ChipCtx, CHIPEventFlags Flags = CHIPEventFlags(),
                  bool UserEvent = false) override;

  virtual CHIPCallbackData *createCallbackData(hipStreamCallback_t Callback,
                                               void *UserData,
                                               CHIPQueue *ChipQueue) override {
    return new CHIPCallbackDataLevel0(Callback, UserData, ChipQueue);
  }

  virtual CHIPEventMonitor *createCallbackEventMonitor_() override {
    auto Evm = new CHIPCallbackEventMonitorLevel0();
    Evm->start();
    return Evm;
  }

  virtual CHIPEventMonitor *createStaleEventMonitor_() override {
    auto Evm = new CHIPStaleEventMonitorLevel0();
    Evm->start();
    return Evm;
  }

  virtual hipEvent_t getHipEvent(void *NativeEvent) override;
  virtual void *getNativeEvent(hipEvent_t HipEvent) override;

}; // CHIPBackendLevel0

#endif
