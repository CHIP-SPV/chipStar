#ifndef CHIP_BACKEND_LEVEL0_H
#define CHIP_BACKEND_LEVEL0_H

#include "../../CHIPBackend.hh"
#include "../include/ze_api.h"
#include "../src/common.hh"

std::string resultToString(ze_result_t Status);

// fw declares
class CHIPContextLevel0;
class CHIPDeviceLevel0;
class CHIPModuleLevel0;
class CHIPTextureLevel0;
class CHIPEventLevel0;
class CHIPQueueLevel0;
class LZCommandList;

class CHIPEventLevel0 : public CHIPEvent {
private:
  friend class CHIPEventLevel0;
  // The handler of event_pool and event
  ze_event_handle_t Event_;
  ze_event_pool_handle_t EventPool_;

  // The timestamp value
  uint64_t Timestamp_;

public:
  CHIPEventLevel0()
      : CHIPEventLevel0((CHIPContextLevel0 *)Backend->getActiveContext()) {}
  CHIPEventLevel0(CHIPContextLevel0 *ChipCtx,
                  CHIPEventFlags Flags = CHIPEventFlags());
  virtual ~CHIPEventLevel0() override;

  void recordStream(CHIPQueue *ChipQueue) override;

  virtual bool wait() override;

  virtual bool updateFinishStatus(bool ThrowErrorIfNotReady = true) override;

  virtual void takeOver(CHIPEvent *Other) override;

  unsigned long getFinishTime();

  virtual float getElapsedTime(CHIPEvent *Other) override;

  virtual void hostSignal() override;

  ze_event_handle_t peek();
  ze_event_handle_t get();
};

class CHIPCallbackDataLevel0 : public CHIPCallbackData {
private:
  // ze_event_pool_handle_t ZeEventPool_;

public:
  std::mutex Mtx;
  CHIPCallbackDataLevel0(hipStreamCallback_t CallbackF, void *CallbackArgs,
                         CHIPQueue *ChipQueue);

  virtual ~CHIPCallbackDataLevel0() override {
    GpuReady->decreaseRefCount();
    CpuCallbackComplete->decreaseRefCount();
    GpuAck->decreaseRefCount();
  }
};

class CHIPCallbackEventMonitorLevel0 : public CHIPEventMonitor {
public:
  ~CHIPCallbackEventMonitorLevel0() { join(); };
  virtual void monitor() override;
};

class CHIPStaleEventMonitorLevel0 : public CHIPEventMonitor {
public:
  ~CHIPStaleEventMonitorLevel0() { join(); };
  virtual void monitor() override;
};

class CHIPQueueLevel0 : public CHIPQueue {
protected:
  ze_context_handle_t ZeCtx_;
  ze_device_handle_t ZeDev_;
  ze_command_list_desc_t CommandListComputeDesc_;
  ze_command_list_desc_t CommandListMemoryDesc_;

  // Queues need ot be created on separate queue group indices in order to be
  // independent from one another. Use this variable to do round-robin
  // distribution across queues every time you create a queue.
  unsigned int NextQueueIndex_ = 0;

  size_t MaxMemoryFillPatternSize = 0;

  // Immediate command list is being used. Command queue is implicit
  ze_command_list_handle_t ZeCmdListCompute_;
  ze_command_list_handle_t ZeCmdListCopy_;

  ze_command_list_handle_t ZeCmdListComputeImm_;
  ze_command_list_handle_t ZeCmdListCopyImm_;

  /**
   * @brief Command queue handle
   * CHIP-SPV Uses the immediate command list for all its operations. However,
   * if you wish to call SYCL from HIP using the Level Zero backend then you
   * need pointers to the command queue as well. This is that command queue.
   * Current implementation does nothing with it.
   */
  ze_command_queue_handle_t ZeCmdQ_;

  // Immediate command lists do not support syncronization via
  // zeCommandQueueSynchronize
  ze_event_pool_handle_t EventPool_;
  ze_event_handle_t FinishEvent_;

  // The shared memory buffer
  void *SharedBuf_;

public:
  CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev);
  CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev, unsigned int Flags);
  CHIPQueueLevel0(CHIPDeviceLevel0 *ChipDev, unsigned int Flags, int Priority);

  virtual CHIPEventLevel0 *getLastEvent() override;

  virtual CHIPEvent *launchImpl(CHIPExecItem *ExecItem) override;

  virtual void finish() override;

  virtual CHIPEvent *memCopyAsyncImpl(void *Dst, const void *Src,
                                      size_t Size) override;

  ze_command_list_handle_t getCmdListCopy() {
#ifdef L0_IMM_QUEUES
    return ZeCmdListCopyImm_;
#else
    auto Status = zeCommandListCreate(ZeCtx_, ZeDev_, &CommandListMemoryDesc_,
                                      &ZeCmdListCopy_);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                                hipErrorInitializationError);
    return ZeCmdListCopy_;
#endif
  }

  ze_command_list_handle_t getCmdListCompute() {
#ifdef L0_IMM_QUEUES
    return ZeCmdListComputeImm_;
#else
    auto Status = zeCommandListCreate(ZeCtx_, ZeDev_, &CommandListComputeDesc_,
                                      &ZeCmdListCompute_);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS,
                                hipErrorInitializationError);
    return ZeCmdListCompute_;
#endif
  }

  void executeCommandList(ze_command_list_handle_t CommandList) {
#ifdef L0_IMM_QUEUES
#else
    logTrace("Executing command list");
    ze_result_t Status;
    Status = zeCommandListClose(CommandList);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
    Status =
        zeCommandQueueExecuteCommandLists(ZeCmdQ_, 1, &CommandList, nullptr);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
#endif
  };

  ze_command_list_handle_t getCmdListComputeImm() {
    return ZeCmdListComputeImm_;
  }

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

  virtual void getBackendHandles(unsigned long *NativeInfo, int *Size) override;

  virtual CHIPEvent *enqueueMarkerImpl() override;

  virtual CHIPEvent *
  enqueueBarrierImpl(std::vector<CHIPEvent *> *EventsToWaitFor) override;

  virtual CHIPEvent *memPrefetchImpl(const void *Ptr, size_t Count) override {
    UNIMPLEMENTED(nullptr);
  }

}; // end CHIPQueueLevel0

class CHIPContextLevel0 : public CHIPContext {
  OpenCLFunctionInfoMap FuncInfos_;

public:
  ze_context_handle_t ZeCtx;
  ze_driver_handle_t ZeDriver;
  CHIPContextLevel0(ze_driver_handle_t ZeDriver, ze_context_handle_t &&ZeCtx)
      : ZeCtx(ZeCtx), ZeDriver(ZeDriver) {}
  CHIPContextLevel0(ze_driver_handle_t ZeDriver, ze_context_handle_t ZeCtx)
      : ZeCtx(ZeCtx), ZeDriver(ZeDriver) {}

  void *allocateImpl(size_t Size, size_t Alignment, hipMemoryType MemTy,
                     CHIPHostAllocFlags Flags = CHIPHostAllocFlags()) override;

  void freeImpl(void *Ptr) override{}; // TODO
  ze_context_handle_t &get() { return ZeCtx; }

}; // CHIPContextLevel0

class CHIPModuleLevel0 : public CHIPModule {
  ze_module_handle_t ZeModule_;

public:
  CHIPModuleLevel0(std::string *ModuleStr) : CHIPModule(ModuleStr) {}
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
  CHIPModuleLevel0 *Module;

public:
  CHIPKernelLevel0();
  CHIPKernelLevel0(ze_kernel_handle_t ZeKernel, std::string FuncName,
                   OCLFuncInfo *FuncInfo, CHIPModuleLevel0 *Parent);
  ze_kernel_handle_t get();

  CHIPModuleLevel0 *getModule() override { return Module; }
  const CHIPModuleLevel0 *getModule() const override { return Module; }
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
    ze_result_t Status = zeImageDestroy(Handle);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  }

  // Destroy the LZ sampler object
  static void destroySampler(ze_sampler_handle_t Handle) {
    ze_result_t Status = zeSamplerDestroy(Handle);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);
  }
};

class CHIPDeviceLevel0 : public CHIPDevice {
  ze_device_handle_t ZeDev_;
  ze_context_handle_t ZeCtx_;

  // The handle of device properties
  ze_device_properties_t ZeDeviceProps_;

public:
  CHIPDeviceLevel0(ze_device_handle_t *ZeDev, CHIPContextLevel0 *ChipCtx,
                   int Idx);
  CHIPDeviceLevel0(ze_device_handle_t &&ZeDev, CHIPContextLevel0 *ChipCtx,
                   int Idx);

  virtual void populateDevicePropertiesImpl() override;
  ze_device_handle_t &get() { return ZeDev_; }

  virtual void resetImpl() override;
  virtual CHIPModuleLevel0 *addModule(std::string *ModuleStr) override {
    logTrace("CHIPModuleLevel0::addModule()");
    CHIPModuleLevel0 *Mod = new CHIPModuleLevel0(ModuleStr);
    ChipModules.insert(std::make_pair(ModuleStr, Mod));
    return Mod;
  }

  virtual CHIPQueue *addQueueImpl(unsigned int Flags, int Priority) override;
  ze_device_properties_t *getDeviceProps() { return &(this->ZeDeviceProps_); };

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
};

class CHIPBackendLevel0 : public CHIPBackend {
  CHIPStaleEventMonitorLevel0 *StaleEventMonitor_;

public:
  virtual void initializeImpl(std::string CHIPPlatformStr,
                              std::string CHIPDeviceTypeStr,
                              std::string CHIPDeviceStr) override;

  virtual std::string getDefaultJitFlags() override;

  virtual CHIPQueue *createCHIPQueue(CHIPDevice *ChipDev) override {
    CHIPDeviceLevel0 *ChipDevLz = (CHIPDeviceLevel0 *)ChipDev;
    auto Q = new CHIPQueueLevel0(ChipDevLz);
    Backend->addQueue(Q);
    return Q;
  }

  virtual CHIPEventLevel0 *
  createCHIPEvent(CHIPContext *ChipCtx, CHIPEventFlags Flags = CHIPEventFlags(),
                  bool UserEvent = false) override {
    auto Ev = new CHIPEventLevel0((CHIPContextLevel0 *)ChipCtx, Flags);

    // User Events start with refc=2
    if (UserEvent)
      Ev->increaseRefCount();

    // User Events do got get garbage collected
    if (!UserEvent)
      Backend->Events.push_back(Ev);

    return Ev;
  }

  virtual CHIPCallbackData *createCallbackData(hipStreamCallback_t Callback,
                                               void *UserData,
                                               CHIPQueue *ChipQueue) override {
    return new CHIPCallbackDataLevel0(Callback, UserData, ChipQueue);
  }

  virtual CHIPEventMonitor *createCallbackEventMonitor() override {
    auto Evm = new CHIPCallbackEventMonitorLevel0();
    Evm->start();
    return Evm;
  }

  virtual CHIPEventMonitor *createStaleEventMonitor() override {
    auto Evm = new CHIPStaleEventMonitorLevel0();
    Evm->start();
    return Evm;
  }

}; // CHIPBackendLevel0

#endif
