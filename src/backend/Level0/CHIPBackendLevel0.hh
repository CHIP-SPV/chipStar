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
  CHIPEventLevel0(CHIPContextLevel0 *ChipCtx,
                  CHIPEventFlags Flags = CHIPEventFlags());
  virtual ~CHIPEventLevel0() override;

  void recordStream(CHIPQueue *ChipQueue) override;

  virtual bool wait() override;

  bool updateFinishStatus() override;

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

class CHIPKernelLevel0 : public CHIPKernel {
protected:
  ze_kernel_handle_t ZeKernel_;

public:
  CHIPKernelLevel0();
  CHIPKernelLevel0(ze_kernel_handle_t ZeKernel, std::string FuncName,
                   OCLFuncInfo *FuncInfo);
  ze_kernel_handle_t get();
};

class CHIPQueueLevel0 : public CHIPQueue {
protected:
  ze_context_handle_t ZeCtx_;
  ze_device_handle_t ZeDev_;

  // Immediate command list is being used. Command queue is implicit
  ze_command_list_handle_t ZeCmdListImm_;

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

  ze_command_list_handle_t getCmdList() { return ZeCmdListImm_; }
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

  // Memory copy to texture object, i.e. image
  virtual CHIPEvent *memCopyToTextureImpl(CHIPTexture *TexObj,
                                          void *Src) override;

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

  void *allocateImpl(size_t Size, size_t Alignment,
                     CHIPMemoryType MemTy) override;

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

// The struct that accomodate the L0/Hip texture object's content
class CHIPTextureLevel0 : public CHIPTexture {
public:
  CHIPTextureLevel0(intptr_t Image, intptr_t Sampler)
      : CHIPTexture(Image, Sampler){};

  // The factory function for creating the LZ texture object
  static CHIPTextureLevel0 *
  createTextureObject(CHIPQueueLevel0 *Queue, const hipResourceDesc *PResDesc,
                      const hipTextureDesc *PTexDesc,
                      const struct hipResourceViewDesc *PResViewDesc) {
    UNIMPLEMENTED(nullptr);
  };

  // Destroy the HIP texture object
  static bool destroyTextureObject(CHIPTextureLevel0 *TexObj) {
    UNIMPLEMENTED(true);
  }

  // The factory function for create the LZ image object
  static ze_image_handle_t *
  createImage(CHIPDeviceLevel0 *ChipDev, const hipResourceDesc *PResDesc,
              const hipTextureDesc *PTexDesc,
              const struct hipResourceViewDesc *PResViewDesc);

  // Destroy the LZ image object
  static bool destroyImage(ze_image_handle_t Handle) {
    // Destroy LZ image handle
    ze_result_t Status = zeImageDestroy(Handle);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

    return true;
  }

  // The factory function for create the LZ sampler object
  static ze_sampler_handle_t *
  createSampler(CHIPDeviceLevel0 *ChipDev, const hipResourceDesc *PResDesc,
                const hipTextureDesc *PTexDesc,
                const struct hipResourceViewDesc *PResViewDesc);

  // Destroy the LZ sampler object
  static bool destroySampler(ze_sampler_handle_t Handle) { // TODO return void
    // Destroy LZ samler
    ze_result_t Status = zeSamplerDestroy(Handle);
    CHIPERR_CHECK_LOG_AND_THROW(Status, ZE_RESULT_SUCCESS, hipErrorTbd);

    return true;
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

  virtual void reset() override;
  virtual CHIPModuleLevel0 *addModule(std::string *ModuleStr) override {
    logTrace("CHIPModuleLevel0::addModule()");
    CHIPModuleLevel0 *Mod = new CHIPModuleLevel0(ModuleStr);
    ChipModules.insert(std::make_pair(ModuleStr, Mod));
    return Mod;
  }

  virtual CHIPQueue *addQueueImpl(unsigned int Flags, int Priority) override;
  ze_device_properties_t *getDeviceProps() { return &(this->ZeDeviceProps_); };
  virtual CHIPTexture *
  createTexture(const hipResourceDesc *PResDesc, const hipTextureDesc *PTexDesc,
                const struct hipResourceViewDesc *PResViewDesc) override;

  virtual void destroyTexture(CHIPTexture *TextureObject) override {
    if (TextureObject == nullptr)
      CHIPERR_LOG_AND_THROW("textureObject is nullptr", hipErrorTbd);

    ze_image_handle_t ImageHandle = (ze_image_handle_t)TextureObject->Image;
    ze_sampler_handle_t SamplerHandle =
        (ze_sampler_handle_t)TextureObject->Sampler;

    if (CHIPTextureLevel0::destroyImage(ImageHandle) &&
        CHIPTextureLevel0::destroySampler(SamplerHandle)) {
      delete TextureObject;
    } else
      CHIPERR_LOG_AND_THROW("Failed to destroy texture", hipErrorTbd);
  }
};

class CHIPBackendLevel0 : public CHIPBackend {
  CHIPStaleEventMonitorLevel0 *StaleEventMonitor_;

public:
  virtual void initializeImpl(std::string CHIPPlatformStr,
                              std::string CHIPDeviceTypeStr,
                              std::string CHIPDeviceStr) override;

  virtual std::string getDefaultJitFlags() override;

  virtual CHIPTexture *createCHIPTexture(intptr_t Image,
                                         intptr_t Sampler) override {
    return new CHIPTextureLevel0(Image, Sampler);
  }
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
