#ifndef CHIP_BACKEND_LEVEL0_H
#define CHIP_BACKEND_LEVEL0_H

#include "../../CHIPBackend.hh"
#include "../include/ze_api.h"
#include "../src/common.hh"

std::string resultToString(ze_result_t status);

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
  ze_event_handle_t event;
  ze_event_pool_handle_t event_pool;

  // The timestamp value
  uint64_t timestamp;

 public:
  CHIPEventLevel0(CHIPContextLevel0* chip_ctx_,
                  CHIPEventFlags flags_ = CHIPEventFlags());
  virtual ~CHIPEventLevel0() override;

  void recordStream(CHIPQueue* chip_queue_) override;

  virtual bool wait() override;

  bool updateFinishStatus() override;

  virtual void takeOver(CHIPEvent* other) override;

  unsigned long getFinishTime();

  virtual float getElapsedTime(CHIPEvent* other_) override;

  virtual void hostSignal() override;

  virtual void barrier(CHIPQueue* chip_queue_) override;
  ze_event_handle_t peek() { return event; }
  ze_event_handle_t get() {
    increaseRefCount();
    return event;
  }
};

class CHIPCallbackDataLevel0 : public CHIPCallbackData {
 private:
  ze_event_pool_handle_t ze_event_pool;

 public:
  CHIPCallbackDataLevel0(hipStreamCallback_t callback_f_, void* callback_args_,
                         CHIPQueue* chip_queue_);
  virtual void setup() override;
};

class CHIPEventMonitorLevel0 : public CHIPEventMonitor {
 public:
  CHIPEventMonitorLevel0();
  virtual void monitor() override;
};

class CHIPKernelLevel0 : public CHIPKernel {
 protected:
  ze_kernel_handle_t ze_kernel;

 public:
  CHIPKernelLevel0();
  CHIPKernelLevel0(ze_kernel_handle_t _ze_kernel, std::string _funcName,
                   OCLFuncInfo* func_info_);

  ze_kernel_handle_t get() { return ze_kernel; }
};

class CHIPQueueLevel0 : public CHIPQueue {
 protected:
  ze_context_handle_t ze_ctx;
  ze_device_handle_t ze_dev;

  // Immediate command list is being used. Command queue is implicit
  ze_command_list_handle_t ze_cmd_list_imm;

  /**
   * @brief Command queue handle
   * CHIP-SPV Uses the immediate command list for all its operations. However,
   * if you wish to call SYCL from HIP using the Level Zero backend then you
   * need pointers to the command queue as well. This is that command queue.
   * Current implementation does nothing with it.
   */
  ze_command_queue_handle_t ze_cmd_q;

  // Immediate command lists do not support syncronization via
  // zeCommandQueueSynchronize
  ze_event_pool_handle_t event_pool;
  ze_event_handle_t finish_event;

  // The shared memory buffer
  void* shared_buf;

 public:
  CHIPQueueLevel0(CHIPDeviceLevel0* chip_dev_);

  virtual CHIPEventLevel0* getLastEvent() override;

  virtual CHIPEvent* launchImpl(CHIPExecItem* exec_item) override;

  virtual void finish() override;

  virtual CHIPEvent* memCopyAsyncImpl(void* dst, const void* src,
                                      size_t size) override;

  ze_command_list_handle_t getCmdList() { return ze_cmd_list_imm; }
  ze_command_queue_handle_t getCmdQueue() { return ze_cmd_q; }
  void* getSharedBufffer() { return shared_buf; };

  virtual CHIPEvent* memFillAsyncImpl(void* dst, size_t size,
                                      const void* pattern,
                                      size_t pattern_size) override;

  virtual CHIPEvent* memCopy2DAsyncImpl(void* dst, size_t dpitch,
                                        const void* src, size_t spitch,
                                        size_t width, size_t height) override;

  virtual CHIPEvent* memCopy3DAsyncImpl(void* dst, size_t dpitch,
                                        size_t dspitch, const void* src,
                                        size_t spitch, size_t sspitch,
                                        size_t width, size_t height,
                                        size_t depth) override;

  // Memory copy to texture object, i.e. image
  virtual CHIPEvent* memCopyToTextureImpl(CHIPTexture* texObj,
                                          void* src) override;

  virtual void getBackendHandles(unsigned long* nativeInfo, int* size) override;

  virtual CHIPEvent* enqueueMarkerImpl() override;

  virtual CHIPEvent* enqueueBarrierImpl(
      std::vector<CHIPEvent*>* eventsToWaitFor) override;

  virtual CHIPEvent* memPrefetchImpl(const void* ptr, size_t count) override {
    UNIMPLEMENTED(nullptr);
  }

};  // end CHIPQueueLevel0

class CHIPContextLevel0 : public CHIPContext {
  OpenCLFunctionInfoMap FuncInfos;

 public:
  ze_context_handle_t ze_ctx;
  ze_driver_handle_t ze_driver;
  CHIPContextLevel0(ze_driver_handle_t ze_driver_,
                    ze_context_handle_t&& _ze_ctx)
      : ze_driver(ze_driver_), ze_ctx(_ze_ctx) {}
  CHIPContextLevel0(ze_driver_handle_t ze_driver_, ze_context_handle_t _ze_ctx)
      : ze_driver(ze_driver_), ze_ctx(_ze_ctx) {}

  void* allocate_(size_t size, size_t alignment, CHIPMemoryType memTy) override;

  void free_(void* ptr) override{};  // TODO
  ze_context_handle_t& get() { return ze_ctx; }

};  // CHIPContextLevel0

class CHIPModuleLevel0 : public CHIPModule {
  ze_module_handle_t ze_module;

 public:
  CHIPModuleLevel0(std::string* module_str) : CHIPModule(module_str) {}
  /**
   * @brief Compile this module.
   * Extracts kernels, sets the ze_module
   *
   * @param chip_dev device for which to compile this module for
   */
  virtual void compile(CHIPDevice* chip_dev) override;
  /**
   * @brief return the raw module handle
   *
   * @return ze_module_handle_t
   */
  ze_module_handle_t get() { return ze_module; }
};

// The struct that accomodate the L0/Hip texture object's content
class CHIPTextureLevel0 : public CHIPTexture {
 public:
  CHIPTextureLevel0(intptr_t image_, intptr_t sampler_)
      : CHIPTexture(image_, sampler_){};

  // The factory function for creating the LZ texture object
  static CHIPTextureLevel0* CreateTextureObject(
      CHIPQueueLevel0* queue, const hipResourceDesc* pResDesc,
      const hipTextureDesc* pTexDesc,
      const struct hipResourceViewDesc* pResViewDesc) {
    UNIMPLEMENTED(nullptr);
  };

  // Destroy the HIP texture object
  static bool DestroyTextureObject(CHIPTextureLevel0* texObj) {
    UNIMPLEMENTED(true);
  }

  // The factory function for create the LZ image object
  static ze_image_handle_t* createImage(
      CHIPDeviceLevel0* chip_dev, const hipResourceDesc* pResDesc,
      const hipTextureDesc* pTexDesc,
      const struct hipResourceViewDesc* pResViewDesc);

  // Destroy the LZ image object
  static bool DestroyImage(ze_image_handle_t handle) {
    // Destroy LZ image handle
    ze_result_t status = zeImageDestroy(handle);
    CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd);

    return true;
  }

  // The factory function for create the LZ sampler object
  static ze_sampler_handle_t* createSampler(
      CHIPDeviceLevel0* chip_dev, const hipResourceDesc* pResDesc,
      const hipTextureDesc* pTexDesc,
      const struct hipResourceViewDesc* pResViewDesc);

  // Destroy the LZ sampler object
  static bool DestroySampler(ze_sampler_handle_t handle) {  // TODO return void
    // Destroy LZ samler
    ze_result_t status = zeSamplerDestroy(handle);
    CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd);

    return true;
  }
};

class CHIPDeviceLevel0 : public CHIPDevice {
  ze_device_handle_t ze_dev;
  ze_context_handle_t ze_ctx;

  // The handle of device properties
  ze_device_properties_t ze_device_props;

 public:
  CHIPDeviceLevel0(ze_device_handle_t* ze_dev_, CHIPContextLevel0* chip_ctx_);
  CHIPDeviceLevel0(ze_device_handle_t&& ze_dev_, CHIPContextLevel0* chip_ctx_);

  virtual void populateDeviceProperties_() override;
  ze_device_handle_t& get() { return ze_dev; }

  virtual void reset() override;
  virtual CHIPModuleLevel0* addModule(std::string* module_str) override {
    logTrace("CHIPModuleLevel0::addModule()");
    CHIPModuleLevel0* mod = new CHIPModuleLevel0(module_str);
    ChipModules.insert(std::make_pair(module_str, mod));
    return mod;
  }

  virtual CHIPQueue* addQueue_(unsigned int flags, int priority) override;
  ze_device_properties_t* getDeviceProps() { return &(this->ze_device_props); };
  virtual CHIPTexture* createTexture(
      const hipResourceDesc* pResDesc, const hipTextureDesc* pTexDesc,
      const struct hipResourceViewDesc* pResViewDesc) override;

  virtual void destroyTexture(CHIPTexture* textureObject) override {
    if (textureObject == nullptr)
      CHIPERR_LOG_AND_THROW("textureObject is nullptr", hipErrorTbd);

    ze_image_handle_t imageHandle = (ze_image_handle_t)textureObject->image;
    ze_sampler_handle_t samplerHandle =
        (ze_sampler_handle_t)textureObject->sampler;

    if (CHIPTextureLevel0::DestroyImage(imageHandle) &&
        CHIPTextureLevel0::DestroySampler(samplerHandle)) {
      delete textureObject;
    } else
      CHIPERR_LOG_AND_THROW("Failed to destroy texture", hipErrorTbd);
  }
};

class CHIPBackendLevel0 : public CHIPBackend {
 public:
  virtual void initialize_(std::string CHIPPlatformStr,
                           std::string CHIPDeviceTypeStr,
                           std::string CHIPDeviceStr) override;

  virtual std::string getDefaultJitFlags() override;

  void uninitialize() override {
    logDebug("");
    logDebug("CHIPBackendLevel0::uninitialize()");
    for (auto q : Backend->getQueues()) {
      CHIPContext* ctx = q->getContext();
      logDebug("Remaining {} events that haven't been collected:",
               ctx->events.size());
      for (auto e : ctx->events)
        logDebug("{} status= {} refc={}", e->msg, e->getEventStatusStr(),
                 e->getCHIPRefc());
    }
  }

  virtual CHIPTexture* createCHIPTexture(intptr_t image_,
                                         intptr_t sampler_) override {
    return new CHIPTextureLevel0(image_, sampler_);
  }
  virtual CHIPQueue* createCHIPQueue(CHIPDevice* chip_dev) override {
    CHIPDeviceLevel0* chip_dev_lz = (CHIPDeviceLevel0*)chip_dev;
    auto q = new CHIPQueueLevel0(chip_dev_lz);
    Backend->addQueue(q);
    return q;
  }
  // virtual CHIPDevice* createCHIPDevice(CHIPContext* ctx_) override {
  //   CHIPContextLevel0* chip_ctx_lz = (CHIPContextLevel0*)ctx_;
  //   return new CHIPDeviceLevel0(chip_ctx_lz);
  // };
  // virtual CHIPContext* createCHIPContext() overricreateCHIPEventde {
  //   return new CHIPContextLevel0();
  // };
  virtual CHIPEventLevel0* createCHIPEvent(
      CHIPContext* chip_ctx_,
      CHIPEventFlags flags_ = CHIPEventFlags()) override {
    auto ev = new CHIPEventLevel0((CHIPContextLevel0*)chip_ctx_, flags_);
    return ev;
  }

  virtual CHIPCallbackData* createCallbackData(
      hipStreamCallback_t callback, void* userData,
      CHIPQueue* chip_queue_) override {
    return new CHIPCallbackDataLevel0(callback, userData, chip_queue_);
  }

  virtual CHIPEventMonitor* createEventMonitor() override {
    return new CHIPEventMonitorLevel0();
  }

};  // CHIPBackendLevel0

#endif
