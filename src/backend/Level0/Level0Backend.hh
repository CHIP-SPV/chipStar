#ifndef CHIP_BACKEND_LEVEL0_H
#define CHIP_BACKEND_LEVEL0_H

#include "../src/common.hh"
#include "../../CHIPBackend.hh"
#include "../include/ze_api.h"

std::string resultToString(ze_result_t status);

// fw declares
class CHIPContextLevel0;
class CHIPDeviceLevel0;
class CHIPModuleLevel0;
class LZCommandList;

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
  ze_command_list_handle_t ze_cmd_list;

  // Immediate command lists do not support syncronization via
  // zeCommandQueueSynchronize
  ze_event_pool_handle_t event_pool;
  ze_event_handle_t finish_event;

  // The shared memory buffer
  void* shared_buf;

 public:
  CHIPQueueLevel0(CHIPDeviceLevel0* chip_dev_);

  virtual hipError_t launch(CHIPExecItem* exec_item) override;

  virtual void finish() override;

  virtual hipError_t memCopy(void* dst, const void* src, size_t size) override;
  virtual hipError_t memCopyAsync(void* dst, const void* src,
                                  size_t size) override;

  ze_command_list_handle_t getCmdList() { return ze_cmd_list; };
  void* getSharedBufffer() { return shared_buf; };

  virtual void memFillAsync(void* dst, size_t size, const void* pattern,
                            size_t pattern_size) override;

  virtual void memCopy2DAsync(void* dst, size_t dpitch, const void* src,
                              size_t spitch, size_t width,
                              size_t height) override;

  virtual void memCopy3DAsync(void* dst, size_t dpitch, size_t dspitch,
                              const void* src, size_t spitch, size_t sspitch,
                              size_t width, size_t height,
                              size_t depth) override;

  // Memory copy to texture object, i.e. image
  virtual void memCopyToTexture(CHIPTexture* texObj, void* src,
                                hipStream_t stream) override;
};

class LZImage : public CHIPTexture {
 protected:
  // Image handle
  ze_image_handle_t hImage;

  // The reference to HipLZ context
  CHIPContextLevel0* lzContext;

 public:
  LZImage(CHIPContextLevel0* lzContext, hipResourceDesc* resDesc,
          hipTextureDesc* texDesc);

  // Get the image handle
  ze_image_handle_t GetImageHandle() { return this->hImage; };

  // Update data to image
  bool upload(hipStream_t stream, void* srcptr);
};

class CHIPContextLevel0 : public CHIPContext {
  ze_context_handle_t ze_ctx;
  OpenCLFunctionInfoMap FuncInfos;

 public:
  CHIPContextLevel0(ze_context_handle_t&& _ze_ctx) : ze_ctx(_ze_ctx) {}
  CHIPContextLevel0(ze_context_handle_t _ze_ctx) : ze_ctx(_ze_ctx) {}

  void* allocate_(size_t size, size_t alignment, CHIPMemoryType memTy) override;

  void free_(void* ptr) override{};  // TODO
  ze_context_handle_t& get() { return ze_ctx; }
  virtual CHIPEvent* createEvent(unsigned flags) override;

  // Create Level-0 image object
  LZImage* createImage(hipResourceDesc* resDesc,
                       hipTextureDesc* texDesc) override;

  // Create HIP texture object
  virtual hipTextureObject_t createTextureObject(
      const hipResourceDesc* pResDesc, const hipTextureDesc* pTexDesc,
      const struct hipResourceViewDesc* pResViewDesc);

  // Destroy HIP texture object
  virtual bool destroyTextureObject(hipTextureObject_t textureObject);

};  // CHIPContextLevel0

class CHIPModuleLevel0 : public CHIPModule {
 public:
  CHIPModuleLevel0(std::string* module_str) : CHIPModule(module_str) {}
  virtual void compile(CHIPDevice* chip_dev) override;
};

class CHIPDeviceLevel0 : public CHIPDevice {
  ze_device_handle_t ze_dev;
  ze_context_handle_t ze_ctx;

  // The handle of device properties
  ze_device_properties_t ze_device_props;

 public:
  CHIPDeviceLevel0(ze_device_handle_t&& ze_dev_, CHIPContextLevel0* chip_ctx_);

  virtual void populateDeviceProperties_() override;
  ze_device_handle_t& get() { return ze_dev; }

  virtual void reset() override;
  virtual CHIPModuleLevel0* addModule(std::string* module_str) override {
    CHIPModuleLevel0* mod = new CHIPModuleLevel0(module_str);
    chip_modules.push_back(mod);
    return mod;
  }

  virtual void addQueue(unsigned int flags, int priority) override;
  ze_device_properties_t* getDeviceProps() { return &(this->ze_device_props); };
};

class CHIPBackendLevel0 : public CHIPBackend {
 public:
  virtual void initialize_(std::string CHIPPlatformStr,
                           std::string CHIPDeviceTypeStr,
                           std::string CHIPDeviceStr) override;

  void uninitialize() override { UNIMPLEMENTED(); }
};  // CHIPBackendLevel0

class CHIPEventLevel0 : public CHIPEvent {
 private:
  // The handler of HipLZ event_pool and event
  ze_event_handle_t event;
  ze_event_pool_handle_t event_pool;

  // The timestamp value
  uint64_t timestamp;

 public:
  CHIPEventLevel0(CHIPContextLevel0* chip_ctx_, CHIPEventType event_type_)
      : CHIPEvent((CHIPContext*)(chip_ctx_), event_type_) {
    CHIPContextLevel0* ze_ctx = (CHIPContextLevel0*)chip_context;
    ze_event_pool_desc_t eventPoolDesc = {
        ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr,
        ZE_EVENT_POOL_FLAG_HOST_VISIBLE,  // event in pool are visible to Host
        1                                 // count
    };

    ze_result_t status = zeEventPoolCreate(ze_ctx->get(), &eventPoolDesc, 0,
                                           nullptr, &event_pool);
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

  void recordStream(CHIPQueue* chip_queue_) override {
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
        q->getCmdList(), (uint64_t*)(q->getSharedBufffer()), nullptr, 0,
        nullptr);
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

  bool wait() override {
    std::lock_guard<std::mutex> Lock(mtx);
    if (event_status != EVENT_STATUS_RECORDING) return false;

    ze_result_t status = zeEventHostSynchronize(event, UINT64_MAX);
    CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd);

    event_status = EVENT_STATUS_RECORDED;
    return true;
  }

  bool isFinished() override { return (event_status == EVENT_STATUS_RECORDED); }
  bool isRecordingOrRecorded() const {
    return (event_status >= EVENT_STATUS_RECORDING);
  }

  bool updateFinishStatus() {
    std::lock_guard<std::mutex> Lock(mtx);
    if (event_status != EVENT_STATUS_RECORDING) return false;

    ze_result_t status = zeEventQueryStatus(event);
    CHIPERR_CHECK_LOG_AND_THROW(status, ZE_RESULT_SUCCESS, hipErrorTbd);
    if (status == ZE_RESULT_SUCCESS) event_status = EVENT_STATUS_RECORDED;

    return true;
  }

  uint64_t getFinishTime() {
    std::lock_guard<std::mutex> Lock(mtx);

    return timestamp;
  }

  float getElapsedTime(CHIPEvent* other_) override {
    CHIPEventLevel0* other = (CHIPEventLevel0*)other_;
    // std::lock_guard<std::mutex> Lock(ContextMutex);

    if (!this->isRecordingOrRecorded() || !other->isRecordingOrRecorded())
      return hipErrorInvalidResourceHandle;

    this->updateFinishStatus();
    other->updateFinishStatus();
    if (!this->isFinished() || !other->isFinished()) return hipErrorNotReady;

    uint64_t Started = this->getFinishTime();
    uint64_t Finished = other->getFinishTime();
    if (Started > Finished) std::swap(Started, Finished);

    CHIPContextLevel0* chip_ctx_lz = (CHIPContextLevel0*)chip_context;
    CHIPDeviceLevel0* chip_dev_lz =
        (CHIPDeviceLevel0*)chip_ctx_lz->getDevices()[0];

    auto props = chip_dev_lz->getDeviceProps();

    uint64_t timerResolution = props->timerResolution;
    uint32_t timestampValidBits = props->timestampValidBits;

    logDebug(
        "EventElapsedTime: Started {} Finished {} timerResolution {} "
        "timestampValidBits {}\n",
        Started, Finished, timerResolution, timestampValidBits);

    Started = (Started & (((uint64_t)1 << timestampValidBits) - 1));
    Finished = (Finished & (((uint64_t)1 << timestampValidBits) - 1));
    if (Started > Finished)
      Finished = Finished + ((uint64_t)1 << timestampValidBits) - Started;
    Started *= timerResolution;
    Finished *= timerResolution;

    logDebug("EventElapsedTime: STARTED  {} FINISHED {} \n", Started, Finished);

    // apparently fails for Intel NEO, god knows why
    // assert(Finished >= Started);
    uint64_t Elapsed;
#define NANOSECS 1000000000
    uint64_t MS = (Elapsed / NANOSECS) * 1000;
    uint64_t NS = Elapsed % NANOSECS;
    float FractInMS = ((float)NS) / 1000000.0f;
    auto ms = (float)MS + FractInMS;

    return ms;
  }
};

// The struct that accomodate the L0/Hip texture object's content
class LZTextureObject {
 public:
  intptr_t image;
  intptr_t sampler;

  LZTextureObject(){};

  // The factory function for creating the LZ texture object
  static LZTextureObject* CreateTextureObject(
      CHIPContextLevel0* lzCtx, const hipResourceDesc* pResDesc,
      const hipTextureDesc* pTexDesc,
      const struct hipResourceViewDesc* pResViewDesc);

  // Destroy the HIP texture object
  static bool DestroyTextureObject(LZTextureObject* texObj);

 protected:
  // The factory function for create the LZ image object
  static bool CreateImage(CHIPContextLevel0* lzCtx,
                          const hipResourceDesc* pResDesc,
                          const hipTextureDesc* pTexDesc,
                          const struct hipResourceViewDesc* pResViewDesc,
                          ze_image_handle_t* handle);

  // Destroy the LZ image object
  static bool DestroyImage(ze_image_handle_t handle);

  // The factory function for create the LZ sampler object
  static bool CreateSampler(CHIPContextLevel0* lzCtx,
                            const hipResourceDesc* pResDesc,
                            const hipTextureDesc* pTexDesc,
                            const struct hipResourceViewDesc* pResViewDesc,
                            ze_sampler_handle_t* handle);

  // Destroy the LZ sampler object
  static bool DestroySampler(ze_sampler_handle_t handle);
};

#endif