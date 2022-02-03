/**
 * @file Backend.hh
 * @author Paulius Velesko (pvelesko@gmail.com)
 * @brief OpenCL backend for CHIP. CHIPBackendOpenCL class definition with
 * inheritance from CHIPBackend. Subsequent virtual function overrides.
 * @version 0.1
 * @date 2021-08-19
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef CHIP_BACKEND_OPENCL_H
#define CHIP_BACKEND_OPENCL_H

#define CL_TARGET_OPENCL_VERSION 210
#define CL_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 210
#define CL_HPP_MINIMUM_OPENCL_VERSION 200

#include <CL/cl_ext_intel.h>

#include <CL/opencl.hpp>

#include "../../CHIPBackend.hh"
#include "exceptions.hh"
#include "spirv.hh"

std::string resultToString(int Status);

class CHIPContextOpenCL;
class CHIPDeviceOpenCL;
class CHIPExecItemOpenCL;
class CHIPKernelOpenCL;
class CHIPQueueOpenCL;
class CHIPEventOpenCL;
class CHIPBackendOpenCL;
class CHIPModuleOpenCL;

class CHIPCallbackDataOpenCL {
private:
public:
  CHIPQueueOpenCL *ChipQueue;
  void *CallbackArgs;
  hipStreamCallback_t CallbackF;
  hipError_t Status;

  CHIPCallbackDataOpenCL(hipStreamCallback_t CallbackF, void *CallbackArgs,
                         CHIPQueue *ChipQueue);
};

class CHIPEventMonitorOpenCL : public CHIPEventMonitor {
public:
  CHIPEventMonitorOpenCL();
  virtual void monitor() override;
};

class CHIPEventOpenCL : public CHIPEvent {
public:
  cl_event ClEvent;
  friend class CHIPEventOpenCL;

public:
  CHIPEventOpenCL(CHIPContextOpenCL *ChipContext, cl_event ClEvent,
                  CHIPEventFlags Flags = CHIPEventFlags());
  CHIPEventOpenCL(CHIPContextOpenCL *ChipContext,
                  CHIPEventFlags Flags = CHIPEventFlags());
  virtual ~CHIPEventOpenCL() override;
  virtual void takeOver(CHIPEvent *Other) override;
  virtual void decreaseRefCount() override;
  virtual void increaseRefCount() override;
  bool wait() override;
  float getElapsedTime(CHIPEvent *Other) override;
  virtual void hostSignal() override;
  virtual bool updateFinishStatus() override;
  cl_event peek();
  cl_event get();
  uint64_t getFinishTime();
  size_t *getRefCount();
};

class CHIPModuleOpenCL : public CHIPModule {
protected:
  cl::Program Program_;

public:
  CHIPModuleOpenCL(std::string *ModuleStr);
  virtual void compile(CHIPDevice *ChipDevice) override;
  cl::Program &get();
};

class SVMemoryRegion {
  // ContextMutex should be enough

  std::map<void *, size_t> SvmAllocations_;
  cl::Context Context_;

public:
  void init(cl::Context &C) { Context_ = C; }
  SVMemoryRegion &operator=(SVMemoryRegion &&Rhs);
  void *allocate(cl::Context Ctx, size_t Size);
  bool free(void *P, size_t *Size);
  bool hasPointer(const void *Ptr);
  bool pointerSize(void *Ptr, size_t *Size);
  bool pointerInfo(void *Ptr, void **Base, size_t *Size);
  int memCopy(void *Dst, const void *Src, size_t Size, cl::CommandQueue &Queue);
  int memFill(void *Dst, size_t Size, const void *Pattern, size_t PatternSize,
              cl::CommandQueue &Queue);
  void clear();
};

class CHIPContextOpenCL : public CHIPContext {
public:
  SVMemoryRegion SvmMemory;
  cl::Context *ClContext;
  CHIPContextOpenCL(cl::Context *ClContext);

  void *allocateImpl(size_t Size, size_t Alignment,
                     CHIPMemoryType MemType) override;

  virtual void freeImpl(void *Ptr) override;
  cl::Context *get();
};

class CHIPDeviceOpenCL : public CHIPDevice {
public:
  cl::Device *ClDevice;
  cl::Context *ClContext;
  CHIPDeviceOpenCL(CHIPContextOpenCL *ChipContext, cl::Device *ClDevice,
                   int Idx);
  cl::Device *get();
  virtual void populateDevicePropertiesImpl() override;
  virtual void reset() override;
  virtual CHIPModuleOpenCL *addModule(std::string *ModuleStr) override;
  virtual CHIPQueue *addQueueImpl(unsigned int Flags, int Priority) override;
  virtual CHIPTexture *
  createTexture(const hipResourceDesc *ResDesc, const hipTextureDesc *TexDesc,
                const struct hipResourceViewDesc *ResViewDesc) override;
  virtual void destroyTexture(CHIPTexture *ChipTexture) override;
};

class CHIPQueueOpenCL : public CHIPQueue {
protected:
  // Any reason to make these private/protected?
  cl::Context *ClContext_;
  cl::Device *ClDevice_;
  cl::CommandQueue *ClQueue_;

public:
  CHIPQueueOpenCL() = delete; // delete default constructor
  CHIPQueueOpenCL(const CHIPQueueOpenCL &) = delete;
  CHIPQueueOpenCL(CHIPDevice *ChipDevice);
  ~CHIPQueueOpenCL();
  virtual CHIPEventOpenCL *getLastEvent() override;
  virtual CHIPEvent *launchImpl(CHIPExecItem *ExecItem) override;
  virtual bool addCallback(hipStreamCallback_t Callback,
                           void *UserData) override;
  virtual void finish() override;
  virtual CHIPEvent *memCopyAsyncImpl(void *Dst, const void *Src,
                                      size_t Size) override;
  cl::CommandQueue *get();
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
  virtual CHIPEvent *memCopyToTextureImpl(CHIPTexture *TexObj,
                                          void *Src) override;
  virtual void getBackendHandles(unsigned long *NativeInfo, int *Size) override;
  virtual CHIPEvent *
  enqueueBarrierImpl(std::vector<CHIPEvent *> *EventsToWaitFor) override;
  virtual CHIPEvent *enqueueMarkerImpl() override;
  virtual CHIPEvent *memPrefetchImpl(const void *Ptr, size_t Count) override;
};

class CHIPKernelOpenCL : public CHIPKernel {
private:
  std::string Name_;
  size_t TotalArgSize_;
  cl::Kernel OclKernel_;

public:
  CHIPKernelOpenCL(const cl::Kernel &&ClKernel, std::string HostFName,
                   OCLFuncInfo *FuncInfo);
  OCLFuncInfo *getFuncInfo() const;
  std::string getName();
  cl::Kernel get() const;
  size_t getTotalArgSize() const;
};

class CHIPExecItemOpenCL : public CHIPExecItem {
private:
  cl::Kernel *ClKernel_;

public:
  OCLFuncInfo FuncInfo;
  int setupAllArgs(CHIPKernelOpenCL *Kernel);
  cl::Kernel *get();
};

class CHIPBackendOpenCL : public CHIPBackend {
public:
  void initializeImpl(std::string CHIPPlatformStr,
                      std::string CHIPDeviceTypeStr,
                      std::string CHIPDeviceStr) override;
  virtual std::string getDefaultJitFlags() override;
  virtual CHIPTexture *createCHIPTexture(intptr_t Image,
                                         intptr_t Sampler) override;
  virtual CHIPQueue *createCHIPQueue(CHIPDevice *ChipDev) override;
  virtual CHIPEventOpenCL *
  createCHIPEvent(CHIPContext *ChipCtx, CHIPEventFlags Flags = CHIPEventFlags(),
                  bool UserEvent = false) override;
  virtual CHIPCallbackData *createCallbackData(hipStreamCallback_t Callback,
                                               void *UserData,
                                               CHIPQueue *ChipQueue) override;
  virtual CHIPEventMonitor *createCallbackEventMonitor() override;
  virtual CHIPEventMonitor *createStaleEventMonitor() override;
};

#endif
