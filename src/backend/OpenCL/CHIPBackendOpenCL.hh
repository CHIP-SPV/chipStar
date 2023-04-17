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

/**
 * @file Backend.hh
 * @author Paulius Velesko (pvelesko@pglc.io)
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

#include <CL/opencl.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-braces"
#include <CL/opencl.hpp>
#pragma GCC diagnostic pop

#include "../../CHIPBackend.hh"
#include "exceptions.hh"
#include "spirv.hh"
#include "Utils.hh"

#define OCL_DEFAULT_QUEUE_PRIORITY CL_QUEUE_PRIORITY_MED_KHR

std::string resultToString(int Status);

class CHIPContextOpenCL;
class CHIPDeviceOpenCL;
class CHIPExecItemOpenCL;
class CHIPKernelOpenCL;
class CHIPQueueOpenCL;
class CHIPEventOpenCL;
class CHIPBackendOpenCL;
class CHIPModuleOpenCL;
class CHIPTextureOpenCL;

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
private:
  cl::Event ClEvent;

public:
  CHIPEventOpenCL(CHIPContextOpenCL *ChipContext, cl_event ClEvent,
                  CHIPEventFlags Flags = CHIPEventFlags(),
                  bool UserEvent = false);
  CHIPEventOpenCL(CHIPContextOpenCL *ChipContext,
                  CHIPEventFlags Flags = CHIPEventFlags());
  virtual ~CHIPEventOpenCL() override{};
  virtual void recordStream(CHIPQueue *ChipQueue) override;
  virtual size_t getCHIPRefc() override;
  bool wait() override;
  float getElapsedTime(CHIPEvent *Other) override;

  // not needed anywhere
  virtual void hostSignal() override{};

  virtual bool updateFinishStatus(bool ThrowErrorIfNotReady = true) override;
  cl::Event &get() { return ClEvent; }
  void reset(cl::Event &&Ev) { ClEvent = Ev; }
  void reset(cl_event Ev) { ClEvent = Ev; }

  // for elapsedTime
  uint64_t getFinishTime();
};

class CHIPModuleOpenCL : public CHIPModule {
protected:
  cl::Program Program_;

public:
  CHIPModuleOpenCL(const SPVModule &SrcMod);
  cl::Program &get() { return Program_; }
  virtual ~CHIPModuleOpenCL() {}
  virtual void compile(CHIPDevice *ChipDevice) override;
};

class SVMemoryRegion {
  enum SVM_ALLOC_GRANULARITY { COARSE_GRAIN, FINE_GRAIN };
  // ContextMutex should be enough

  std::map<std::shared_ptr<void>, size_t, PointerCmp<void>> SvmAllocations_;
  cl::Context Context_;

public:
  using const_svm_alloc_iterator = ConstMapKeyIterator<
      std::map<std::shared_ptr<void>, size_t, PointerCmp<void>>>;

  void init(cl::Context C) { Context_ = C; }
  SVMemoryRegion &operator=(SVMemoryRegion &&Rhs);
  void *allocate(size_t Size, SVM_ALLOC_GRANULARITY Granularity = COARSE_GRAIN);
  bool free(void *P);
  bool hasPointer(const void *Ptr);
  bool pointerSize(void *Ptr, size_t *Size);
  bool pointerInfo(void *Ptr, void **Base, size_t *Size);
  void clear();

  size_t getNumAllocations() const { return SvmAllocations_.size(); }
  IteratorRange<const_svm_alloc_iterator> getSvmPointers() const {
    return IteratorRange<const_svm_alloc_iterator>(
        const_svm_alloc_iterator(SvmAllocations_.begin()),
        const_svm_alloc_iterator(SvmAllocations_.end()));
  }
};

typedef struct {
  clCreateCommandBufferKHR_fn clCreateCommandBufferKHR;
  clCommandCopyBufferKHR_fn clCommandCopyBufferKHR;
  clCommandCopyBufferRectKHR_fn clCommandCopyBufferRectKHR;
  clCommandFillBufferKHR_fn clCommandFillBufferKHR;
  clCommandNDRangeKernelKHR_fn clCommandNDRangeKernelKHR;
  clCommandBarrierWithWaitListKHR_fn clCommandBarrierWithWaitListKHR;
  clFinalizeCommandBufferKHR_fn clFinalizeCommandBufferKHR;
  clEnqueueCommandBufferKHR_fn clEnqueueCommandBufferKHR;
  clReleaseCommandBufferKHR_fn clReleaseCommandBufferKHR;
  clGetCommandBufferInfoKHR_fn clGetCommandBufferInfoKHR;

#ifdef cl_pocl_command_buffer_svm
  clCommandSVMMemcpyPOCL_fn clCommandSVMMemcpyPOCL;
  clCommandSVMMemcpyRectPOCL_fn clCommandSVMMemcpyRectPOCL;
  clCommandSVMMemfillPOCL_fn clCommandSVMMemfillPOCL;
  clCommandSVMMemfillRectPOCL_fn clCommandSVMMemfillRectPOCL;
#endif

#ifdef cl_pocl_command_buffer_host_exec
  clCommandHostFuncPOCL_fn clCommandHostFuncPOCL;
  clCommandWaitForEventPOCL_fn clCommandWaitForEventPOCL;
  clCommandSignalEventPOCL_fn clCommandSignalEventPOCL;
#endif

} CHIPContextClExts;

class CHIPContextOpenCL : public CHIPContext {
private:
  cl::Context ClContext;
  bool SupportsCommandBuffers;
  bool SupportsCommandBuffersSVM;
  bool SupportsCommandBuffersHost;
  CHIPContextClExts Exts;
  SVMemoryRegion SvmMemory;

public:
  bool allDevicesSupportFineGrainSVM();
  CHIPContextOpenCL(cl::Context ClContext, cl::Device Dev, cl::Platform Plat);
  virtual ~CHIPContextOpenCL() {}
  void *allocateImpl(size_t Size, size_t Alignment, hipMemoryType MemType,
                     CHIPHostAllocFlags Flags = CHIPHostAllocFlags()) override;

  bool isAllocatedPtrMappedToVM(void *Ptr) override { return false; } // TODO
  const SVMemoryRegion &getRegion() const { return SvmMemory; }
  virtual void freeImpl(void *Ptr) override;
  cl::Context &get() { return ClContext; }
  bool supportsCommandBuffers() { return SupportsCommandBuffers; }
  bool supportsCommandBuffersSVM() { return SupportsCommandBuffersSVM; }
  bool supportsCommandBuffersHost() { return SupportsCommandBuffersHost; }
  const CHIPContextClExts *exts() { return &Exts; }
};

class CHIPDeviceOpenCL : public CHIPDevice {
private:
  bool SupportsFineGrainSVM = false;
  CHIPDeviceOpenCL(CHIPContextOpenCL *ChipContext, cl::Device ClDevice,
                   int Idx);
  cl::Device ClDevice;

public:
  static CHIPDeviceOpenCL *create(cl::Device ClDevice,
                                  CHIPContextOpenCL *ChipContext, int Idx);
  cl::Device &get() { return ClDevice; }
  bool supportsFineGrainSVM() { return SupportsFineGrainSVM; }
  virtual void populateDevicePropertiesImpl() override;
  // unused
  virtual void resetImpl() override{};
  virtual CHIPQueue *createQueue(CHIPQueueFlags Flags, int Priority) override;
  virtual CHIPQueue *createQueue(const uintptr_t *NativeHandles,
                                 int NumHandles) override;

  virtual CHIPTexture *
  createTexture(const hipResourceDesc *ResDesc, const hipTextureDesc *TexDesc,
                const struct hipResourceViewDesc *ResViewDesc) override;
  virtual void destroyTexture(CHIPTexture *ChipTexture) override {
    logTrace("CHIPDeviceOpenCL::destroyTexture");
    delete ChipTexture;
  }

  CHIPModuleOpenCL *compile(const SPVModule &SrcMod) override {
    auto CompiledModule = std::make_unique<CHIPModuleOpenCL>(SrcMod);
    CompiledModule->compile(this);
    return CompiledModule.release();
  }
};

class CHIPQueueOpenCL : public CHIPQueue {
protected:
  // Any reason to make these private/protected?
  cl::CommandQueue ClQueue;

  /**
   * @brief Map memory to device.
   *
   * All OpenCL allocations are done using SVM allocator. On systems with only
   * coarse-grain SVM, we need to map the memory before performing any
   * operations on the host. If the device supports fine-grain SVM, then no
   * mapping will be done.
   *
   * @param AllocInfo AllocationInfo object to be mapped for the host
   * @param Type Type of mapping to be performed. Either READ or WRITE
   */
  virtual void MemMap(const AllocationInfo *AllocInfo,
                      CHIPQueue::MEM_MAP_TYPE Type) override;

  /**
   * @brief Unmap memory from host.
   * Once the memory is unmapped from the host, the device will get updated data
   * and be able to perform operations on it.
   *
   * @param AllocInfo
   */
  virtual void MemUnmap(const AllocationInfo *AllocInfo) override;

public:
  CHIPQueueOpenCL() = delete; // delete default constructor
  CHIPQueueOpenCL(const CHIPQueueOpenCL &) = delete;
  CHIPQueueOpenCL(CHIPDevice *ChipDevice, int Priority,
                  cl_command_queue Queue = nullptr);
  virtual ~CHIPQueueOpenCL() override;
  virtual CHIPEventOpenCL *getLastEvent() override;
  virtual CHIPEvent *launchImpl(CHIPExecItem *ExecItem) override;
  virtual void addCallback(hipStreamCallback_t Callback,
                           void *UserData) override;
  virtual void finish() override;
  virtual CHIPEvent *memCopyAsyncImpl(void *Dst, const void *Src,
                                      size_t Size) override;
  cl::CommandQueue &get() { return ClQueue; }
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

  virtual hipError_t getBackendHandles(uintptr_t *NativeInfo,
                                       int *NumHandles) override;
  virtual CHIPEvent *
  enqueueBarrierImpl(std::vector<CHIPEvent *> *EventsToWaitFor) override;
  virtual CHIPEvent *enqueueMarkerImpl() override;
  virtual CHIPEvent *memPrefetchImpl(const void *Ptr, size_t Count) override;

  virtual CHIPEvent *enqueueNativeGraph(CHIPGraphNative *NativeGraph) override;
  virtual CHIPGraphNative *createNativeGraph() override;
  virtual void destroyNativeGraph(CHIPGraphNative *NativeGraph) override;
};

class CHIPKernelOpenCL : public CHIPKernel {
private:
  std::string Name_;
  cl::Kernel OclKernel_;
  size_t MaxDynamicLocalSize_;
  size_t MaxWorkGroupSize_;
  size_t StaticLocalSize_;
  size_t PrivateSize_;

  CHIPModuleOpenCL *Module;
  CHIPDeviceOpenCL *Device;

public:
  CHIPKernelOpenCL(cl::Kernel ClKernel, CHIPDeviceOpenCL *Dev,
                   std::string HostFName, SPVFuncInfo *FuncInfo,
                   CHIPModuleOpenCL *Parent);

  virtual ~CHIPKernelOpenCL() {}

  SPVFuncInfo *getFuncInfo() const { return FuncInfo_; }
  std::string getName() { return Name_; }
  cl::Kernel &get() { return OclKernel_; }
  CHIPKernelOpenCL *clone();

  CHIPModuleOpenCL *getModule() override { return Module; }
  const CHIPModuleOpenCL *getModule() const override { return Module; }
  virtual hipError_t getAttributes(hipFuncAttributes *Attr) override;
};

class CHIPExecItemOpenCL : public CHIPExecItem {
private:
  std::unique_ptr<CHIPKernelOpenCL> ChipKernel_;
  cl::Kernel ClKernel_;

public:
  CHIPExecItemOpenCL(const CHIPExecItemOpenCL &Other)
      : CHIPExecItemOpenCL(Other.GridDim_, Other.BlockDim_, Other.SharedMem_,
                           Other.ChipQueue_) {
    // TOOD Graphs Is this safe?
    ClKernel_ = Other.ClKernel_;
    ChipKernel_.reset(Other.ChipKernel_->clone());
    // ChipKernel cloning currently does not copy the argument setup
    // of the cl_kernel, therefore, mark arguments being unset.
    this->ArgsSetup = false;
    this->Args_ = Other.Args_;
  }
  CHIPExecItemOpenCL(dim3 GirdDim, dim3 BlockDim, size_t SharedMem,
                     hipStream_t ChipQueue)
      : CHIPExecItem(GirdDim, BlockDim, SharedMem, ChipQueue) {}

  virtual ~CHIPExecItemOpenCL() override {}
  SPVFuncInfo FuncInfo;
  virtual void setupAllArgs() override;

  cl::Kernel &get() { return ClKernel_; }
  virtual CHIPExecItem *clone() const override {
    auto NewExecItem = new CHIPExecItemOpenCL(*this);
    return NewExecItem;
  }

  void setKernel(CHIPKernel *Kernel) override;
  CHIPKernelOpenCL *getKernel() override { return ChipKernel_.get(); }
};

class CHIPBackendOpenCL : public CHIPBackend {
public:
  virtual CHIPExecItem *createCHIPExecItem(dim3 GirdDim, dim3 BlockDim,
                                           size_t SharedMem,
                                           hipStream_t ChipQueue) override;

  virtual void uninitialize() override { waitForThreadExit(); }
  virtual void initializeImpl(std::string CHIPPlatformStr,
                              std::string CHIPDeviceTypeStr,
                              std::string CHIPDeviceStr) override;
  virtual void initializeFromNative(const uintptr_t *NativeHandles,
                                    int NumHandles) override;

  virtual std::string getDefaultJitFlags() override;

  virtual int ReqNumHandles() override { return 4; }

  virtual CHIPQueue *createCHIPQueue(CHIPDevice *ChipDev) override;
  virtual CHIPEventOpenCL *
  createCHIPEvent(CHIPContext *ChipCtx, CHIPEventFlags Flags = CHIPEventFlags(),
                  bool UserEvent = false) override;
  virtual CHIPCallbackData *createCallbackData(hipStreamCallback_t Callback,
                                               void *UserData,
                                               CHIPQueue *ChipQueue) override;
  virtual CHIPEventMonitor *createCallbackEventMonitor_() override;
  virtual CHIPEventMonitor *createStaleEventMonitor_() override;

  virtual hipEvent_t getHipEvent(void *NativeEvent) override;
  virtual void *getNativeEvent(hipEvent_t HipEvent) override;
};

class CHIPTextureOpenCL : public CHIPTexture {
  cl_mem Image;
  cl_sampler Sampler;

public:
  CHIPTextureOpenCL() = delete;
  CHIPTextureOpenCL(const hipResourceDesc &ResDesc, cl_mem TheImage,
                    cl_sampler TheSampler)
      : CHIPTexture(ResDesc), Image(TheImage), Sampler(TheSampler) {}

  virtual ~CHIPTextureOpenCL() {
    cl_int Status;
    Status = clReleaseMemObject(Image);
    assert(Status == CL_SUCCESS && "Invalid image handler?");
    Status = clReleaseSampler(Sampler);
    assert(Status == CL_SUCCESS && "Invalid sampler handler?");
    (void)Status;
  }

  cl_mem getImage() const { return Image; }
  cl_sampler getSampler() const { return Sampler; }
};

class CHIPGraphNativeOpenCL : public CHIPGraphNative {
  cl_command_buffer_khr Handle;
  cl_command_queue CmdQ;
  std::map<CHIPGraphNode *, cl_sync_point_khr> SyncPointMap;
  const CHIPContextClExts *Exts;

  bool addKernelNode(CHIPGraphNodeKernel *Node,
                     std::vector<cl_sync_point_khr> &SyncPointDeps,
                     cl_sync_point_khr *SyncPoint);
#ifdef cl_pocl_command_buffer_svm
  bool addMemcpyNode(CHIPGraphNodeMemcpy *Node,
                     std::vector<cl_sync_point_khr> &SyncPointDeps,
                     cl_sync_point_khr *SyncPoint);
  bool addMemcpyNode(CHIPGraphNodeMemcpyFromSymbol *Node,
                     std::vector<cl_sync_point_khr> &SyncPointDeps,
                     cl_sync_point_khr *SyncPoint);
  bool addMemcpyNode(CHIPGraphNodeMemcpyToSymbol *Node,
                     std::vector<cl_sync_point_khr> &SyncPointDeps,
                     cl_sync_point_khr *SyncPoint);
  bool addMemsetNode(CHIPGraphNodeMemset *Node,
                     std::vector<cl_sync_point_khr> &SyncPointDeps,
                     cl_sync_point_khr *SyncPoint);
#endif

#ifdef cl_pocl_command_buffer_host_exec
  bool addHostNode(CHIPGraphNodeHost *Node,
                   std::vector<cl_sync_point_khr> &SyncPointDeps,
                   cl_sync_point_khr *SyncPoint);
  bool addEventRecordNode(CHIPGraphNodeEventRecord *Node,
                          std::vector<cl_sync_point_khr> &SyncPointDeps,
                          cl_sync_point_khr *SyncPoint);
  bool addEventWaitNode(CHIPGraphNodeWaitEvent *Node,
                        std::vector<cl_sync_point_khr> &SyncPointDeps,
                        cl_sync_point_khr *SyncPoint);
#endif

public:
  CHIPGraphNativeOpenCL(cl_command_buffer_khr H, cl_command_queue CQ,
                        const CHIPContextClExts *E)
      : Handle(H), CmdQ(CQ), Exts(E) {}
  virtual ~CHIPGraphNativeOpenCL();
  cl_command_buffer_khr get() const { return Handle; }
  virtual bool finalize() override;
  virtual bool addNode(CHIPGraphNode *NewNode) override;
};

#endif
