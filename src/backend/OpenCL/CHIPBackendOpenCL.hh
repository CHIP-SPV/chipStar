/*
 * Copyright (c) 2021-24 chipStar developers
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
 * inheritance from Backend. Subsequent virtual function overrides.
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

#include <CL/cl_ext.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-braces"
#include <CL/opencl.hpp>
#pragma GCC diagnostic pop

#include "../../CHIPBackend.hh"
#include "exceptions.hh"
#include "spirv.hh"
#include "Utils.hh"

#include "clHipErrorConversion.hh"
#include "common/SerializationBuffer.hh"

static thread_local cl_int clStatus;

#define OCL_DEFAULT_QUEUE_PRIORITY CL_QUEUE_PRIORITY_MED_KHR

// Temporary definitions (copied from <POCL>/include/CL/cl_ext_pocl.h)
// until the extension is made official.
#ifndef cl_ext_buffer_device_address
#define cl_ext_buffer_device_address

#ifndef CL_KERNEL_EXEC_INFO_DEVICE_PTRS_EXT
#define CL_KERNEL_EXEC_INFO_DEVICE_PTRS_EXT 0x11B8
#endif

#ifndef CL_MEM_DEVICE_ADDRESS_EXT
#define CL_MEM_DEVICE_ADDRESS_EXT (1ul << 31)
#endif

#ifndef CL_MEM_DEVICE_PTR_EXT
#define CL_MEM_DEVICE_PTR_EXT 0xff01
#endif

typedef cl_ulong cl_mem_device_address_EXT;

typedef cl_int(CL_API_CALL *clSetKernelArgDevicePointerEXT_fn)(
    cl_kernel kernel, cl_uint arg_index, cl_mem_device_address_EXT dev_addr);

#endif // cl_ext_buffer_device_address

std::string resultToString(int clStatus);

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
  hipError_t CallbackStatus;

  CHIPCallbackDataOpenCL(hipStreamCallback_t CallbackF, void *CallbackArgs,
                         chipstar::Queue *ChipQueue);
};

class EventMonitorOpenCL : public chipstar::EventMonitor {
public:
  EventMonitorOpenCL();
  virtual void monitor() override;
};

class CHIPEventOpenCL : public chipstar::Event {
public:
  cl_event ClEvent;
  friend class CHIPEventOpenCL;
  std::shared_ptr<chipstar::Event> RecordedEvent;
  uint64_t HostTimeStamp;

public:
  CHIPEventOpenCL(CHIPContextOpenCL *ChipContext, cl_event ClEvent,
                  chipstar::EventFlags Flags = chipstar::EventFlags());
  CHIPEventOpenCL(CHIPContextOpenCL *ChipContext,
                  chipstar::EventFlags Flags = chipstar::EventFlags());
  virtual ~CHIPEventOpenCL() override;

  void recordEventCopy(const std::shared_ptr<chipstar::Event> &Other);
  bool wait() override;
  float getElapsedTime(chipstar::Event *Other) override;
  virtual void hostSignal() override;
  virtual bool updateFinishStatus(bool ThrowErrorIfNotReady = true) override;
  cl_event *getNativePtr() { return &ClEvent; }
  cl_event &getNativeRef() { return ClEvent; }
  uint64_t getFinishTime();
  size_t getRefCount();
};

class CHIPModuleOpenCL : public chipstar::Module {
protected:
  cl::Program Program_;

public:
  CHIPModuleOpenCL(const SPVModule &SrcMod);
  virtual ~CHIPModuleOpenCL() {
    logTrace("CHIPModuleOpenCL::~CHIPModuleOpenCL");
  }
  virtual void compile(chipstar::Device *ChipDevice) override;
  cl::Program *get();
};

struct CHIPContextUSMExts {
  clSharedMemAllocINTEL_fn clSharedMemAllocINTEL;
  clDeviceMemAllocINTEL_fn clDeviceMemAllocINTEL;
  clHostMemAllocINTEL_fn clHostMemAllocINTEL;
  clMemFreeINTEL_fn clMemFreeINTEL;
};

using const_alloc_iterator = ConstMapKeyIterator<
    std::map<std::shared_ptr<void>, size_t, PointerCmp<void>>>;

/// Used to select a strategy for managing HIP allocations.
enum class AllocationStrategy {
  Unset = 0,
  CoarseGrainSVM,
  FineGrainSVM,
  IntelUSM,
  /// Use regular cl_mem buffers and an experimental
  /// cl_ext_buffer_device_address extension to obtain their fixed
  /// device addresses.
  BufferDevAddr
};

class MemoryManager {
private:
  // ContextMutex should be enough
  std::map<std::shared_ptr<void>, size_t, PointerCmp<void>> Allocations_;
  cl::Context Context_;
  cl::Device Device_;

  CHIPContextOpenCL *ChipCtxCl;

  /// Valid and used when AllocStrategy_ == IntelUSM.
  CHIPContextUSMExts USM_;

  /// Used when AllocStrategy_ == BufferDevAddr.
  std::map<const void *, cl_mem> DevPtrToBuffer_;

  /// The allocation strategy to use.
  AllocationStrategy AllocStrategy_;

  // Add these three bools
  bool hostAllocUsed;
  bool deviceAllocUsed;
  bool sharedAllocUsed;

public:
  void init(CHIPContextOpenCL *ChipCtxCl);
  MemoryManager &operator=(MemoryManager &&Rhs);
  void *allocate(size_t Size, size_t Alignment, hipMemoryType MemType);
  bool free(void *P);
  bool hasPointer(const void *Ptr);
  bool pointerSize(void *Ptr, size_t *Size);
  bool pointerInfo(void *Ptr, void **Base, size_t *Size);
  int memCopy(void *Dst, const void *Src, size_t Size, cl::CommandQueue &Queue);
  int memFill(void *Dst, size_t Size, const void *Pattern, size_t PatternSize,
              cl::CommandQueue &Queue);
  void clear();

  size_t getNumAllocations() const { return Allocations_.size(); }
  IteratorRange<const_alloc_iterator> getAllocPointers() const {
    return IteratorRange<const_alloc_iterator>(
        const_alloc_iterator(Allocations_.begin()),
        const_alloc_iterator(Allocations_.end()));
  }

  AllocationStrategy getAllocStrategy() const noexcept {
    return AllocStrategy_;
  }

  // Add getter methods for the new bools
  bool isHostAllocUsed() const { return hostAllocUsed; }
  bool isDeviceAllocUsed() const { return deviceAllocUsed; }
  bool isSharedAllocUsed() const { return sharedAllocUsed; }

  std::pair<cl_mem, size_t> translateDevPtrToBuffer(const void *DevPtr) const;

private:
  std::shared_ptr<void> allocateSVM(size_t Size, size_t Alignment,
                                    hipMemoryType MemType, bool FineGrain);
  std::shared_ptr<void> allocateUSM(size_t Size, size_t Alignment,
                                    hipMemoryType MemType);
  std::shared_ptr<void> allocateBufferDevAddr(size_t Size, size_t Alignment,
                                              hipMemoryType MemType);
};

class CHIPContextOpenCL : public chipstar::Context {
private:
  cl::Platform Platform_;
  cl::Context ClContext;
  mutable clSetKernelArgDevicePointerEXT_fn clSetKernelArgDevicePointerEXT_ =
      nullptr;

public:
  MemoryManager MemManager_;
  bool allDevicesSupportFineGrainSVMorUSM();
  CHIPContextOpenCL(cl::Context CtxIn, cl::Device Dev, cl::Platform Plat);
  virtual ~CHIPContextOpenCL() {
    logTrace("CHIPContextOpenCL::~CHIPContextOpenCL");
    MemManager_.clear();
    delete ChipDevice_;
  }
  void *allocateImpl(
      size_t Size, size_t Alignment, hipMemoryType MemType,
      chipstar::HostAllocFlags Flags = chipstar::HostAllocFlags()) override;

  bool isAllocatedPtrMappedToVM(void *Ptr) override { return false; } // TODO
  virtual void freeImpl(void *Ptr) override;
  cl::Context *get();

  size_t getNumAllocations() const { return MemManager_.getNumAllocations(); }
  IteratorRange<const_alloc_iterator> getAllocPointers() const {
    return MemManager_.getAllocPointers();
  }

  AllocationStrategy getAllocStrategy() const noexcept {
    return MemManager_.getAllocStrategy();
  }

  cl_int clSetKernelArgDevicePointerEXT(cl_kernel Kernel, cl_uint ArgIdx,
                                        const void *DevPtr) const {
    assert(getAllocStrategy() == AllocationStrategy::BufferDevAddr);
    if (!clSetKernelArgDevicePointerEXT_) {
      clSetKernelArgDevicePointerEXT_ =
          reinterpret_cast<clSetKernelArgDevicePointerEXT_fn>(
              clGetExtensionFunctionAddressForPlatform(
                  Platform_(), "clSetKernelArgDevicePointerEXT"));
    }
    assert(clSetKernelArgDevicePointerEXT_);
    static_assert(
        sizeof(cl_mem_device_address_EXT) == sizeof(void *),
        "sizeof(cl_mem_device_address_EXT) does not match host pointer size!");
    auto IntPtr = reinterpret_cast<cl_mem_device_address_EXT>(DevPtr);
    return clSetKernelArgDevicePointerEXT_(Kernel, ArgIdx, IntPtr);
  }

  std::pair<cl_mem, size_t> translateDevPtrToBuffer(const void *DevPtr) const {
    return MemManager_.translateDevPtrToBuffer(DevPtr);
  }

  const cl::Platform &getPlatform() const { return Platform_; }
};

class CHIPDeviceOpenCL : public chipstar::Device {
private:
  CHIPDeviceOpenCL(CHIPContextOpenCL *ChipContext, cl::Device *ClDevice,
                   int Idx);

  cl_device_fp_atomic_capabilities_ext Fp32AtomicAddCapabilities_;
  cl_device_fp_atomic_capabilities_ext Fp64AtomicAddCapabilities_;
  bool HasSubgroupBallot_ = false;

public:
  ~CHIPDeviceOpenCL() override {
    logTrace("CHIPDeviceOpenCL::~CHIPDeviceOpenCL");
    delete AllocTracker;
    delete ClDevice;
  }

  virtual CHIPContextOpenCL *createContext() override { return nullptr; }

  static CHIPDeviceOpenCL *create(cl::Device *ClDevice,
                                  CHIPContextOpenCL *ChipContext, int Idx);
  cl::Device *ClDevice;
  cl::Context *ClContext;
  cl::Device *get() { return ClDevice; }
  virtual void populateDevicePropertiesImpl() override;
  virtual void resetImpl() override;
  virtual chipstar::Queue *createQueue(chipstar::QueueFlags Flags,
                                       int Priority) override;
  virtual chipstar::Queue *createQueue(const uintptr_t *NativeHandles,
                                       int NumHandles) override;

  virtual chipstar::Texture *
  createTexture(const hipResourceDesc *ResDesc, const hipTextureDesc *TexDesc,
                const struct hipResourceViewDesc *ResViewDesc) override;
  virtual void destroyTexture(chipstar::Texture *ChipTexture) override {
    logTrace("CHIPDeviceOpenCL::destroyTexture");
    delete ChipTexture;
  }

  CHIPModuleOpenCL *compile(const SPVModule &SrcMod) override {
    auto CompiledModule = std::make_unique<CHIPModuleOpenCL>(SrcMod);
    CompiledModule->compile(this);
    return CompiledModule.release();
  }

  bool hasFP32AtomicAdd() noexcept {
    return (Fp32AtomicAddCapabilities_ & CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT) &&
           (Fp32AtomicAddCapabilities_ & CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT);
  }
  bool hasFP64AtomicAdd() noexcept {
    return (Fp64AtomicAddCapabilities_ & CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT) &&
           (Fp64AtomicAddCapabilities_ & CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT);
  }

  bool hasBallot() const noexcept { return HasSubgroupBallot_; }
  bool hasDoubles() const { return HipDeviceProps_.arch.hasDoubles; }

  CHIPContextOpenCL *getContext() override {
    return static_cast<CHIPContextOpenCL *>(this->Device::getContext());
  }
};

template <typename T>
static void CL_CALLBACK deleteArrayCallback(cl_event Event,
                                            cl_int CommandExecStatus,
                                            void *UserData) {
  delete[] static_cast<T *>(UserData);
}

class CHIPQueueOpenCL : public chipstar::Queue {
  // Profiling queue is known to slow down driver API calls on Intel
  // OpenCL implementation but profiling is only needed for acquiring
  // device timestamps for fulfilling hipEventElapsedTime() calls. At
  // start we use non-profiling queue and switch to profiling one when
  // needed (hipEventRecord() is called).
  //
  // TODO: Switch back to non-profiling queue when possible -
  //       e.g. when HIP event object count drops to zero.
  //
  // An alternative would be modifying the queue's properties via
  // clSetCommandQueueProperty() but that's optional driver feature.
  cl::CommandQueue ClRegularQueue_;
  cl::CommandQueue ClProfilingQueue_;

  // Enumeration for indicating the currently active queue.
  enum QueueMode {
    Regular,  /// The non-profiling queue. ClRegularQueue_ is active.
    Profiling /// ClProfilingQueue_ is active.
  };

  QueueMode QueueMode_ = Regular;

  /// Set to true when this instance is shared with another API.
  bool UsedInInterOp = false;

protected:
  /**
   * @brief Map memory to device.
   *
   * All OpenCL allocations are done using SVM allocator. On systems with only
   * coarse-grain SVM, we need to map the memory before performing any
   * operations on the host. If the device supports fine-grain SVM, then no
   * mapping will be done.
   *
   * @param AllocInfo chipstar::AllocationInfo object to be mapped for the host
   * @param Type Type of mapping to be performed. Either READ or WRITE
   */
  virtual void MemMap(const chipstar::AllocationInfo *AllocInfo,
                      chipstar::Queue::MEM_MAP_TYPE Type) override;

  /**
   * @brief Unmap memory from host.
   * Once the memory is unmapped from the host, the device will get updated data
   * and be able to perform operations on it.
   *
   * @param AllocInfo
   */
  virtual void MemUnmap(const chipstar::AllocationInfo *AllocInfo) override;

public:
  CHIPQueueOpenCL() = delete; // delete default constructor
  CHIPQueueOpenCL(const CHIPQueueOpenCL &) = delete;
  CHIPQueueOpenCL(chipstar::Device *ChipDevice, int Priority,
                  cl_command_queue Queue = nullptr);
  virtual ~CHIPQueueOpenCL() override;
  virtual void recordEvent(chipstar::Event *ChipEvent) override;
  virtual std::shared_ptr<chipstar::Event>
  launchImpl(chipstar::ExecItem *ExecItem) override;
  virtual void addCallback(hipStreamCallback_t Callback,
                           void *UserData) override;
  virtual void finish() override;
  virtual std::shared_ptr<chipstar::Event>
  memCopyAsyncImpl(void *Dst, const void *Src, size_t Size,
                   hipMemcpyKind Kind) override;
  cl::CommandQueue *get();
  virtual std::shared_ptr<chipstar::Event>
  memFillAsyncImpl(void *Dst, size_t Size, const void *Pattern,
                   size_t PatternSize) override;
  virtual std::shared_ptr<chipstar::Event>
  memCopy2DAsyncImpl(void *Dst, size_t Dpitch, const void *Src, size_t Spitch,
                     size_t Width, size_t Height, hipMemcpyKind Kind) override;
  virtual std::shared_ptr<chipstar::Event>
  memCopy3DAsyncImpl(void *Dst, size_t Dpitch, size_t Dspitch, const void *Src,
                     size_t Spitch, size_t Sspitch, size_t Width, size_t Height,
                     size_t Depth, hipMemcpyKind Kind) override;

  virtual hipError_t getBackendHandles(uintptr_t *NativeInfo,
                                       int *NumHandles) override;
  virtual std::shared_ptr<chipstar::Event> enqueueBarrierImpl(
      const std::vector<std::shared_ptr<chipstar::Event>> &EventsToWaitFor)
      override;
  virtual std::shared_ptr<chipstar::Event> enqueueMarkerImpl() override;
  virtual std::shared_ptr<chipstar::Event>
  memPrefetchImpl(const void *Ptr, size_t Count) override;

  /// Enqueues a virtual command that deletes the give host array
  /// after previously enqueud commands have finished.
  ///
  /// Precondition: HostPtr must be a valid pointer to a host allocation
  /// created with new T[].
  template <typename T> cl_int enqueueDeleteHostArray(T *HostPtr) {
    assert(HostPtr);
    cl::Event CallbackEv;
    clStatus = get()->enqueueMarkerWithWaitList(nullptr, &CallbackEv);
    if (clStatus != CL_SUCCESS)
      return clStatus;

    return CallbackEv.setCallback(CL_COMPLETE, deleteArrayCallback<T>,
                                  reinterpret_cast<void *>(HostPtr));
  }

  CHIPContextOpenCL *getContext() override {
    return static_cast<CHIPContextOpenCL *>(ChipContext_);
  };

private:
  void switchModeTo(QueueMode Mode);
};

class CHIPKernelOpenCL : public chipstar::Kernel {
private:
  std::string Name_;
  size_t MaxDynamicLocalSize_;
  size_t MaxWorkGroupSize_;
  size_t StaticLocalSize_;
  size_t PrivateSize_;

  CHIPModuleOpenCL *Module;
  CHIPDeviceOpenCL *Device;

  // Pool of kernels that can be "borrowed" via
  // borrowUniqueKernelHandle(). Their custom deleter return the borrowed
  // object back to this pool.
  std::stack<std::unique_ptr<cl::Kernel>> KernelPool_;
  std::mutex KernelPoolMutex_;

public:
  // This is for acquiring unique kernel handles. Note that this class
  // intentionally does not have cl_kernel/cl::Kernel getter function.
  friend class CHIPExecItemOpenCL;

  CHIPKernelOpenCL(cl::Kernel ClKernel, CHIPDeviceOpenCL *Dev,
                   std::string HostFName, SPVFuncInfo *FuncInfo,
                   CHIPModuleOpenCL *Parent);

  virtual ~CHIPKernelOpenCL() {
    logTrace("CHIPKernelOpenCL::~CHIPKernelOpenCL");
  }
  SPVFuncInfo *getFuncInfo() const;
  std::string getName();

  CHIPModuleOpenCL *getModule() override { return Module; }
  const CHIPModuleOpenCL *getModule() const override { return Module; }
  virtual hipError_t getAttributes(hipFuncAttributes *Attr) override;

  CHIPContextOpenCL *getContext() {
    assert(Device);
    return Device->getContext();
  }

  void serialize(chipstar::SerializationBuffer &Buffer) {
    Buffer.write(Name_);
  }
  void deserialize(chipstar::SerializationBuffer &Buffer) {
    Buffer.read(Name_);
  }

private:
  // Only allowed for CHIPExecItemOpenCL instances.
  Borrowed<cl::Kernel> borrowUniqueKernelHandle();
};

class CHIPExecItemOpenCL : public chipstar::ExecItem {
private:
  CHIPKernelOpenCL *ChipKernel_;
  Borrowed<cl::Kernel> ClKernel_;

  std::string getKernelCacheKey() const {
    if (!ChipKernel_)
      return "";
    return ChipKernel_->getName() + "_" + 
           std::to_string(GridDim_.x) + "_" + std::to_string(GridDim_.y) + "_" + 
           std::to_string(GridDim_.z) + "_" +
           std::to_string(BlockDim_.x) + "_" + std::to_string(BlockDim_.y) + "_" + 
           std::to_string(BlockDim_.z) + "_" +
           std::to_string(SharedMem_);
  }

  // Store kernel state instead of the kernel object
  struct KernelState {
    std::string Name;
    std::string ModuleName;
    dim3 GridDim;
    dim3 BlockDim;
    size_t SharedMem;
  };

public:
  CHIPExecItemOpenCL(const CHIPExecItemOpenCL &Other)
      : CHIPExecItemOpenCL(Other.GridDim_, Other.BlockDim_, Other.SharedMem_,
                           Other.ChipQueue_) {
    // TOOD Graphs Is this safe?

    ChipKernel_ = Other.ChipKernel_;
    ClKernel_ = ChipKernel_->borrowUniqueKernelHandle();

    // ChipKernel cloning currently does not copy the argument setup
    // of the cl_kernel, therefore, mark arguments being unset.
    this->ArgsSetup = false;
    this->Args_ = Other.Args_;
  }
  CHIPExecItemOpenCL(dim3 GirdDim, dim3 BlockDim, size_t SharedMem,
                     hipStream_t ChipQueue)
      : ExecItem(GirdDim, BlockDim, SharedMem, ChipQueue) {}

  virtual ~CHIPExecItemOpenCL() override {}
  virtual void setupAllArgs() override;
  cl_kernel getKernelHandle();

  virtual chipstar::ExecItem *clone() const override {
    auto NewExecItem = new CHIPExecItemOpenCL(*this);
    return NewExecItem;
  }

  void setKernel(chipstar::Kernel *Kernel) override;
  CHIPKernelOpenCL *getKernel() const override { return ChipKernel_; }

  // Add serialization support
  void serialize(chipstar::SerializationBuffer &Buffer) const override {
    // Try to load from cache first
    std::string CacheKey = getKernelCacheKey();
    if (!CacheKey.empty() && Buffer.loadFromFile(CacheKey)) {
      logInfo("Loaded kernel {} from cache", CacheKey);
      return;
    }

    // If not in cache, serialize kernel state
    ExecItem::serialize(Buffer);
    
    KernelState State;
    if (ChipKernel_) {
      State.Name = ChipKernel_->getName();
      State.GridDim = GridDim_;
      State.BlockDim = BlockDim_;
      State.SharedMem = SharedMem_;
    }
    Buffer.write(State);

    // Save to cache for future use
    if (!CacheKey.empty()) {
      if (Buffer.saveToFile(CacheKey))
        logInfo("Saved kernel {} to cache", CacheKey);
    }
  }

  void deserialize(chipstar::SerializationBuffer &Buffer) override {
    ExecItem::deserialize(Buffer);
  }
};

class CHIPBackendOpenCL : public chipstar::Backend {
public:
  /// OpenCL events don't require tracking so override and do nothing
  virtual void
  trackEvent(const std::shared_ptr<chipstar::Event> &Event) override {};
  virtual chipstar::ExecItem *createExecItem(dim3 GirdDim, dim3 BlockDim,
                                             size_t SharedMem,
                                             hipStream_t ChipQueue) override;

  virtual void uninitialize() override { waitForThreadExit(); }
  virtual void initializeImpl() override;
  virtual void initializeFromNative(const uintptr_t *NativeHandles,
                                    int NumHandles) override;

  virtual std::string getDefaultJitFlags() override;

  virtual int ReqNumHandles() override { return 4; }

  virtual chipstar::Queue *createCHIPQueue(chipstar::Device *ChipDev) override;
  virtual std::shared_ptr<chipstar::Event> createEventShared(
      chipstar::Context *ChipCtx,
      chipstar::EventFlags Flags = chipstar::EventFlags()) override;
  virtual chipstar::Event *
  createEvent(chipstar::Context *ChipCtx,
              chipstar::EventFlags Flags = chipstar::EventFlags()) override;
  virtual chipstar::CallbackData *
  createCallbackData(hipStreamCallback_t Callback, void *UserData,
                     chipstar::Queue *ChipQueue) override;
  virtual chipstar::EventMonitor *createEventMonitor_() override;

  virtual hipEvent_t getHipEvent(void *NativeEvent) override;
  virtual void *getNativeEvent(hipEvent_t HipEvent) override;
};

class CHIPTextureOpenCL : public chipstar::Texture {
  cl_mem Image;
  cl_sampler Sampler;

public:
  CHIPTextureOpenCL() = delete;
  CHIPTextureOpenCL(const hipResourceDesc &ResDesc, cl_mem TheImage,
                    cl_sampler TheSampler)
      : chipstar::Texture(ResDesc), Image(TheImage), Sampler(TheSampler) {}

  virtual ~CHIPTextureOpenCL() {
    clStatus = clReleaseMemObject(Image);
    assert(clStatus == CL_SUCCESS && "Invalid image handler?");
    clStatus = clReleaseSampler(Sampler);
    assert(clStatus == CL_SUCCESS && "Invalid sampler handler?");
    (void)clStatus;
  }

  cl_mem getImage() const { return Image; }
  cl_sampler getSampler() const { return Sampler; }
};

#endif
