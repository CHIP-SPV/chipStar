/*
 * Copyright (c) 2021-26 chipStar developers
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
 * @file CHIPBackendVulkan.hh
 * @brief chipStar backend that runs HIP on Vulkan compute (CHIP_BE=vulkan).
 */

#ifndef CHIP_BACKEND_VULKAN_H
#define CHIP_BACKEND_VULKAN_H

#include <map>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

#include "vk_mem_alloc.h"

#include "../../CHIPBackend.hh"
#include "../../SPVReflection.hh"

// ============================================================================
// Forward declarations
// ============================================================================
class CHIPBackendVulkan;
class CHIPContextVulkan;
class CHIPDeviceVulkan;
class CHIPQueueVulkan;
class CHIPModuleVulkan;
class CHIPKernelVulkan;
class CHIPExecItemVulkan;
class CHIPEventVulkan;
class EventMonitorVulkan;

// ============================================================================
// KernelReflection: whether each argument is a descriptor or push constant
// ============================================================================
struct VulkanStorageBufferArg {
  uint32_t Ordinal = 0;    ///< Kernel-arg index.
  uint32_t Binding = 0;    ///< Binding within the descriptor set.
  /// Position in Args_[], or -1 for a hidden argument.
  int32_t HipSourceIndex = -1;
  /// Push-constant offset of the i64 that receives the pointer's byte offset
  /// into the bound buffer, or -1.
  int32_t PCOffset = -1;
  /// For a device global argument, the global's name.
  std::string DevGlobalName;
  /// For a pointer field of a by-value argument, that argument and the
  /// field's byte offset.
  int32_t FieldArg = -1;
  uint32_t FieldOffset = 0;
};

struct VulkanPushConstantArg {
  uint32_t Ordinal = 0;    ///< Kernel-arg index.
  uint32_t Offset = 0;     ///< Byte offset within the push-constant block.
  uint32_t Size = 0;       ///< Size in bytes (4, 8, 16, ...).
  /// Position in Args_[], or -1 for a hidden argument.
  int32_t HipSourceIndex = -1;
};

struct VulkanKernelReflection {
  std::string Name;                              ///< Kernel name.
  std::vector<VulkanStorageBufferArg> Buffers;   ///< set=0 bindings.
  std::vector<VulkanPushConstantArg>  PushConst; ///< Push-constant slots.
  uint32_t PushConstantBlockSize = 0;            ///< Total bytes used by all PushConst entries (rounded up).
  uint32_t MaxDescriptorBinding = 0;             ///< Highest binding used in set=0.
  /// Binding of the storage buffer the POD arguments are read from instead of
  /// push constants, or -1.
  int32_t PodBufferBinding = -1;
  uint32_t pushConstantRangeSize() const {
    return PodBufferBinding >= 0 ? 0 : PushConstantBlockSize;
  }
};

// ============================================================================
// CHIPEventVulkan: a VkFence plus a timestamp query slot
// ============================================================================
class CHIPEventVulkan : public chipstar::Event {
  VkFence Fence_ = VK_NULL_HANDLE;

  /// Slot in the device's TimestampQueryPool_, or -1 for none.
  int32_t TimestampSlot_ = -1;

  /// Host timestamp, used for timing when there is no query slot.
  uint64_t HostTimestamp_ = 0;

  /// Device timestamp in nanoseconds, or UINT64_MAX while pending.
  uint64_t Timestamp_ = UINT64_MAX;

  /// IPC: a shareable timeline semaphore, signaled to IpcValue_ by each
  /// record, and a shared page holding the latest recorded value.
  VkSemaphore IpcSem_ = VK_NULL_HANDLE;
  uint64_t *IpcTarget_ = nullptr;
  uint64_t IpcValue_ = 0;
  int IpcSemFd_ = -1;
  int IpcShmFd_ = -1;
  bool IpcOpened_ = false;

public:
  CHIPEventVulkan(chipstar::Context *Ctx,
                  chipstar::EventFlags Flags = chipstar::EventFlags());
  virtual ~CHIPEventVulkan() override;

  virtual bool updateFinishStatus(bool ThrowErrorIfNotReady = true) override;
  virtual bool wait() override;
  virtual float getElapsedTime(chipstar::Event *Other) override;
  virtual void hostSignal() override;
  virtual void getIpcHandle(hipIpcEventHandle_t *Handle) override;
  virtual bool isIpcOpened() override { return IpcOpened_; }
  /// Makes this the event another process exported with SemFd and ShmFd.
  void openIpc(int SemFd, int ShmFd);
  VkSemaphore getIpcSemaphore() const { return IpcSem_; }
  uint64_t nextIpcValue() { return ++IpcValue_; }
  void publishIpcValue(uint64_t V) {
    __atomic_store_n(IpcTarget_, V, __ATOMIC_RELEASE);
  }

  VkFence getFence() const { return Fence_; }
  void setFence(VkFence F) { Fence_ = F; }
  int32_t getTimestampSlot() const { return TimestampSlot_; }
  void setTimestampSlot(int32_t Slot) { TimestampSlot_ = Slot; }
  uint64_t &getHostTimestamp() { return HostTimestamp_; }
  uint64_t &getTimestamp() { return Timestamp_; }
};

// chipstar::CallbackData has a protected destructor, so it needs a subclass.
class CHIPCallbackDataVulkan : public chipstar::CallbackData {
public:
  CHIPCallbackDataVulkan(hipStreamCallback_t CallbackF, void *CallbackArgs,
                         chipstar::Queue *ChipQueue);
  virtual ~CHIPCallbackDataVulkan() override = default;
  /// Timeline value signaled once the callback has run.
  uint64_t DoneValue = 0;
};

/// Polls events and runs stream callbacks; Vulkan has no completion callbacks.
class EventMonitorVulkan : public chipstar::EventMonitor {
public:
  EventMonitorVulkan();
  virtual ~EventMonitorVulkan();
  virtual void monitor() override;
};

// ============================================================================
// CHIPModuleVulkan: one VkShaderModule, with per-kernel pipelines built lazily
// ============================================================================
class CHIPModuleVulkan : public chipstar::Module {
  CHIPDeviceVulkan *ChipDevice_ = nullptr;

  VkShaderModule ShaderModule_ = VK_NULL_HANDLE;

  /// This module's pipelines; its data is the module-cache artifact.
  VkPipelineCache PipelineCache_ = VK_NULL_HANDLE;
  /// Module-cache key still to be stored (set on a miss, cleared on store).
  std::string PendingCacheKey_;

  /// Per-kernel reflection records, keyed by kernel name.
  std::unordered_map<std::string, VulkanKernelReflection> Reflection_;

  std::unordered_map<std::string, VkDescriptorSetLayout> DSLayouts_;
  std::unordered_map<std::string, VkPipelineLayout> PipelineLayouts_;

  /// Compute pipelines keyed by "<kernel>:WxHxD:<dynamic shared bytes>".
  std::unordered_map<std::string, VkPipeline> Pipelines_;
  /// Largest element of a dynamic shared array; 0 if the module has none.
  uint32_t DynSharedElemBytes_ = 0;
  /// Each kernel's workgroup memory other than dynamic shared memory.
  std::unordered_map<std::string, uint32_t> StaticSharedBytes_;
  /// Each kernel's dynamic shared array element size.
  std::unordered_map<std::string, uint32_t> DynElemBytes_;
  /// Specialization constant id of the dynamic shared memory size.
  uint32_t DynSpecId_ = 0;

  mutable std::mutex ModuleMtx_;

public:
  CHIPModuleVulkan(const SPVModule &SrcMod);
  virtual ~CHIPModuleVulkan() override;
  void createModulePipelineCache(std::string_view Spv);
  void storeModulePipelineCache();

  virtual void compile(chipstar::Device *ChipDev) override;

  VkShaderModule getShaderModule() const { return ShaderModule_; }
  /// Largest dynamic shared memory a launch of the kernel may request.
  size_t maxDynamicSharedBytes(const std::string &Kernel) const;
  uint32_t getStaticSharedBytes(const std::string &Kernel) const {
    auto It = StaticSharedBytes_.find(Kernel);
    return It == StaticSharedBytes_.end() ? 0 : It->second;
  }
  CHIPDeviceVulkan *getDevice() const { return ChipDevice_; }

  /// Reflection for a kernel, or nullptr if absent.
  const VulkanKernelReflection *getReflection(const std::string &Name) const;

  VkDescriptorSetLayout getOrCreateDescriptorSetLayout(const std::string &KernelName);
  VkPipelineLayout getOrCreatePipelineLayout(const std::string &KernelName);

  /// The kernel's pipeline for a block size and dynamic shared memory size,
  /// which specialization constants 0, 1, 2 (the LocalSizeId workgroup size)
  /// and DynSpecId_ bake in.
  VkPipeline getOrCreatePipeline(const std::string &KernelName, dim3 BlockDim,
                                 size_t DynSharedBytes);
};

// ============================================================================
// CHIPKernelVulkan: an entry point; its pipelines live on the module
// ============================================================================
class CHIPKernelVulkan : public chipstar::Kernel {
  CHIPModuleVulkan *Module_ = nullptr;

  /// Points into Module_->Reflection_, or nullptr if the kernel has none.
  const VulkanKernelReflection *Reflection_ = nullptr;

public:
  CHIPKernelVulkan(std::string HostFName, SPVFuncInfo *FuncInfo,
                   CHIPModuleVulkan *Parent);
  virtual ~CHIPKernelVulkan() override;

  virtual hipError_t getAttributes(hipFuncAttributes *Attr) override;
  virtual chipstar::Module *getModule() override;
  virtual const chipstar::Module *getModule() const override;

  CHIPModuleVulkan *getVulkanModule() const { return Module_; }
  const VulkanKernelReflection *getReflection() const { return Reflection_; }
  void bindReflection(const VulkanKernelReflection *R) { Reflection_ = R; }
};

// ============================================================================
// CHIPExecItemVulkan: a launch's push constant bytes and buffer bindings
// ============================================================================
class CHIPExecItemVulkan : public chipstar::ExecItem {
  CHIPKernelVulkan *ChipKernel_ = nullptr;

  /// The push constant block, as passed to vkCmdPushConstants.
  std::vector<uint8_t> PushConstantBlob_;

  /// Buffer and range of each descriptor binding, indexed by binding.
  std::vector<VkBuffer> BufferBindings_;
  std::vector<VkDeviceSize> BufferRanges_;
public:
  CHIPExecItemVulkan(dim3 GridDim, dim3 BlockDim, size_t SharedMem,
                     hipStream_t ChipQueue);
  CHIPExecItemVulkan(const CHIPExecItemVulkan &Other);
  virtual ~CHIPExecItemVulkan() override;

  virtual chipstar::ExecItem *clone() const override;
  virtual void setKernel(chipstar::Kernel *Kernel) override;
  virtual chipstar::Kernel *getKernel() override;
  virtual void setupAllArgs() override;

  CHIPKernelVulkan *getVulkanKernel() const { return ChipKernel_; }
  const std::vector<uint8_t> &getPushConstantBlob() const { return PushConstantBlob_; }
  const std::vector<VkBuffer> &getBufferBindings() const { return BufferBindings_; }
  const std::vector<VkDeviceSize> &getBufferRanges() const { return BufferRanges_; }
};

// ============================================================================
// CHIPContextVulkan: one device; maps device pointers to their VkBuffers
// ============================================================================
class CHIPContextVulkan : public chipstar::Context {
  /// The buffer behind each allocation, keyed by its base pointer.
  struct DevPtrEntry {
    VkBuffer Buffer = VK_NULL_HANDLE;
    VmaAllocation Allocation = VK_NULL_HANDLE;
    VmaAllocationInfo AllocInfo{};
    size_t Size = 0;
    hipMemoryType MemType = hipMemoryTypeDevice;
    chipstar::HostAllocFlags Flags;
  };
  std::map<const void *, DevPtrEntry> DevPtrToEntry_;

public:
  /// Free every buffer still allocated, at device teardown.
  void freeAll(VmaAllocator Allocator);
  /// Device buffer bound for null pointer kernel arguments.
  void *getNullArgPlaceholder();
  /// Device buffer holding the POD arguments of a launch whose arguments are
  /// too large to push; each launch rewrites it in its command buffer.
  void *getPodArgBuffer();
  static constexpr size_t PodArgBufferSize = 65536; // vkCmdUpdateBuffer max

private:
  std::once_flag NullArgOnce_;
  void *NullArgPlaceholder_ = nullptr;
  std::once_flag PodArgOnce_;
  void *PodArgBuffer_ = nullptr;

public:
  void importHostMemory(void *HostPtr, size_t SizeBytes) override {}
  void releaseHostMemory(void *HostPtr) override {}
  CHIPContextVulkan();
  virtual ~CHIPContextVulkan() override;

  virtual void *
  allocateImpl(size_t Size, size_t Alignment, hipMemoryType MemType,
               chipstar::HostAllocFlags Flags = chipstar::HostAllocFlags())
      override;
  virtual bool isAllocatedPtrMappedToVM(void *Ptr) override;
  virtual void freeImpl(void *Ptr) override;

  /// The entry of an allocation's base pointer, or nullptr.
  const DevPtrEntry *getDevPtrEntry(const void *DevPtr) const;

  /// The entry of the allocation containing DevPtr, and DevPtr's offset in it.
  const DevPtrEntry *getDevPtrEntryContaining(const void *DevPtr,
                                              size_t &OutOffset) const;
  CHIPDeviceVulkan *getVulkanDevice() const;
};

// ============================================================================
// CHIPDeviceVulkan: the VkDevice and its shared pools
// ============================================================================
class CHIPDeviceVulkan : public chipstar::Device {
  VkPhysicalDevice PhysicalDevice_ = VK_NULL_HANDLE;

  VkDevice LogicalDevice_ = VK_NULL_HANDLE;

  /// The one VkQueue every stream of the device submits to.
  VkQueue ComputeQueue_ = VK_NULL_HANDLE;
  uint32_t ComputeQueueFamilyIndex_ = ~0u;

  VkPhysicalDeviceProperties Properties_{};
  VkPhysicalDeviceFeatures Features_{};
  VkPhysicalDeviceSubgroupProperties SubgroupProperties_{};
  VkPhysicalDeviceFloatControlsProperties FloatControls_{};

  VmaAllocator Allocator_ = VK_NULL_HANDLE;

  /// Pipeline cache for modules without their own.
  VkPipelineCache PipelineCache_ = VK_NULL_HANDLE;

  /// Timestamp query slots, one per timed event.
  VkQueryPool TimestampQueryPool_ = VK_NULL_HANDLE;
  static constexpr uint32_t TimestampPoolSize_ = 4096;
  std::vector<int32_t> TimestampFreeList_;
  std::mutex TimestampMtx_;

  std::vector<VkFence> FencePool_;
  std::mutex FencePoolMtx_;

  /// Serializes use of ComputeQueue_, which must be externally synchronized.
  mutable std::mutex SubmitMtx_;

  bool HasShaderInt8_ = false;
  bool HasShaderInt64_ = false;

  /// Timeline semaphores export and import as opaque fds (IPC events).
  bool HasIpcSemaphore_ = false;
  /// deviceUUID then driverUUID; opaque fds only import where both match.
  uint8_t IpcUUID_[2 * VK_UUID_SIZE] = {};
  PFN_vkGetSemaphoreFdKHR GetSemaphoreFd_ = nullptr;
  PFN_vkImportSemaphoreFdKHR ImportSemaphoreFd_ = nullptr;
  uint32_t TimestampValidBits_ = 64;

  // Only through create(), so Device::init() calls virtuals on a whole object.
  CHIPDeviceVulkan(CHIPContextVulkan *ChipContext, VkPhysicalDevice PhysDev,
                   int Idx);

public:
  /// Device printf buffers of every compiled module.
  std::vector<chipstar::DeviceVar *> getDevicePrintfBuffers();

  virtual ~CHIPDeviceVulkan() override;

  static CHIPDeviceVulkan *create(CHIPContextVulkan *ChipContext,
                                  VkPhysicalDevice PhysDev, int Idx);

  virtual chipstar::Context *createContext() override;
  virtual void populateDevicePropertiesImpl() override;
  virtual chipstar::Queue *createQueue(chipstar::QueueFlags Flags,
                                       int Priority) override;
  virtual chipstar::Queue *createQueue(const uintptr_t *NativeHandles,
                                       int NumHandles) override;
  virtual chipstar::Texture *
  createTexture(const hipResourceDesc *ResDesc, const hipTextureDesc *TexDesc,
                const struct hipResourceViewDesc *ResViewDesc) override;
  virtual void destroyTexture(chipstar::Texture *TextureObject) override;
  virtual void resetImpl() override;
  virtual chipstar::Module *compile(const SPVModule &Src) override;

  VkPhysicalDevice getPhysicalDevice() const { return PhysicalDevice_; }
  VkDevice getLogicalDevice() const { return LogicalDevice_; }
  VkQueue getComputeQueue() const { return ComputeQueue_; }
  uint32_t getComputeQueueFamilyIndex() const { return ComputeQueueFamilyIndex_; }
  const VkPhysicalDeviceProperties &getProperties() const { return Properties_; }
  const VkPhysicalDeviceFeatures &getFeatures() const { return Features_; }
  const VkPhysicalDeviceSubgroupProperties &getSubgroupProperties() const {
    return SubgroupProperties_;
  }
  const VkPhysicalDeviceFloatControlsProperties &getFloatControls() const {
    return FloatControls_;
  }
  VmaAllocator getAllocator() const { return Allocator_; }
  std::mutex &getSubmitMtx() const { return SubmitMtx_; }
  VkPipelineCache getPipelineCache() const { return PipelineCache_; }
  VkQueryPool getTimestampQueryPool() const { return TimestampQueryPool_; }


  VkFence acquireFence();
  void releaseFence(VkFence F);
  /// A timestamp slot, or -1 when the pool is exhausted.
  int32_t acquireTimestampSlot();
  void releaseTimestampSlot(int32_t Slot);
  uint32_t getTimestampValidBits() const { return TimestampValidBits_; }

  bool hasIpcSemaphore() const { return HasIpcSemaphore_; }
  const uint8_t *getIpcUUID() const { return IpcUUID_; }
  PFN_vkGetSemaphoreFdKHR getSemaphoreFdFn() const { return GetSemaphoreFd_; }
  PFN_vkImportSemaphoreFdKHR importSemaphoreFdFn() const {
    return ImportSemaphoreFd_;
  }
  /// A timeline semaphore, exportable as an opaque fd if Export.
  VkSemaphore createIpcSemaphore(bool Export);

  CHIPContextVulkan *getContext() override {
    return static_cast<CHIPContextVulkan *>(this->Device::getContext());
  }
};

// ============================================================================
// CHIPQueueVulkan: a stream on the device's VkQueue, ordered by a timeline
// semaphore
// ============================================================================
class CHIPQueueVulkan : public chipstar::Queue {
  CHIPDeviceVulkan *ChipDevice_ = nullptr;

  VkCommandPool CommandPool_ = VK_NULL_HANDLE;
  /// Descriptor sets of this queue's launches.
  VkDescriptorPool DescPool_ = VK_NULL_HANDLE;

  /// Ring of command buffers, one per enqueued operation.
  static constexpr uint32_t RingCapacity_ = 16;
  std::vector<VkCommandBuffer> CmdBufferRing_;
  uint32_t RingHead_ = 0;
  /// Timeline value signalled by each ring slot's last submit.
  std::vector<uint64_t> RingSlotValue_ = std::vector<uint64_t>(RingCapacity_);
  /// Record that Cb's submit signals timeline value Val.
  void noteRingSubmit(VkCommandBuffer Cb, uint64_t Val);

  /// Signalled with an incremented value by every submit.
  VkSemaphore TimelineSemaphore_ = VK_NULL_HANDLE;
  uint64_t TimelineValue_ = 0;

  /// Signalled by submits that have no event fence.
  VkFence FinishFence_ = VK_NULL_HANDLE;

  std::atomic<bool> IsEmptyQueue_{true};

  /// Guards the ring, pools and semaphore.
  std::mutex QueueOpMtx_;

  /// Serializes acquire, record and submit; otherwise another thread can
  /// rotate the ring back to a slot still being recorded.
  std::mutex CmdRecordMtx_;

public:
  /// Held across one operation's acquire, record and submit.
  std::unique_lock<std::mutex> lockCmdRecord() {
    return std::unique_lock<std::mutex>(CmdRecordMtx_);
  }

public:
  CHIPQueueVulkan() = delete;
  CHIPQueueVulkan(const CHIPQueueVulkan &) = delete;
  CHIPQueueVulkan(chipstar::Device *ChipDevice, chipstar::QueueFlags Flags,
                  int Priority);
  virtual ~CHIPQueueVulkan() override;

  virtual void recordEvent(chipstar::Event *Event) override;
  virtual bool isEmptyQueue() override { return IsEmptyQueue_.load(); }
  virtual std::shared_ptr<chipstar::Event>
  memCopyAsyncImpl(void *Dst, const void *Src, size_t Size,
                   hipMemcpyKind Kind) override;
  virtual std::shared_ptr<chipstar::Event>
  memFillAsyncImpl(void *Dst, size_t Size, const void *Pattern,
                   size_t PatternSize) override;
  virtual std::shared_ptr<chipstar::Event>
  memCopy2DAsyncImpl(void *Dst, size_t DPitch, const void *Src, size_t SPitch,
                     size_t Width, size_t Height, hipMemcpyKind Kind)
      override;
  virtual std::shared_ptr<chipstar::Event>
  memCopy3DAsyncImpl(void *Dst, size_t DPitch, size_t DSPitch, const void *Src,
                     size_t SPitch, size_t SSPitch, size_t Width, size_t Height,
                     size_t Depth, hipMemcpyKind Kind) override;
  // One submit for the whole fill, rather than the base class's one per row.
  virtual void memFillAsync2D(void *Dst, size_t Pitch, int Value, size_t Width,
                              size_t Height) override;
  virtual void memFillAsync3D(hipPitchedPtr PitchedDevPtr, int Value,
                              hipExtent Extent) override;
  virtual std::shared_ptr<chipstar::Event>
  launchImpl(chipstar::ExecItem *ExecItem) override;
  virtual void finish() override;
  /// Print and clear device printf records; aborts on a device-side abort.
  void drainDevicePrintf();
  /// Block until all work this queue must follow has finished.
  void waitSubmitted();
  bool Draining_ = false;
  /// Later submits on this queue wait for the value until it is signaled.
  uint64_t reserveTimelineValue();
  void signalTimelineValue(uint64_t Value);
  virtual bool query() override;
  virtual std::shared_ptr<chipstar::Event> enqueueBarrierImpl(
      const std::vector<std::shared_ptr<chipstar::Event>> &EventsToWaitFor)
      override;
  virtual std::shared_ptr<chipstar::Event> enqueueMarkerImpl() override;
  virtual std::shared_ptr<chipstar::Event>
  memPrefetchImpl(const void *Ptr, size_t Count, int DstDevId) override;
  virtual hipError_t getBackendHandles(uintptr_t *NativeHandles,
                                       int *NumHandles) override;

  CHIPDeviceVulkan *getVulkanDevice() const { return ChipDevice_; }
  VkCommandPool getCommandPool() const { return CommandPool_; }
  VkSemaphore getTimelineSemaphore() const { return TimelineSemaphore_; }
  uint64_t getTimelineValue() const { return TimelineValue_; }

  /// The next ring command buffer; the caller begins it.
  VkCommandBuffer acquireCmdBuffer();

  /// Submits Buf after EventsToWaitFor; returns its completion event.
  std::shared_ptr<chipstar::Event> submitWithEvent(
      VkCommandBuffer Buf,
      const std::vector<std::shared_ptr<chipstar::Event>> &EventsToWaitFor);

  CHIPContextVulkan *getContext() override {
    return static_cast<CHIPContextVulkan *>(ChipContext_);
  }
};

// ============================================================================
// CHIPBackendVulkan: the VkInstance and its devices
// ============================================================================
class CHIPBackendVulkan : public chipstar::Backend {
  VkInstance Instance_ = VK_NULL_HANDLE;

  VkDebugUtilsMessengerEXT DebugMessenger_ = VK_NULL_HANDLE;

  bool ValidationEnabled_ = false;
  bool DebugUtilsEnabled_ = false;

public:
  CHIPBackendVulkan();
  virtual ~CHIPBackendVulkan() override;

  /// Set by uninitialize(); queue destructors running later in other threads
  /// then skip Vulkan calls on the device being destroyed.
  static std::atomic<bool> ShuttingDown_;
  static std::recursive_mutex TeardownMtx_;

  virtual chipstar::ExecItem *createExecItem(dim3 GridDim, dim3 BlockDim,
                                             size_t SharedMem,
                                             hipStream_t ChipQueue) override;
  virtual std::string getDefaultJitFlags() override;
  virtual int ReqNumHandles() override;
  virtual void initializeImpl() override;
  virtual void initializeFromNative(const uintptr_t *NativeHandles,
                                    int NumHandles) override;
  virtual void uninitialize() override;
  virtual chipstar::Queue *createCHIPQueue(chipstar::Device *ChipDev) override;
  virtual std::shared_ptr<chipstar::Event>
  createEventShared(chipstar::Context *ChipCtx, chipstar::EventFlags Flags,
                    std::string Msg) override;
  virtual chipstar::Event *
  createEvent(chipstar::Context *ChipCtx,
              chipstar::EventFlags Flags = chipstar::EventFlags()) override;
  virtual chipstar::CallbackData *
  createCallbackData(hipStreamCallback_t Callback, void *UserData,
                     chipstar::Queue *ChipQ) override;
  virtual chipstar::EventMonitor *createEventMonitor_() override;
  virtual chipstar::Event *
  openIpcEvent(chipstar::Context *ChipCtx,
               const hipIpcEventHandle_t &Handle) override;
  virtual hipEvent_t getHipEvent(void *NativeEvent) override;
  virtual void *getNativeEvent(hipEvent_t HipEvent) override;

  VkInstance getInstance() const { return Instance_; }
};

#endif // CHIP_BACKEND_VULKAN_H
