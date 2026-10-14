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
 * @file CHIPBackendVulkan.cc
 * @brief chipStar backend that runs HIP on Vulkan compute (CHIP_BE=vulkan).
 *
 * The one translation unit that defines VMA_IMPLEMENTATION.
 */

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include "CHIPBackendVulkan.hh"
#include "../../ModuleCache.hh"

#include "../../CHIPException.hh"
#include "../../Utils.hh"
#include "../../logging.hh"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <unordered_set>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
// Throws Err with Msg unless R is VK_SUCCESS.
inline void checkVk(VkResult R, const char *Msg, hipError_t Err) {
  if (R != VK_SUCCESS) {
    CHIPERR_LOG_AND_THROW(std::string(Msg) +
                              " VkResult=" + std::to_string(static_cast<int>(R)),
                          Err);
  }
}

// Routes validation messages to the chipStar log at their severity.
VKAPI_ATTR VkBool32 VKAPI_CALL chipVkDebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT Severity,
    VkDebugUtilsMessageTypeFlagsEXT /*Type*/,
    const VkDebugUtilsMessengerCallbackDataEXT *Data, void * /*UserData*/) {
  if (!Data || !Data->pMessage)
    return VK_FALSE;
  if (Severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    logError("[VK] {}", Data->pMessage);
  } else if (Severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    logWarn("[VK] {}", Data->pMessage);
  } else if (Severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
    logInfo("[VK] {}", Data->pMessage);
  } else {
    logDebug("[VK] {}", Data->pMessage);
  }
  return VK_FALSE; // Per spec: app must return VK_FALSE from this callback.
}

// True iff the CHIP_VK_VALIDATION env var is on.
bool chipVkValidationRequested() {
  std::string Val;
  if (!readEnvVar("CHIP_VK_VALIDATION", Val, /*Lower=*/true))
    return false;
  return Val == "on" || Val == "1" || Val == "true" || Val == "yes";
}

// True iff `Name` appears in the (LayerName, ExtensionName) list returned by
// the corresponding Vulkan enumeration call.
template <typename TProp, typename TName>
bool vkPropertyListContains(const std::vector<TProp> &Props, TName TProp::*Field,
                            const char *Name) {
  for (const auto &P : Props) {
    if (std::strcmp(P.*Field, Name) == 0)
      return true;
  }
  return false;
}
} // namespace

// ============================================================================
// CHIPEventVulkan
// ============================================================================

CHIPEventVulkan::CHIPEventVulkan(chipstar::Context *Ctx,
                                 chipstar::EventFlags Flags)
    : chipstar::Event(Ctx, Flags) {
  Fence_ = VK_NULL_HANDLE;
  TimestampSlot_ = -1;
  HostTimestamp_ = 0;
  Timestamp_ = UINT64_MAX;

  // Staging buffer reclaim treats an event without a fence as complete.
  if (Ctx) {
    auto *VkCtx = static_cast<CHIPContextVulkan *>(Ctx);
    if (auto *Dev = VkCtx->getVulkanDevice()) {
      Fence_ = Dev->acquireFence();
    }
  }
}

CHIPEventVulkan::~CHIPEventVulkan() {
  if (ChipContext_) {
    auto *VkCtx = static_cast<CHIPContextVulkan *>(ChipContext_);
    if (auto *Dev = VkCtx->getVulkanDevice()) {
      if (Fence_ != VK_NULL_HANDLE) {
        // A user event may be destroyed while its recording is pending.
        if (EventStatus_ == EVENT_STATUS_RECORDING)
          vkWaitForFences(Dev->getLogicalDevice(), 1, &Fence_, VK_TRUE,
                          UINT64_MAX);
        Dev->releaseFence(Fence_);
        Fence_ = VK_NULL_HANDLE;
      }
      if (TimestampSlot_ >= 0) {
        Dev->releaseTimestampSlot(TimestampSlot_);
        TimestampSlot_ = -1;
      }
      if (IpcSem_ != VK_NULL_HANDLE)
        vkDestroySemaphore(Dev->getLogicalDevice(), IpcSem_, nullptr);
    }
  }
  if (IpcTarget_)
    munmap(IpcTarget_, sizeof(uint64_t));
  if (IpcSemFd_ >= 0)
    close(IpcSemFd_);
  if (IpcShmFd_ >= 0)
    close(IpcShmFd_);
}

namespace {
// What hipIpcGetEventHandle stores in hipIpcEventHandle_t::reserved.
struct IpcEventHandleVulkan {
  uint32_t Magic;
  int32_t Pid;
  int32_t SemFd;
  int32_t ShmFd;
  uint8_t UUID[2 * VK_UUID_SIZE];
};
static_assert(sizeof(IpcEventHandleVulkan) <= HIP_IPC_HANDLE_SIZE);
constexpr uint32_t IpcEventMagic = 0x43565049; // "IPVC"
} // namespace

void CHIPEventVulkan::getIpcHandle(hipIpcEventHandle_t *Handle) {
  auto *Dev = static_cast<CHIPContextVulkan *>(ChipContext_)->getVulkanDevice();
  if (!Dev->hasIpcSemaphore())
    CHIPERR_LOG_AND_THROW("Vulkan device cannot export timeline semaphores",
                          hipErrorNotSupported);
  // Each step keeps its result only on success, so a retry resumes.
  if (IpcSemFd_ < 0) {
    // Records made before now are not signaled on the new semaphore.
    if (EventStatus_ == EVENT_STATUS_RECORDING)
      wait();
    if (IpcShmFd_ < 0) {
      int Fd = memfd_create("chipstar-ipc-event", MFD_CLOEXEC);
      if (Fd >= 0 && ftruncate(Fd, sizeof(uint64_t)) != 0) {
        close(Fd);
        Fd = -1;
      }
      if (Fd < 0)
        CHIPERR_LOG_AND_THROW("memfd_create failed", hipErrorOutOfMemory);
      IpcShmFd_ = Fd;
    }
    if (!IpcTarget_) {
      void *P = mmap(nullptr, sizeof(uint64_t), PROT_READ | PROT_WRITE,
                     MAP_SHARED, IpcShmFd_, 0);
      if (P == MAP_FAILED)
        CHIPERR_LOG_AND_THROW("mmap failed", hipErrorOutOfMemory);
      IpcTarget_ = static_cast<uint64_t *>(P);
    }
    if (IpcSem_ == VK_NULL_HANDLE)
      IpcSem_ = Dev->createIpcSemaphore(true);
    VkSemaphoreGetFdInfoKHR FdInfo{};
    FdInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    FdInfo.semaphore = IpcSem_;
    FdInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    int Fd = -1;
    checkVk(Dev->getSemaphoreFdFn()(Dev->getLogicalDevice(), &FdInfo, &Fd),
            "vkGetSemaphoreFdKHR failed", hipErrorOutOfMemory);
    IpcSemFd_ = Fd;
  }
  IpcEventHandleVulkan H{IpcEventMagic, static_cast<int32_t>(getpid()),
                         IpcSemFd_, IpcShmFd_, {}};
  std::memcpy(H.UUID, Dev->getIpcUUID(), sizeof(H.UUID));
  std::memset(Handle->reserved, 0, sizeof(Handle->reserved));
  std::memcpy(Handle->reserved, &H, sizeof(H));
}

void CHIPEventVulkan::openIpc(int SemFd, int ShmFd) {
  auto *Dev = static_cast<CHIPContextVulkan *>(ChipContext_)->getVulkanDevice();
  Flags_ = chipstar::EventFlags(hipEventDisableTiming | hipEventInterprocess);
  IpcOpened_ = true;
  Dev->releaseFence(Fence_);
  Fence_ = VK_NULL_HANDLE;
  void *P = mmap(nullptr, sizeof(uint64_t), PROT_READ, MAP_SHARED, ShmFd, 0);
  close(ShmFd);
  if (P == MAP_FAILED) {
    close(SemFd);
    CHIPERR_LOG_AND_THROW("mmap of the IPC event page failed",
                          hipErrorMapFailed);
  }
  IpcTarget_ = static_cast<uint64_t *>(P);
  IpcSem_ = Dev->createIpcSemaphore(false);
  VkImportSemaphoreFdInfoKHR Import{};
  Import.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR;
  Import.semaphore = IpcSem_;
  Import.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
  Import.fd = SemFd;
  VkResult R = Dev->importSemaphoreFdFn()(Dev->getLogicalDevice(), &Import);
  if (R != VK_SUCCESS)
    close(SemFd); // A failed import leaves the fd with the caller.
  checkVk(R, "vkImportSemaphoreFdKHR failed", hipErrorMapFailed);
  EventStatus_ = EVENT_STATUS_RECORDING;
}

VkSemaphore CHIPDeviceVulkan::createIpcSemaphore(bool Export) {
  VkExportSemaphoreCreateInfo ExportInfo{};
  ExportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
  ExportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
  VkSemaphoreTypeCreateInfo TypeInfo{};
  TypeInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  TypeInfo.pNext = Export ? &ExportInfo : nullptr;
  TypeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkSemaphoreCreateInfo CI{};
  CI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  CI.pNext = &TypeInfo;
  VkSemaphore Sem = VK_NULL_HANDLE;
  checkVk(vkCreateSemaphore(LogicalDevice_, &CI, nullptr, &Sem),
          "vkCreateSemaphore (IPC) failed", hipErrorOutOfMemory);
  return Sem;
}

chipstar::Event *
CHIPBackendVulkan::openIpcEvent(chipstar::Context *ChipCtx,
                                const hipIpcEventHandle_t &Handle) {
  IpcEventHandleVulkan H;
  std::memcpy(&H, Handle.reserved, sizeof(H));
  if (H.Magic != IpcEventMagic)
    CHIPERR_LOG_AND_THROW("Invalid IPC event handle", hipErrorInvalidValue);
  if (H.Pid == getpid())
    CHIPERR_LOG_AND_THROW("IPC event handle opened in its own process",
                          hipErrorInvalidContext);
  auto *Dev = static_cast<CHIPContextVulkan *>(ChipCtx)->getVulkanDevice();
  if (!Dev->hasIpcSemaphore())
    CHIPERR_LOG_AND_THROW("Vulkan device cannot import timeline semaphores",
                          hipErrorNotSupported);
  if (std::memcmp(H.UUID, Dev->getIpcUUID(), sizeof(H.UUID)) != 0)
    CHIPERR_LOG_AND_THROW("IPC event handle is from another device or driver",
                          hipErrorInvalidValue);

  // Copies the exporter's fds into this process.
  int PidFd = static_cast<int>(syscall(SYS_pidfd_open, H.Pid, 0));
  if (PidFd < 0)
    CHIPERR_LOG_AND_THROW("pidfd_open of the exporting process failed",
                          hipErrorMapFailed);
  int SemFd = static_cast<int>(syscall(SYS_pidfd_getfd, PidFd, H.SemFd, 0));
  int ShmFd = static_cast<int>(syscall(SYS_pidfd_getfd, PidFd, H.ShmFd, 0));
  close(PidFd);
  if (SemFd < 0 || ShmFd < 0) {
    if (SemFd >= 0)
      close(SemFd);
    if (ShmFd >= 0)
      close(ShmFd);
    CHIPERR_LOG_AND_THROW("pidfd_getfd failed (needs ptrace access to the "
                          "exporting process)",
                          hipErrorMapFailed);
  }
  std::unique_ptr<CHIPEventVulkan> Event(new CHIPEventVulkan(ChipCtx));
  Event->openIpc(SemFd, ShmFd);
  return Event.release();
}

bool CHIPEventVulkan::updateFinishStatus(bool ThrowErrorIfNotReady) {
  isDeletedSanityCheck();

  if (IpcOpened_) {
    auto *Dev =
        static_cast<CHIPContextVulkan *>(ChipContext_)->getVulkanDevice();
    uint64_t Value = 0;
    checkVk(vkGetSemaphoreCounterValue(Dev->getLogicalDevice(), IpcSem_,
                                       &Value),
            "vkGetSemaphoreCounterValue (IPC) failed", hipErrorTbd);
    bool Done = Value >= __atomic_load_n(IpcTarget_, __ATOMIC_ACQUIRE);
    EventStatus_ = Done ? EVENT_STATUS_RECORDED : EVENT_STATUS_RECORDING;
    if (!Done && ThrowErrorIfNotReady)
      CHIPERR_LOG_AND_THROW("chipstar::Event Not Ready", hipErrorNotReady);
    return Done;
  }

  // Only a RECORDING event can become RECORDED.
  if (EventStatus_ != EVENT_STATUS_RECORDING)
    return false;

  auto *VkCtx = static_cast<CHIPContextVulkan *>(ChipContext_);
  auto *Dev = VkCtx ? VkCtx->getVulkanDevice() : nullptr;
  if (!Dev || Fence_ == VK_NULL_HANDLE) {
    if (ThrowErrorIfNotReady)
      CHIPERR_LOG_AND_THROW("Vulkan event has no fence to query",
                            hipErrorNotReady);
    return false;
  }

  VkResult Status = vkGetFenceStatus(Dev->getLogicalDevice(), Fence_);
  if (Status == VK_SUCCESS) {
    EventStatus_ = EVENT_STATUS_RECORDED;
    HostTimestamp_ = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
    releaseDependencies();
    return true;
  }
  if (Status == VK_NOT_READY) {
    if (ThrowErrorIfNotReady)
      CHIPERR_LOG_AND_THROW("chipstar::Event Not Ready", hipErrorNotReady);
    return false;
  }
  CHIPERR_LOG_AND_THROW("vkGetFenceStatus returned a hard error",
                        hipErrorTbd);
}

bool CHIPEventVulkan::wait() {
  isDeletedSanityCheck();

  if (IpcOpened_) {
    auto *Dev =
        static_cast<CHIPContextVulkan *>(ChipContext_)->getVulkanDevice();
    uint64_t Target = __atomic_load_n(IpcTarget_, __ATOMIC_ACQUIRE);
    VkSemaphoreWaitInfo WI{};
    WI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    WI.semaphoreCount = 1;
    WI.pSemaphores = &IpcSem_;
    WI.pValues = &Target;
    checkVk(vkWaitSemaphores(Dev->getLogicalDevice(), &WI, UINT64_MAX),
            "vkWaitSemaphores (IPC) failed", hipErrorTbd);
    LOCK(EventMtx);
    EventStatus_ = EVENT_STATUS_RECORDED;
    return true;
  }

  if (EventStatus_ == EVENT_STATUS_RECORDED)
    return true;

  auto *VkCtx = static_cast<CHIPContextVulkan *>(ChipContext_);
  auto *Dev = VkCtx ? VkCtx->getVulkanDevice() : nullptr;
  if (!Dev || Fence_ == VK_NULL_HANDLE ||
      EventStatus_ == EVENT_STATUS_INIT) {
    // A never recorded event is complete; its fence would never signal.
    LOCK(EventMtx);
    EventStatus_ = EVENT_STATUS_RECORDED;
    return true;
  }

  VkResult Status =
      vkWaitForFences(Dev->getLogicalDevice(), 1, &Fence_, VK_TRUE, UINT64_MAX);
  if (Status != VK_SUCCESS) {
    CHIPERR_LOG_AND_THROW("vkWaitForFences failed", hipErrorTbd);
  }

  {
    LOCK(EventMtx); // chipstar::Event::EventStatus_
    EventStatus_ = EVENT_STATUS_RECORDED;
    HostTimestamp_ = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
  }
  releaseDependencies();
  return true;
}

float CHIPEventVulkan::getElapsedTime(chipstar::Event *OtherIn) {
  auto *Other = static_cast<CHIPEventVulkan *>(OtherIn);

  if (this->getContext() != Other->getContext())
    CHIPERR_LOG_AND_THROW(
        "Attempted to get elapsed time between two events that are not part "
        "of the same context",
        hipErrorTbd);

  if (this->getEventStatus() == EVENT_STATUS_RECORDING)
    this->updateFinishStatus(false);
  if (Other != this && Other->getEventStatus() == EVENT_STATUS_RECORDING)
    Other->updateFinishStatus(false);

  if (!this->isFinished() || !Other->isFinished())
    CHIPERR_LOG_AND_THROW("one of the events hasn't finished",
                          hipErrorNotReady);

  auto *VkCtx = static_cast<CHIPContextVulkan *>(ChipContext_);
  auto *Dev = VkCtx ? VkCtx->getVulkanDevice() : nullptr;

  // Raw ticks of a timestamp slot; false if there are none.
  auto readSlot = [&](int32_t Slot, uint64_t &OutTicks) -> bool {
    if (!Dev || Slot < 0 || Dev->getTimestampQueryPool() == VK_NULL_HANDLE)
      return false;
    uint64_t Raw = 0;
    VkResult R = vkGetQueryPoolResults(
        Dev->getLogicalDevice(), Dev->getTimestampQueryPool(),
        static_cast<uint32_t>(Slot), /*queryCount=*/1, sizeof(Raw), &Raw,
        sizeof(Raw),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (R != VK_SUCCESS)
      return false;
    OutTicks = Raw;
    return true;
  };

  uint64_t StartedNs = 0;
  uint64_t FinishedNs = 0;
  bool HaveGpuStart = false, HaveGpuEnd = false;

  if (Dev) {
    uint64_t StartTicks = 0, EndTicks = 0;
    const auto &Limits = Dev->getProperties().limits;
    // Nanoseconds per tick.
    const double Period = static_cast<double>(Limits.timestampPeriod);

    if (readSlot(this->TimestampSlot_, StartTicks)) {
      Timestamp_ = static_cast<uint64_t>(
          static_cast<double>(StartTicks) * Period);
      StartedNs = Timestamp_;
      HaveGpuStart = true;
    }
    if (readSlot(Other->TimestampSlot_, EndTicks)) {
      Other->Timestamp_ = static_cast<uint64_t>(
          static_cast<double>(EndTicks) * Period);
      FinishedNs = Other->Timestamp_;
      HaveGpuEnd = true;
    }
    if (HaveGpuStart && HaveGpuEnd) {
      // Ticks wrap at the queue's timestampValidBits.
      uint32_t Bits = Dev->getTimestampValidBits();
      uint64_t Mask = Bits >= 64 ? UINT64_MAX : (1ull << Bits) - 1;
      uint64_t D = (EndTicks - StartTicks) & Mask;
      bool Neg = D > Mask / 2;
      uint64_t Ns = static_cast<uint64_t>(
          static_cast<double>(Neg ? ((Mask - D) + 1) & Mask : D) * Period);
      StartedNs = Neg ? Ns : 0;
      FinishedNs = Neg ? 0 : Ns;
    }
  }

  // Without device timestamps, use the host times completion was observed.
  if (!HaveGpuStart || !HaveGpuEnd) {
    StartedNs = this->HostTimestamp_;
    FinishedNs = Other->HostTimestamp_;
  }

  // Whole and fractional milliseconds apart, to keep float precision.
  bool Reversed = false;
  uint64_t Begin = StartedNs;
  uint64_t End = FinishedNs;
  if (End < Begin) {
    Reversed = true;
    std::swap(Begin, End);
  }
  int64_t ElapsedNs = static_cast<int64_t>(End - Begin);

  constexpr int64_t NsPerSec = 1000000000;
  int64_t WholeMs = (ElapsedNs / NsPerSec) * 1000;
  int64_t FracNs = ElapsedNs % NsPerSec;
  float FracMs = static_cast<float>(FracNs) / 1000000.0f;
  float Ms = static_cast<float>(WholeMs) + FracMs;
  if (Reversed)
    Ms = -Ms;
  return Ms;
}

void CHIPEventVulkan::hostSignal() {
  isDeletedSanityCheck();
  // Vulkan cannot signal a fence from the host; no GPU work waits on these
  // events, so marking them RECORDED is enough.
  LOCK(EventMtx);
  EventStatus_ = EVENT_STATUS_RECORDED;
  HostTimestamp_ = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}
// ============================================================================
// CHIPCallbackDataVulkan
// ============================================================================
// The EventMonitor runs the callback once GpuReady completes, then signals
// DoneValue.

static thread_local bool InCallbackThread = false;

CHIPCallbackDataVulkan::CHIPCallbackDataVulkan(hipStreamCallback_t CallbackF,
                                                void *CallbackArgs,
                                                chipstar::Queue *ChipQueue)
    : chipstar::CallbackData(CallbackF, CallbackArgs, ChipQueue) {
  GpuReady = ChipQueue->enqueueMarkerImpl();
  // Later submits on the queue wait on the GPU until the callback has run.
  auto *QVk = static_cast<CHIPQueueVulkan *>(ChipQueue);
  DoneValue = QVk->reserveTimelineValue();
  if (GpuReady)
    GpuReady->Msg = "CallbackGpuReady";
}

// ============================================================================
// EventMonitorVulkan
// ============================================================================

EventMonitorVulkan::EventMonitorVulkan() = default;

EventMonitorVulkan::~EventMonitorVulkan() {
  // EventMonitor::stop() joins the thread.
}

/// Every 200us: update event status, run ready callbacks, drop finished events.
void EventMonitorVulkan::monitor() {
  logTrace("EventMonitorVulkan::monitor() started");
  while (true) {
    std::this_thread::sleep_for(std::chrono::microseconds(200));

    // try_to_lock: never stall a thread that is changing Events.
    {
      std::unique_lock<std::mutex> Lock(Backend->EventsMtx, std::try_to_lock);
      if (Lock.owns_lock()) {
        for (auto &Ev : Backend->Events) {
          if (!Ev)
            continue;
          if (Ev->getEventStatus() == EVENT_STATUS_RECORDED)
            continue;
          LOCK(Ev->EventMtx);
          Ev->updateFinishStatus(false);
        }
      }
    }

    // Every ready callback, not one per tick, which falls behind.
    {
      std::vector<chipstar::CallbackData *> ToExecute;
      {
        LOCK(Backend->CallbackQueueMtx);
        size_t Pending = Backend->CallbackQueue.size();
        for (size_t i = 0; i < Pending; ++i) {
          chipstar::CallbackData *CbData = Backend->CallbackQueue.front();
          Backend->CallbackQueue.pop();
          bool Ready = false;
          if (CbData && CbData->GpuReady) {
            LOCK(CbData->GpuReady->EventMtx);
            CbData->GpuReady->updateFinishStatus(false);
            Ready = (CbData->GpuReady->getEventStatus() ==
                     EVENT_STATUS_RECORDED);
          } else if (CbData) {
            Ready = true;
          }
          if (CbData && Ready) {
            ToExecute.push_back(CbData);
          } else if (CbData) {
            Backend->CallbackQueue.push(CbData);
          }
        }
      }
      // Outside the lock: callbacks may enqueue callbacks.
      for (auto *CbData : ToExecute) {
        InCallbackThread = true;
        CbData->execute(hipSuccess);
        InCallbackThread = false;
        auto *Q = static_cast<CHIPQueueVulkan *>(CbData->ChipQueue);
        Q->signalTimelineValue(
            static_cast<CHIPCallbackDataVulkan *>(CbData)->DoneValue);
        delete static_cast<CHIPCallbackDataVulkan *>(CbData);
      }
    }

    // Drop finished events nothing depends on, except user events.
    {
      std::unique_lock<std::mutex> Lock(Backend->EventsMtx, std::try_to_lock);
      if (Lock.owns_lock()) {
        auto NewEnd = std::remove_if(
            Backend->Events.begin(), Backend->Events.end(),
            [](const std::shared_ptr<chipstar::Event> &Ev) {
              if (!Ev)
                return true;
              if (Ev->isUserEvent())
                return false;
              if (Ev->getEventStatus() != EVENT_STATUS_RECORDED)
                return false;
              LOCK(Ev->DependsOnListMtx);
              return Ev->DependsOnList.empty();
            });
        Backend->Events.erase(NewEnd, Backend->Events.end());
      }
    }

    {
      LOCK(EventMonitorMtx); // chipstar::EventMonitor::Stop
      if (Stop) {
        logTrace("EventMonitorVulkan::monitor() exiting on Stop");
        return;
      }
    }
  }
}

// ============================================================================
// CHIPModuleVulkan
// ============================================================================

CHIPModuleVulkan::CHIPModuleVulkan(const SPVModule &SrcMod)
    : chipstar::Module(SrcMod) {}

CHIPModuleVulkan::~CHIPModuleVulkan() {
  if (ChipDevice_ == nullptr)
    return;
  VkDevice Dev = ChipDevice_->getLogicalDevice();
  if (Dev == VK_NULL_HANDLE)
    return;
  for (auto &Kv : Pipelines_)
    if (Kv.second != VK_NULL_HANDLE)
      vkDestroyPipeline(Dev, Kv.second, nullptr);
  for (auto &Kv : PipelineLayouts_)
    if (Kv.second != VK_NULL_HANDLE)
      vkDestroyPipelineLayout(Dev, Kv.second, nullptr);
  for (auto &Kv : DSLayouts_)
    if (Kv.second != VK_NULL_HANDLE)
      vkDestroyDescriptorSetLayout(Dev, Kv.second, nullptr);
  if (PipelineCache_ != VK_NULL_HANDLE)
    vkDestroyPipelineCache(Dev, PipelineCache_, nullptr);
  if (ShaderModule_ != VK_NULL_HANDLE)
    vkDestroyShaderModule(Dev, ShaderModule_, nullptr);
}

/// The module-cache artifact is this module's VkPipelineCache data, keyed by
/// the SPIR-V, the device and driver, the loaded libraries and the compiler
/// environment.
void CHIPModuleVulkan::createModulePipelineCache(std::string_view Spv) {
  namespace cache = chipstar::cache;
  VkDevice Dev = ChipDevice_->getLogicalDevice();
  std::string Key;
  cache::Entry Hit;
  if (ChipEnvVars.getModuleCacheDir().has_value()) {
    const VkPhysicalDeviceProperties &P = ChipDevice_->getProperties();
    cache::KeyBuilder KB;
    KB.add(cache::KeyField::BackendTag, "vulkan")
        .add(cache::KeyField::Il, Spv)
        .add(cache::KeyField::BuildOptions, ChipEnvVars.getJitFlagsOverride())
        .add(cache::KeyField::DeviceName, std::string_view(P.deviceName))
        .add(cache::KeyField::DriverVersion, uint64_t(P.driverVersion))
        .add(cache::KeyField::VendorId, uint64_t(P.vendorID))
        .add(cache::KeyField::DeviceId, uint64_t(P.deviceID))
        .add(cache::KeyField::LoaderDelta, cache::loaderDeltaDigest())
        .add(cache::KeyField::Environment,
             collectCompilerEnvironmentVariables());
    Key = KB.finish();
    Hit = cache::load(ChipEnvVars.getModuleCacheDir().value(), "vulkan", Key);
  }
  VkPipelineCacheCreateInfo Ci{};
  Ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
  if (Hit) {
    Ci.initialDataSize = Hit.data().size();
    Ci.pInitialData = Hit.data().data();
    if (vkCreatePipelineCache(Dev, &Ci, nullptr, &PipelineCache_) ==
        VK_SUCCESS) {
      cache::logOutcome("vulkan", Key, cache::Outcome::Hit, "");
      return;
    }
    cache::logOutcome("vulkan", Key, cache::Outcome::Rejected,
                      "pipeline-cache");
    Ci.initialDataSize = 0;
    Ci.pInitialData = nullptr;
  }
  if (vkCreatePipelineCache(Dev, &Ci, nullptr, &PipelineCache_) != VK_SUCCESS)
    PipelineCache_ = VK_NULL_HANDLE;
  PendingCacheKey_ = Key;
}

/// Store the pipeline cache once the first pipeline has been compiled.
void CHIPModuleVulkan::storeModulePipelineCache() {
  if (PendingCacheKey_.empty() || PipelineCache_ == VK_NULL_HANDLE)
    return;
  VkDevice Dev = ChipDevice_->getLogicalDevice();
  size_t Size = 0;
  std::vector<uint8_t> Data;
  if (vkGetPipelineCacheData(Dev, PipelineCache_, &Size, nullptr) ==
          VK_SUCCESS &&
      Size > 0) {
    Data.resize(Size);
    if (vkGetPipelineCacheData(Dev, PipelineCache_, &Size, Data.data()) ==
        VK_SUCCESS)
      chipstar::cache::store(ChipEnvVars.getModuleCacheDir().value(), "vulkan",
                             PendingCacheKey_, Data.data(), Size);
  }
  PendingCacheKey_.clear();
}

// Gives each ChipVulkanDynSharedName variable its own array type whose length
// is a new specialization constant, decorated SpecId and holding the size in
// bytes, divided by its element size, and fills StaticBytes with each entry point's other workgroup
// memory and ElemBytes with its dynamic array's element size. Returns the
// largest element size, or 0 if the module has none.
static uint32_t
specializeDynamicShared(std::vector<uint32_t> &W,
                        std::unordered_map<std::string, uint32_t> &StaticBytes,
                        std::unordered_map<std::string, uint32_t> &ElemBytes,
                        uint32_t &SpecId) {
  std::unordered_map<uint32_t, std::string> Names;
  std::unordered_map<uint32_t, uint32_t> Pointee, Constants, ConstType, Size,
      Align, IntWidth;
  std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> Arrays;
  std::unordered_map<uint32_t, uint32_t> DynPtrs; // pointer type -> new one
  std::unordered_map<uint32_t, uint32_t> SharedVars; // static var -> bytes
  std::unordered_map<uint32_t, uint32_t> DynVars;    // var -> element bytes
  std::vector<std::pair<std::string, std::vector<uint32_t>>> Entries;
  auto Valid = [&](size_t I) {
    return (W[I] >> 16) != 0 && I + (W[I] >> 16) <= W.size();
  };
  if (W.size() < 5)
    return 0;
  SpecId = 3; // 0, 1, 2 are the workgroup size.
  for (size_t I = 5; I < W.size(); I += W[I] >> 16) {
    if (!Valid(I))
      return 0;
    uint32_t Op = W[I] & 0xffff, Wc = W[I] >> 16;
    const uint32_t *A = &W[I + 1];
    if (Op == 15 /*OpEntryPoint*/ && Wc >= 4) {
      std::string Name(reinterpret_cast<const char *>(&A[2]),
                       strnlen(reinterpret_cast<const char *>(&A[2]),
                               (Wc - 3) * 4));
      size_t Words = Name.size() / 4 + 1;
      Entries.push_back({Name, std::vector<uint32_t>(A + 2 + Words, A + Wc - 1)});
    } else if (Op == 5 /*OpName*/ && Wc >= 3)
      Names[A[0]] = std::string(reinterpret_cast<const char *>(&A[1]),
                                strnlen(reinterpret_cast<const char *>(&A[1]),
                                        (Wc - 2) * 4));
    else if (Op == 71 /*OpDecorate*/ && Wc >= 4 && A[1] == 1 /*SpecId*/)
      SpecId = std::max(SpecId, A[2] + 1);
    else if ((Op == 21 /*OpTypeInt*/ || Op == 22 /*OpTypeFloat*/) && Wc >= 3) {
      Size[A[0]] = Align[A[0]] = A[1] / 8;
      if (Op == 21)
        IntWidth[A[0]] = A[1];
    }
    else if (Op == 23 /*OpTypeVector*/ && Wc >= 4) {
      Size[A[0]] = Size[A[1]] * A[2];
      Align[A[0]] = Size[A[1]] * (A[2] == 3 ? 4 : A[2]);
    } else if (Op == 28 /*OpTypeArray*/ && Wc >= 4) {
      Arrays[A[0]] = {A[1], A[2]};
      uint32_t EA = std::max(Align[A[1]], 1u);
      Size[A[0]] = (Size[A[1]] + EA - 1) / EA * EA * Constants[A[2]];
      Align[A[0]] = EA;
    } else if (Op == 30 /*OpTypeStruct*/) { // Natural alignment.
      uint32_t &S = Size[A[0]], &Al = Align[A[0]];
      Al = 1;
      for (uint32_t M = 1; M + 1 < Wc; ++M) {
        uint32_t MA = std::max(Align[A[M]], 1u);
        S = (S + MA - 1) / MA * MA + Size[A[M]];
        Al = std::max(Al, MA);
      }
      S = (S + Al - 1) / Al * Al;
    }
    else if (Op == 32 /*OpTypePointer*/ && Wc >= 4)
      Pointee[A[0]] = A[2];
    else if (Op == 43 /*OpConstant*/ && Wc >= 4) {
      Constants[A[1]] = A[2];
      ConstType[A[1]] = A[0];
    } else if (Op == 59 /*OpVariable*/ && Wc >= 4 && A[2] == 4 /*Workgroup*/) {
      auto Arr = Arrays.find(Pointee[A[0]]);
      if (Names[A[1]].rfind(ChipVulkanDynSharedName, 0) == 0 &&
          Arr != Arrays.end() &&
          IntWidth[ConstType[Arr->second.second]] == 32 &&
          Constants[Arr->second.second] != 0 &&
          ChipVulkanDynSharedBytes % Constants[Arr->second.second] == 0) {
        DynPtrs[A[0]] = 0;
        DynVars[A[1]] =
            ChipVulkanDynSharedBytes / Constants[Arr->second.second];
      } else // Rounded up: the driver may pad between variables.
        SharedVars[A[1]] = (Size[Pointee[A[0]]] + 15) / 16 * 16;
    }
  }
  for (auto &[Name, Interface] : Entries)
    for (uint32_t Id : Interface) {
      if (auto It = SharedVars.find(Id); It != SharedVars.end())
        StaticBytes[Name] += It->second;
      if (auto It = DynVars.find(Id); It != DynVars.end())
        ElemBytes[Name] = It->second;
    }
  if (DynPtrs.empty())
    return 0;

  std::vector<uint32_t> Out(W.begin(), W.begin() + 5);
  uint32_t Bound = W[3], Bytes = Bound++, MaxElem = 0;
  bool Decorated = false;
  for (size_t I = 5; I < W.size(); I += W[I] >> 16) {
    uint32_t Op = W[I] & 0xffff, Wc = W[I] >> 16;
    // First annotation, type, constant or global variable.
    bool Section = (Op >= 71 && Op <= 75) || Op == 332 ||
                   (Op >= 19 && Op <= 52) || Op == 59;
    if (Section && !Decorated) {
      Out.insert(Out.end(), {(4u << 16) | 71, Bytes, 1 /*SpecId*/, SpecId});
      Decorated = true;
    }
    std::vector<uint32_t> Inst(W.begin() + I, W.begin() + I + Wc);
    if (Op == 59 /*OpVariable*/ && DynPtrs.count(Inst[1]) &&
        Names[Inst[2]].rfind(ChipVulkanDynSharedName, 0) == 0)
      Inst[1] = DynPtrs[Inst[1]];
    Out.insert(Out.end(), Inst.begin(), Inst.end());
    auto Dyn = DynPtrs.find(Op == 32 /*OpTypePointer*/ ? Inst[1] : 0);
    if (Dyn == DynPtrs.end())
      continue;
    auto [Elem, LenId] = Arrays[Pointee[Inst[1]]];
    uint32_t UInt = ConstType[LenId];
    uint32_t ElemBytes = ChipVulkanDynSharedBytes / Constants[LenId];
    if (MaxElem == 0)
      Out.insert(Out.end(), {(4u << 16) | 50 /*OpSpecConstant*/, UInt, Bytes,
                             ChipVulkanDynSharedBytes});
    uint32_t Len = Bytes;
    if (ElemBytes != 1) {
      uint32_t ElemId = Bound++;
      Len = Bound++;
      Out.insert(Out.end(), {(4u << 16) | 43 /*OpConstant*/, UInt, ElemId,
                             ElemBytes, (6u << 16) | 52 /*OpSpecConstantOp*/,
                             UInt, Len, 134 /*OpUDiv*/, Bytes, ElemId});
    }
    uint32_t Arr = Bound++;
    Dyn->second = Bound++;
    Out.insert(Out.end(), {(4u << 16) | 28 /*OpTypeArray*/, Arr, Elem, Len,
                           (4u << 16) | 32 /*OpTypePointer*/, Dyn->second,
                           4 /*Workgroup*/, Arr});
    MaxElem = std::max(MaxElem, ElemBytes);
  }
  if (MaxElem == 0)
    return 0;
  Out[3] = Bound;
  W = std::move(Out);
  return MaxElem;
}

// Removes the float-controls execution modes the device does not support,
// which must not be used (VUID-RuntimeSpirv-shaderDenormPreserveFloat32-06297).
static void
dropUnsupportedFloatModes(std::vector<uint32_t> &W,
                          const VkPhysicalDeviceFloatControlsProperties &P) {
  const std::unordered_map<uint32_t, std::array<VkBool32, 3>> Modes = {
      {4459 /*DenormPreserve*/,
       {P.shaderDenormPreserveFloat16, P.shaderDenormPreserveFloat32,
        P.shaderDenormPreserveFloat64}},
      {4460 /*DenormFlushToZero*/,
       {P.shaderDenormFlushToZeroFloat16, P.shaderDenormFlushToZeroFloat32,
        P.shaderDenormFlushToZeroFloat64}},
      {4461 /*SignedZeroInfNanPreserve*/,
       {P.shaderSignedZeroInfNanPreserveFloat16,
        P.shaderSignedZeroInfNanPreserveFloat32,
        P.shaderSignedZeroInfNanPreserveFloat64}}};
  auto Dropped = [&](size_t I) {
    auto M = (W[I] >> 16) == 4 && (W[I] & 0xffff) == 16 /*OpExecutionMode*/
                 ? Modes.find(W[I + 2])
                 : Modes.end();
    uint32_t Width = M != Modes.end() ? W[I + 3] : 0;
    return M != Modes.end() && (Width == 16 || Width == 32 || Width == 64) &&
           !M->second[Width == 16 ? 0 : Width == 32 ? 1 : 2];
  };
  // Each mode's capability is the mode's value plus 5.
  std::unordered_set<uint32_t> KeptCaps;
  for (size_t I = 5; I < W.size(); I += W[I] >> 16) {
    if ((W[I] >> 16) == 0 || I + (W[I] >> 16) > W.size())
      return;
    if ((W[I] & 0xffff) == 16 && (W[I] >> 16) >= 3 && !Dropped(I))
      KeptCaps.insert(W[I + 2] + 5);
  }
  std::vector<uint32_t> Out(W.begin(), W.begin() + std::min<size_t>(5, W.size()));
  for (size_t I = 5; I < W.size(); I += W[I] >> 16) {
    uint32_t Op = W[I] & 0xffff, Wc = W[I] >> 16;
    bool UnusedCap = Op == 17 /*OpCapability*/ && Wc == 2 &&
                     Modes.count(W[I + 1] - 5) && !KeptCaps.count(W[I + 1]);
    if (Dropped(I) || UnusedCap)
      continue;
    Out.insert(Out.end(), W.begin() + I, W.begin() + I + Wc);
  }
  W = std::move(Out);
}

void CHIPModuleVulkan::compile(chipstar::Device *ChipDev) {
  logTrace("CHIPModuleVulkan::compile()");
  std::lock_guard<std::mutex> Lock(ModuleMtx_);
  ChipDevice_ = static_cast<CHIPDeviceVulkan *>(ChipDev);
  if (ChipDevice_ == nullptr)
    CHIPERR_LOG_AND_THROW("CHIPModuleVulkan::compile called with null device",
                          hipErrorInitializationError);

  VkDevice Dev = ChipDevice_->getLogicalDevice();
  if (Dev == VK_NULL_HANDLE)
    CHIPERR_LOG_AND_THROW("CHIPModuleVulkan::compile: logical device not "
                          "initialized",
                          hipErrorInitializationError);

  auto SrcBin = Src_->getBinary();
  const size_t SizeBytes = SrcBin.size();
  if (SizeBytes == 0 || (SizeBytes % sizeof(uint32_t)) != 0)
    CHIPERR_LOG_AND_THROW("CHIPModuleVulkan::compile: SPV binary size invalid",
                          hipErrorInvalidImage);

  createModulePipelineCache(SrcBin);
  if (!getInfo().BuildError.empty())
    CHIPERR_LOG_AND_THROW("Device compiler errors:\n" + getInfo().BuildError,
                          hipErrorSharedObjectInitFailed);

  // A function without a body is a device symbol nobody defined; the SPIR-V
  // preprocessing already logged its name.
  {
    const auto *W = reinterpret_cast<const uint32_t *>(SrcBin.data());
    size_t N = SizeBytes / sizeof(uint32_t);
    bool InFn = false, HasBody = false;
    for (size_t I = 5; I < N;) {
      uint32_t Op = W[I] & 0xffff, Wc = W[I] >> 16;
      if (Wc == 0)
        break;
      if (Op == 54 /*OpFunction*/) {
        InFn = true;
        HasBody = false;
      } else if (Op == 248 /*OpLabel*/) {
        HasBody = true;
      } else if (Op == 56 /*OpFunctionEnd*/ && InFn) {
        if (!HasBody)
          CHIPERR_LOG_AND_THROW("CHIPModuleVulkan::compile: module calls "
                                "undefined device functions",
                                hipErrorSharedObjectInitFailed);
        InFn = false;
      }
      I += Wc;
    }
  }

  // One shader module for every entry point.
  {
    std::vector<uint32_t> Words(SizeBytes / sizeof(uint32_t));
    std::memcpy(Words.data(), SrcBin.data(), SizeBytes);
    DynSharedElemBytes_ = specializeDynamicShared(Words, StaticSharedBytes_,
                                                  DynElemBytes_, DynSpecId_);
    dropUnsupportedFloatModes(Words, ChipDevice_->getFloatControls());
    // WORKAROUND(CHIP-SPV/chipStar#1817, no public NVIDIA tracker): the NVIDIA
    // driver gets wrong results from calls to DontInline functions; the hint is
    // optional. Remove when NVIDIA drivers compute them correctly.
    for (size_t I = 5; ChipDevice_->getProperties().vendorID == 0x10de &&
                       I < Words.size() && (Words[I] >> 16);
         I += Words[I] >> 16)
      if ((Words[I] & 0xffff) == 54 /*OpFunction*/ && (Words[I] >> 16) >= 5)
        Words[I + 3] &= ~2u /*DontInline*/;
    VkShaderModuleCreateInfo Ci{};
    Ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    Ci.codeSize = Words.size() * sizeof(uint32_t);
    Ci.pCode = Words.data();
    VkResult R =
        vkCreateShaderModule(Dev, &Ci, /*pAlloc=*/nullptr, &ShaderModule_);
    if (R != VK_SUCCESS || ShaderModule_ == VK_NULL_HANDLE)
      CHIPERR_LOG_AND_THROW(
          "CHIPModuleVulkan::compile: vkCreateShaderModule failed",
          hipErrorInitializationError);
  }

  // Per-kernel argument bindings, from the reflection.
  const SPVModuleInfo &Info = getInfo();
  for (auto &Kv : Info.FuncInfoMap) {
    const std::string &Name = Kv.first;
    const SPVFuncInfo *FInfo = Kv.second.get();
    if (FInfo == nullptr)
      continue;

    VulkanKernelReflection Refl;
    Refl.Name = Name;
    uint32_t NextBinding = 0;
    uint32_t PCRunningOffset = 0;
    uint32_t MaxPCEnd = 0;
    // Position in Args_[]; device global arguments are hidden and take none.
    int32_t HipSrcIdx = 0;
    Refl.PodBufferBinding = FInfo->ArgsBinding;
    if (FInfo->ArgsBinding >= 0)
      Refl.MaxDescriptorBinding = FInfo->ArgsBinding;
    FInfo->visitKernelArgs([&](const SPVFuncInfo::KernelArg &A) {
      const uint32_t Ord = static_cast<uint32_t>(A.Index);
      switch (A.Kind) {
      case SPVTypeKind::Pointer:
      case SPVTypeKind::DeviceGlobal: {
        VulkanStorageBufferArg Buf;
        Buf.Ordinal = Ord;
        Buf.Binding = A.Binding >= 0 ? static_cast<uint32_t>(A.Binding)
                                     : NextBinding;
        NextBinding = std::max(NextBinding, Buf.Binding + 1);
        if (A.Kind == SPVTypeKind::Pointer)
          Buf.HipSourceIndex = HipSrcIdx++;
        if (A.DevGlobalName.rfind(ChipArgFieldPrefix, 0) == 0)
          sscanf(A.DevGlobalName.c_str() + strlen(ChipArgFieldPrefix), "%d_%u",
                 &Buf.FieldArg, &Buf.FieldOffset);
        else
          Buf.DevGlobalName = A.DevGlobalName;
        Buf.PCOffset = A.PushConstOffset;
        if (A.PushConstOffset >= 0)
          MaxPCEnd = std::max<uint32_t>(MaxPCEnd, A.PushConstOffset + 8);
        Refl.Buffers.push_back(Buf);
        if (Buf.Binding > Refl.MaxDescriptorBinding)
          Refl.MaxDescriptorBinding = Buf.Binding;
        break;
      }
      case SPVTypeKind::POD: {
        VulkanPushConstantArg Pc;
        Pc.Ordinal = Ord;
        Pc.Offset = A.PushConstOffset >= 0
                        ? static_cast<uint32_t>(A.PushConstOffset)
                        : PCRunningOffset;
        Pc.Size = static_cast<uint32_t>(A.Size);
        PCRunningOffset += Pc.Size;
        if (Pc.Offset + Pc.Size > MaxPCEnd)
          MaxPCEnd = Pc.Offset + Pc.Size;
        Pc.HipSourceIndex = HipSrcIdx++;
        Refl.PushConst.push_back(Pc);
        break;
      }
      case SPVTypeKind::Image:
      case SPVTypeKind::Sampler:
        CHIPERR_LOG_AND_THROW("CHIPModuleVulkan::compile: image/sampler "
                              "kernel arg unsupported on Vulkan",
                              hipErrorNotSupported);
      case SPVTypeKind::PODByRef:
        CHIPERR_LOG_AND_THROW("CHIPModuleVulkan::compile: PODByRef arg "
                              "(spilled) unsupported on Vulkan",
                              hipErrorNotSupported);
      default:
        break;
      }
    });
    // Vulkan requires push-constant ranges to be 4-byte aligned.
    Refl.PushConstantBlockSize = (MaxPCEnd + 3u) & ~3u;

    Reflection_.emplace(Name, std::move(Refl));
  }

  for (auto &Kv : Info.FuncInfoMap) {
    const std::string &HostFName = Kv.first;
    SPVFuncInfo *FuncInfo = findFunctionInfo(HostFName);
    if (FuncInfo == nullptr)
      continue;
    CHIPKernelVulkan *KernelPtr =
        new CHIPKernelVulkan(HostFName, FuncInfo, this);
    KernelPtr->bindReflection(getReflection(HostFName));
    addKernel(KernelPtr);
  }

  logTrace("CHIPModuleVulkan::compile: {} kernels, shader-module={}",
           Reflection_.size(),
           reinterpret_cast<void *>(ShaderModule_));
}

const VulkanKernelReflection *
CHIPModuleVulkan::getReflection(const std::string &Name) const {
  auto It = Reflection_.find(Name);
  return It == Reflection_.end() ? nullptr : &It->second;
}

namespace {

// Free functions, so callers holding ModuleMtx_ need not lock it again.
inline VkDescriptorSetLayout
buildDescriptorSetLayout(VkDevice Dev, const VulkanKernelReflection &Refl) {
  std::vector<VkDescriptorSetLayoutBinding> Bindings;
  Bindings.reserve(Refl.Buffers.size());
  for (const auto &B : Refl.Buffers) {
    VkDescriptorSetLayoutBinding LB{};
    LB.binding = B.Binding;
    LB.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    LB.descriptorCount = 1;
    LB.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    LB.pImmutableSamplers = nullptr;
    Bindings.push_back(LB);
  }
  if (Refl.PodBufferBinding >= 0) {
    VkDescriptorSetLayoutBinding LB{};
    LB.binding = static_cast<uint32_t>(Refl.PodBufferBinding);
    LB.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    LB.descriptorCount = 1;
    LB.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    Bindings.push_back(LB);
  }
  VkDescriptorSetLayoutCreateInfo Ci{};
  Ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  Ci.bindingCount = static_cast<uint32_t>(Bindings.size());
  Ci.pBindings = Bindings.empty() ? nullptr : Bindings.data();
  VkDescriptorSetLayout Layout = VK_NULL_HANDLE;
  VkResult R =
      vkCreateDescriptorSetLayout(Dev, &Ci, /*pAlloc=*/nullptr, &Layout);
  if (R != VK_SUCCESS || Layout == VK_NULL_HANDLE)
    CHIPERR_LOG_AND_THROW("vkCreateDescriptorSetLayout failed",
                          hipErrorInitializationError);
  return Layout;
}

inline VkPipelineLayout buildPipelineLayout(VkDevice Dev,
                                            VkDescriptorSetLayout DSL,
                                            uint32_t PCSize) {
  VkPushConstantRange PCRange{};
  PCRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  PCRange.offset = 0;
  PCRange.size = PCSize;
  VkPipelineLayoutCreateInfo Ci{};
  Ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  Ci.setLayoutCount = 1;
  Ci.pSetLayouts = &DSL;
  Ci.pushConstantRangeCount = (PCSize > 0) ? 1u : 0u;
  Ci.pPushConstantRanges = (PCSize > 0) ? &PCRange : nullptr;
  VkPipelineLayout Layout = VK_NULL_HANDLE;
  VkResult R = vkCreatePipelineLayout(Dev, &Ci, /*pAlloc=*/nullptr, &Layout);
  if (R != VK_SUCCESS || Layout == VK_NULL_HANDLE)
    CHIPERR_LOG_AND_THROW("vkCreatePipelineLayout failed",
                          hipErrorInitializationError);
  return Layout;
}

} // namespace

VkDescriptorSetLayout
CHIPModuleVulkan::getOrCreateDescriptorSetLayout(const std::string &KernelName) {
  std::lock_guard<std::mutex> Lock(ModuleMtx_);
  auto It = DSLayouts_.find(KernelName);
  if (It != DSLayouts_.end())
    return It->second;

  const VulkanKernelReflection *Refl = getReflection(KernelName);
  if (Refl == nullptr)
    CHIPERR_LOG_AND_THROW(
        "CHIPModuleVulkan::getOrCreateDescriptorSetLayout: no reflection",
        hipErrorInvalidImage);
  if (ChipDevice_ == nullptr)
    CHIPERR_LOG_AND_THROW(
        "CHIPModuleVulkan::getOrCreateDescriptorSetLayout: device unbound",
        hipErrorInitializationError);

  VkDescriptorSetLayout Layout =
      buildDescriptorSetLayout(ChipDevice_->getLogicalDevice(), *Refl);
  DSLayouts_[KernelName] = Layout;
  return Layout;
}

VkPipelineLayout
CHIPModuleVulkan::getOrCreatePipelineLayout(const std::string &KernelName) {
  VkDescriptorSetLayout DSL = getOrCreateDescriptorSetLayout(KernelName);
  std::lock_guard<std::mutex> Lock(ModuleMtx_);
  auto It = PipelineLayouts_.find(KernelName);
  if (It != PipelineLayouts_.end())
    return It->second;

  const VulkanKernelReflection *Refl = getReflection(KernelName);
  if (Refl == nullptr)
    CHIPERR_LOG_AND_THROW(
        "CHIPModuleVulkan::getOrCreatePipelineLayout: no reflection",
        hipErrorInvalidImage);
  if (ChipDevice_ == nullptr)
    CHIPERR_LOG_AND_THROW(
        "CHIPModuleVulkan::getOrCreatePipelineLayout: device unbound",
        hipErrorInitializationError);
  VkPipelineLayout Layout = buildPipelineLayout(
      ChipDevice_->getLogicalDevice(), DSL, Refl->pushConstantRangeSize());
  PipelineLayouts_[KernelName] = Layout;
  return Layout;
}

size_t
CHIPModuleVulkan::maxDynamicSharedBytes(const std::string &Kernel) const {
  size_t Limit = ChipDevice_
                     ? ChipDevice_->getAttr(
                           hipDeviceAttributeMaxSharedMemoryPerBlock)
                     : 0;
  size_t Static = getStaticSharedBytes(Kernel);
  auto It = DynElemBytes_.find(Kernel);
  size_t Elem = It == DynElemBytes_.end() ? 1 : It->second;
  size_t Free = Static > Limit ? 0 : (Limit - Static) / Elem * Elem;
  // Launches allocate at least the module's widest dynamic element.
  return It != DynElemBytes_.end() && Free < DynSharedElemBytes_ ? 0 : Free;
}

VkPipeline CHIPModuleVulkan::getOrCreatePipeline(const std::string &KernelName,
                                                  dim3 BlockDim,
                                                  size_t DynSharedBytes) {
  // Whole elements of this kernel's array, and at least the module's widest
  // element so that every dynamic array type keeps a nonzero length.
  auto Elem = DynElemBytes_.find(KernelName);
  size_t E = Elem == DynElemBytes_.end() ? 1 : Elem->second;
  size_t Max = maxDynamicSharedBytes(KernelName);
  size_t Shared = 0;
  if (Elem == DynElemBytes_.end())
    Shared = DynSharedElemBytes_; // Unused here: one pipeline per block size.
  else if (DynSharedBytes <= Max)
    Shared = std::max<size_t>((std::max<size_t>(DynSharedBytes, 1) + E - 1) /
                                  E * E,
                              DynSharedElemBytes_);
  if (DynSharedBytes > Max || (Elem != DynElemBytes_.end() && Shared > Max) ||
      getStaticSharedBytes(KernelName) >
          static_cast<size_t>(ChipDevice_->getAttr(
              hipDeviceAttributeMaxSharedMemoryPerBlock)))
    CHIPERR_LOG_AND_THROW("CHIPModuleVulkan::getOrCreatePipeline: shared "
                          "memory does not fit the device limit",
                          hipErrorInvalidValue);
  const std::string CacheKey = KernelName + ":" + std::to_string(BlockDim.x) +
                               "x" + std::to_string(BlockDim.y) + "x" +
                               std::to_string(BlockDim.z) + ":" +
                               std::to_string(Shared);
  VkPipelineLayout PLayout = getOrCreatePipelineLayout(KernelName);
  std::lock_guard<std::mutex> Lock(ModuleMtx_);
  auto It = Pipelines_.find(CacheKey);
  if (It != Pipelines_.end())
    return It->second;

  if (ChipDevice_ == nullptr || ShaderModule_ == VK_NULL_HANDLE)
    CHIPERR_LOG_AND_THROW(
        "CHIPModuleVulkan::getOrCreatePipeline: module not compiled",
        hipErrorInitializationError);
  VkDevice Dev = ChipDevice_->getLogicalDevice();

  // The compiler declares the workgroup size as LocalSizeId of
  // specialization constants 0, 1, 2; fill them with the block size.
  struct SpecData {
    uint32_t X, Y, Z, Shared;
  } Spec{BlockDim.x, BlockDim.y, BlockDim.z, static_cast<uint32_t>(Shared)};
  VkSpecializationMapEntry Entries[4] = {
      {0, offsetof(SpecData, X), sizeof(uint32_t)},
      {1, offsetof(SpecData, Y), sizeof(uint32_t)},
      {2, offsetof(SpecData, Z), sizeof(uint32_t)},
      {DynSpecId_, offsetof(SpecData, Shared), sizeof(uint32_t)},
  };
  VkSpecializationInfo SpecInfo{};
  SpecInfo.mapEntryCount = DynSharedElemBytes_ != 0 ? 4 : 3;
  SpecInfo.pMapEntries = Entries;
  SpecInfo.dataSize = sizeof(SpecData);
  SpecInfo.pData = &Spec;

  VkPipelineShaderStageCreateInfo Stage{};
  Stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  Stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  Stage.module = ShaderModule_;
  Stage.pName = KernelName.c_str();
  Stage.pSpecializationInfo = &SpecInfo;

  VkComputePipelineCreateInfo Ci{};
  Ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  Ci.stage = Stage;
  Ci.layout = PLayout;
  Ci.basePipelineHandle = VK_NULL_HANDLE;
  Ci.basePipelineIndex = -1;

  VkPipeline Pipeline = VK_NULL_HANDLE;
  VkResult R = vkCreateComputePipelines(
      Dev,
      PipelineCache_ != VK_NULL_HANDLE ? PipelineCache_
                                       : ChipDevice_->getPipelineCache(),
      /*createInfoCount=*/1, &Ci, /*pAlloc=*/nullptr, &Pipeline);
  if (R != VK_SUCCESS || Pipeline == VK_NULL_HANDLE)
    CHIPERR_LOG_AND_THROW("vkCreateComputePipelines failed (VkResult=" +
                              std::to_string(static_cast<int>(R)) +
                              " kernel=" + KernelName + ")",
                          hipErrorInitializationError);
  Pipelines_[CacheKey] = Pipeline;
  storeModulePipelineCache();
  return Pipeline;
}

// ============================================================================
// CHIPKernelVulkan
// ============================================================================

CHIPKernelVulkan::CHIPKernelVulkan(std::string HostFName, SPVFuncInfo *FuncInfo,
                                   CHIPModuleVulkan *Parent)
    : chipstar::Kernel(std::move(HostFName), FuncInfo), Module_(Parent) {
}

CHIPKernelVulkan::~CHIPKernelVulkan() = default;

hipError_t CHIPKernelVulkan::getAttributes(hipFuncAttributes *Attr) {
  chipstar::Device *Dev = ::Backend->getActiveDevice();
  *Attr = hipFuncAttributes{};
  Attr->binaryVersion = 10;
  Attr->ptxVersion = 10;
  Attr->maxThreadsPerBlock = Dev->getAttr(hipDeviceAttributeMaxThreadsPerBlock);
  Attr->sharedSizeBytes = Module_->getStaticSharedBytes(getName());
  Attr->maxDynamicSharedSizeBytes = static_cast<int>(
      Module_->maxDynamicSharedBytes(getName()));
  return hipSuccess;
}

chipstar::Module *CHIPKernelVulkan::getModule() { return Module_; }

const chipstar::Module *CHIPKernelVulkan::getModule() const { return Module_; }

// ============================================================================
// CHIPExecItemVulkan
// ============================================================================

CHIPExecItemVulkan::CHIPExecItemVulkan(dim3 GridDim, dim3 BlockDim,
                                       size_t SharedMem, hipStream_t ChipQueue)
    : chipstar::ExecItem(GridDim, BlockDim, SharedMem, ChipQueue) {}

CHIPExecItemVulkan::CHIPExecItemVulkan(const CHIPExecItemVulkan &Other)
    : CHIPExecItemVulkan(Other.GridDim_, Other.BlockDim_, Other.SharedMem_,
                         Other.ChipQueue_) {
  ChipKernel_ = Other.ChipKernel_;
  this->ArgsSetup = false;
  this->Args_ = Other.Args_;
}

CHIPExecItemVulkan::~CHIPExecItemVulkan() = default;

chipstar::ExecItem *CHIPExecItemVulkan::clone() const {
  return new CHIPExecItemVulkan(*this);
}

void CHIPExecItemVulkan::setKernel(chipstar::Kernel *Kernel) {
  assert(Kernel && "Kernel is nullptr!");
  ChipKernel_ = static_cast<CHIPKernelVulkan *>(Kernel);
  this->ArgsSetup = false;
  const auto *Refl = ChipKernel_->getReflection();
  if (Refl != nullptr) {
    PushConstantBlob_.assign(Refl->PushConstantBlockSize, 0);
    BufferBindings_.assign(Refl->MaxDescriptorBinding + 1, VK_NULL_HANDLE);
    BufferRanges_.assign(Refl->MaxDescriptorBinding + 1, VK_WHOLE_SIZE);
  }
}

chipstar::Kernel *CHIPExecItemVulkan::getKernel() { return ChipKernel_; }

void CHIPExecItemVulkan::setupAllArgs() {
  assert(ChipKernel_ && "setupAllArgs called before setKernel");
  const auto *Refl = ChipKernel_->getReflection();
  if (Refl == nullptr)
    CHIPERR_LOG_AND_THROW("ExecItem::setupAllArgs: kernel has no reflection",
                          hipErrorLaunchFailure);
  auto *Mod = static_cast<CHIPModuleVulkan *>(ChipKernel_->getModule());
  auto *Dev = Mod->getDevice();
  auto *Ctx = Dev->getContext();

  std::vector<std::pair<int32_t, uint64_t>> PointerOffsets;
  for (const auto &Buf : Refl->Buffers) {
    int32_t ArgsIdx =
        Buf.HipSourceIndex >= 0 ? Buf.HipSourceIndex : (int32_t)Buf.Ordinal;
    void *HipPtr;
    if (Buf.FieldArg >= 0) {
      std::memcpy(&HipPtr,
                  static_cast<char *>(Args_[Buf.FieldArg]) + Buf.FieldOffset,
                  sizeof(HipPtr));
    } else if (Buf.DevGlobalName.empty()) {
      HipPtr = *reinterpret_cast<void **>(Args_[ArgsIdx]);
    } else {
      SPVFuncInfo::KernelArg DG{};
      DG.DevGlobalName = Buf.DevGlobalName;
      HipPtr = chipstar::getDeviceGlobalArgAddr(ChipKernel_, DG);
    }
    size_t Offset = 0;
    // A null pointer argument is legal as long as the kernel never
    // dereferences it; bind a small placeholder buffer.
    bool IsNull = HipPtr == nullptr;
    if (IsNull)
      HipPtr = Ctx->getNullArgPlaceholder();
    const auto *Entry = Ctx->getDevPtrEntryContaining(HipPtr, Offset);
    // Pointers chipStar did not allocate (plain host memory) are not
    // accessible from the device; bind the placeholder, so a kernel that only
    // passes the pointer along (e.g. to a stateless functor) still runs.
    if (Entry == nullptr) {
      logWarn("kernel pointer argument {} is not a device allocation; the "
              "kernel must not dereference it",
              HipPtr);
      Offset = 0;
      Entry = Ctx->getDevPtrEntryContaining(Ctx->getNullArgPlaceholder(),
                                            Offset);
    }
    if (Entry == nullptr) {
      std::string Msg = "ExecItem::setupAllArgs: unregistered device pointer "
                        "for kernel arg at ordinal " +
                        std::to_string(Buf.Ordinal);
      CHIPERR_LOG_AND_THROW(Msg, hipErrorInvalidDevicePointer);
    }
    // Bind the whole allocation; the kernel adds the pointer's offset.
    BufferBindings_[Buf.Binding] = Entry->Buffer;
    BufferRanges_[Buf.Binding] = Entry->Size;
    // The kernel compares pointers with null by this offset.
    PointerOffsets.emplace_back(Buf.PCOffset,
                                IsNull ? 1ull << 63 : static_cast<uint64_t>(Offset));
  }
  for (const auto &Pc : Refl->PushConst) {
    int32_t ArgsIdx =
        Pc.HipSourceIndex >= 0 ? Pc.HipSourceIndex : (int32_t)Pc.Ordinal;
    std::memcpy(PushConstantBlob_.data() + Pc.Offset, Args_[ArgsIdx], Pc.Size);
  }
  for (auto [PCOffset, Offset] : PointerOffsets)
    if (PCOffset >= 0 && PCOffset + sizeof(Offset) <= PushConstantBlob_.size())
      std::memcpy(PushConstantBlob_.data() + PCOffset, &Offset, sizeof(Offset));
  this->ArgsSetup = true;
}

// ============================================================================
// CHIPContextVulkan
// ============================================================================

CHIPContextVulkan::CHIPContextVulkan() = default;

CHIPContextVulkan::~CHIPContextVulkan() = default;

void *CHIPContextVulkan::allocateImpl(size_t Size, size_t Alignment,
                                      hipMemoryType MemType,
                                      chipstar::HostAllocFlags Flags) {
  LOCK(ContextMtx); // CHIPContextVulkan::DevPtrToEntry_

  CHIPDeviceVulkan *Dev = getVulkanDevice();
  if (!Dev)
    CHIPERR_LOG_AND_THROW("CHIPContextVulkan::allocateImpl: no device bound "
                          "to context",
                          hipErrorInvalidContext);
  VmaAllocator Allocator = Dev->getAllocator();
  if (Allocator == VK_NULL_HANDLE)
    CHIPERR_LOG_AND_THROW("CHIPContextVulkan::allocateImpl: VMA allocator is "
                          "not initialized",
                          hipErrorInvalidContext);

  VkBufferCreateInfo BufInfo{};
  BufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  BufInfo.size = Size;
  BufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  BufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo AllocInfo{};
  // HIP guarantees 256-byte aligned device allocations.
  AllocInfo.minAlignment =
      std::max<VkDeviceSize>(static_cast<VkDeviceSize>(Alignment), 256);
  AllocInfo.pUserData = nullptr;

  // Host memory is HOST_VISIBLE | HOST_COHERENT whatever the flags.
  (void)Flags;

  // A device allocation's HIP pointer is its buffer device address, so
  // [pointer, pointer + size) ranges never overlap; a host-visible one's is
  // its mapped host address.
  void *RetPtr = nullptr;
  VkBuffer Buffer = VK_NULL_HANDLE;
  VmaAllocation Allocation = VK_NULL_HANDLE;
  VmaAllocationInfo AllocResult{};
  VkResult Status = VK_ERROR_UNKNOWN;

  switch (MemType) {
  case hipMemoryTypeDevice: {
    AllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    AllocInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    Status = vmaCreateBuffer(Allocator, &BufInfo, &AllocInfo, &Buffer,
                             &Allocation, &AllocResult);
    if (Status != VK_SUCCESS || Allocation == VK_NULL_HANDLE)
      CHIPERR_LOG_AND_THROW("vmaCreateBuffer (device) failed",
                            hipErrorOutOfMemory);
    VkBufferDeviceAddressInfo Bdai{};
    Bdai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    Bdai.buffer = Buffer;
    RetPtr = reinterpret_cast<void *>(
        vkGetBufferDeviceAddress(Dev->getLogicalDevice(), &Bdai));
    break;
  }
  case hipMemoryTypeHost: {
    // System memory, not the device-local BAR that CPU_TO_GPU picks on a
    // ReBAR GPU: host reads from BAR are uncached and very slow.
    AllocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    AllocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    AllocInfo.preferredFlags = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    AllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                      VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    Status = vmaCreateBuffer(Allocator, &BufInfo, &AllocInfo, &Buffer,
                             &Allocation, &AllocResult);
    if (Status != VK_SUCCESS || Allocation == VK_NULL_HANDLE ||
        AllocResult.pMappedData == nullptr)
      CHIPERR_LOG_AND_THROW("vmaCreateBuffer (host) failed",
                            hipErrorOutOfMemory);
    RetPtr = AllocResult.pMappedData;
    break;
  }
  case hipMemoryTypeManaged:
  case hipMemoryTypeUnified: {
    // Host visible, preferably also device local.
    AllocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    AllocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    AllocInfo.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    AllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                      VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    Status = vmaCreateBuffer(Allocator, &BufInfo, &AllocInfo, &Buffer,
                             &Allocation, &AllocResult);
    if (Status != VK_SUCCESS || Allocation == VK_NULL_HANDLE ||
        AllocResult.pMappedData == nullptr) {
      AllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
      AllocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      AllocInfo.preferredFlags = 0;
      AllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
      Status = vmaCreateBuffer(Allocator, &BufInfo, &AllocInfo, &Buffer,
                               &Allocation, &AllocResult);
      if (Status != VK_SUCCESS || Allocation == VK_NULL_HANDLE ||
          AllocResult.pMappedData == nullptr)
        CHIPERR_LOG_AND_THROW("vmaCreateBuffer (managed) failed",
                              hipErrorOutOfMemory);
    }
    RetPtr = AllocResult.pMappedData;
    break;
  }
  default:
    CHIPERR_LOG_AND_THROW(
        "CHIPContextVulkan::allocateImpl: unsupported hipMemoryType",
        hipErrorInvalidValue);
  }

  DevPtrEntry Entry;
  Entry.Buffer = Buffer;
  Entry.Allocation = Allocation;
  Entry.AllocInfo = AllocResult;
  Entry.Size = Size;
  Entry.MemType = MemType;
  Entry.Flags = Flags;
  DevPtrToEntry_.emplace(static_cast<const void *>(RetPtr), Entry);
  return RetPtr;
}

bool CHIPContextVulkan::isAllocatedPtrMappedToVM(void *Ptr) {
  LOCK(ContextMtx); // CHIPContextVulkan::DevPtrToEntry_
  return DevPtrToEntry_.find(Ptr) != DevPtrToEntry_.end();
}

void CHIPContextVulkan::freeAll(VmaAllocator Allocator) {
  LOCK(ContextMtx); // CHIPContextVulkan::DevPtrToEntry_
  for (auto &[Ptr, Entry] : DevPtrToEntry_)
    vmaDestroyBuffer(Allocator, Entry.Buffer, Entry.Allocation);
  DevPtrToEntry_.clear();
}

void CHIPContextVulkan::freeImpl(void *Ptr) {
  if (!Ptr)
    return;

  LOCK(ContextMtx); // CHIPContextVulkan::DevPtrToEntry_

  auto It = DevPtrToEntry_.find(Ptr);
  if (It == DevPtrToEntry_.end()) {
    logError("CHIPContextVulkan::freeImpl: pointer {} not tracked by this "
             "context", Ptr);
    return;
  }

  CHIPDeviceVulkan *Dev = getVulkanDevice();
  VmaAllocator Allocator = Dev ? Dev->getAllocator() : VK_NULL_HANDLE;
  if (Allocator != VK_NULL_HANDLE)
    vmaDestroyBuffer(Allocator, It->second.Buffer, It->second.Allocation);
  DevPtrToEntry_.erase(It);
}

const CHIPContextVulkan::DevPtrEntry *
CHIPContextVulkan::getDevPtrEntry(const void *DevPtr) const {
  auto It = DevPtrToEntry_.find(DevPtr);
  return It == DevPtrToEntry_.end() ? nullptr : &It->second;
}

const CHIPContextVulkan::DevPtrEntry *
CHIPContextVulkan::getDevPtrEntryContaining(const void *DevPtr,
                                            size_t &OutOffset) const {
  OutOffset = 0;
  if (!DevPtr)
    return nullptr;
  auto It = DevPtrToEntry_.find(DevPtr);
  if (It != DevPtrToEntry_.end()) {
    OutOffset = 0;
    return &It->second;
  }
  // Otherwise the entry with the greatest base below DevPtr, if it covers it.
  auto Ub = DevPtrToEntry_.upper_bound(DevPtr);
  if (Ub != DevPtrToEntry_.begin()) {
    --Ub;
    const auto *Bytes = static_cast<const uint8_t *>(DevPtr);
    const auto *BaseBytes = static_cast<const uint8_t *>(Ub->first);
    if (Bytes < BaseBytes + Ub->second.Size) {
      OutOffset = static_cast<size_t>(Bytes - BaseBytes);
      return &Ub->second;
    }
  }
  return nullptr;
}

void *CHIPContextVulkan::getPodArgBuffer() {
  std::call_once(PodArgOnce_, [this]() {
    PodArgBuffer_ = allocateImpl(PodArgBufferSize, 256, hipMemoryTypeDevice);
  });
  return PodArgBuffer_;
}

void *CHIPContextVulkan::getNullArgPlaceholder() {
  std::call_once(NullArgOnce_, [this]() {
    NullArgPlaceholder_ = allocateImpl(256, 256, hipMemoryTypeDevice);
  });
  return NullArgPlaceholder_;
}

CHIPDeviceVulkan *CHIPContextVulkan::getVulkanDevice() const {
  return static_cast<CHIPDeviceVulkan *>(ChipDevice_);
}

// ============================================================================
// CHIPDeviceVulkan
// ============================================================================

CHIPDeviceVulkan::CHIPDeviceVulkan(CHIPContextVulkan *ChipContext,
                                   VkPhysicalDevice PhysDev, int Idx)
    : chipstar::Device(ChipContext, Idx), PhysicalDevice_(PhysDev) {}

namespace {
inline void i8DrainCompletedStagings(VkDevice Dev);
} // namespace

CHIPDeviceVulkan::~CHIPDeviceVulkan() {
  // create() can fail partway, and the query pool is optional.
  if (LogicalDevice_ != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(LogicalDevice_);
    // Buffers still alive at exit, which VMA requires freed first.
    if (Allocator_ != VK_NULL_HANDLE) {
      i8DrainCompletedStagings(LogicalDevice_);
      static_cast<CHIPContextVulkan *>(getContext())->freeAll(Allocator_);
    }

    {
      std::lock_guard<std::mutex> Lock(FencePoolMtx_);
      for (VkFence F : FencePool_)
        vkDestroyFence(LogicalDevice_, F, nullptr);
      FencePool_.clear();
    }
    if (TimestampQueryPool_ != VK_NULL_HANDLE) {
      vkDestroyQueryPool(LogicalDevice_, TimestampQueryPool_, nullptr);
      TimestampQueryPool_ = VK_NULL_HANDLE;
    }
    if (PipelineCache_ != VK_NULL_HANDLE) {
      vkDestroyPipelineCache(LogicalDevice_, PipelineCache_, nullptr);
      PipelineCache_ = VK_NULL_HANDLE;
    }
    if (Allocator_ != VK_NULL_HANDLE) {
      vmaDestroyAllocator(Allocator_);
      Allocator_ = VK_NULL_HANDLE;
    }
    vkDestroyDevice(LogicalDevice_, nullptr);
    LogicalDevice_ = VK_NULL_HANDLE;
  }
}
CHIPDeviceVulkan *CHIPDeviceVulkan::create(CHIPContextVulkan *ChipContext,
                                           VkPhysicalDevice PhysDev, int Idx) {
  CHIPDeviceVulkan *Dev = new CHIPDeviceVulkan(ChipContext, PhysDev, Idx);
  // Device::init() may allocate through the context.
  ChipContext->setDevice(Dev);
  // The destructor frees whatever handles were created.
  auto Fail = [&](const std::string &Msg) {
    ChipContext->setDevice(nullptr);
    delete Dev;
    CHIPERR_LOG_AND_THROW(Msg, hipErrorInitializationError);
  };

  Dev->Properties_ = {};
  vkGetPhysicalDeviceProperties(PhysDev, &Dev->Properties_);
  Dev->Features_ = {};
  vkGetPhysicalDeviceFeatures(PhysDev, &Dev->Features_);
  Dev->SubgroupProperties_ = {};
  Dev->SubgroupProperties_.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
  Dev->FloatControls_ = {};
  Dev->FloatControls_.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES;
  Dev->SubgroupProperties_.pNext = &Dev->FloatControls_;
  VkPhysicalDeviceIDProperties IDProps{};
  IDProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
  Dev->FloatControls_.pNext = &IDProps;
  VkPhysicalDeviceProperties2 Props2{};
  Props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  Props2.pNext = &Dev->SubgroupProperties_;
  vkGetPhysicalDeviceProperties2(PhysDev, &Props2);
  Dev->SubgroupProperties_.pNext = nullptr;
  Dev->FloatControls_.pNext = nullptr;
  std::memcpy(Dev->IpcUUID_, IDProps.deviceUUID, VK_UUID_SIZE);
  std::memcpy(Dev->IpcUUID_ + VK_UUID_SIZE, IDProps.driverUUID, VK_UUID_SIZE);

  VkPhysicalDeviceShaderFloat16Int8Features F16I8Features{};
  F16I8Features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
  // Features kernels may need: 16-bit storage (1.1), float atomics.
  VkPhysicalDeviceShaderClockFeaturesKHR Clock{};
  Clock.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR;
  VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT AtomicFloat2{};
  AtomicFloat2.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT;
  AtomicFloat2.pNext = &Clock;
  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT AtomicFloat{};
  AtomicFloat.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
  AtomicFloat.pNext = &AtomicFloat2;
  VkPhysicalDeviceVulkan11Features V11Features{};
  V11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
  V11Features.pNext = &AtomicFloat;
  F16I8Features.pNext = &V11Features;
  VkPhysicalDeviceVulkan12Features V12Features{};
  V12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  V12Features.pNext = &F16I8Features;
  VkPhysicalDeviceFeatures2 Feat2{};
  Feat2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  Feat2.pNext = &V12Features;
  vkGetPhysicalDeviceFeatures2(PhysDev, &Feat2);

  Dev->HasShaderInt8_ = (V12Features.shaderInt8 == VK_TRUE) ||
                        (F16I8Features.shaderInt8 == VK_TRUE);
  Dev->HasShaderInt64_ = Feat2.features.shaderInt64 == VK_TRUE;

  uint32_t QFamCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(PhysDev, &QFamCount, nullptr);
  std::vector<VkQueueFamilyProperties> QFamProps(QFamCount);
  vkGetPhysicalDeviceQueueFamilyProperties(PhysDev, &QFamCount,
                                           QFamProps.data());
  uint32_t QFamIdx = ~0u;
  // Prefer the universal (graphics+compute) family: on Mesa ANV the
  // compute-only family on Xe2 (Arc B570) leaves dispatch writes invisible
  // to the following transfer. Fall back to any COMPUTE family.
  for (uint32_t i = 0; i < QFamCount; ++i) {
    if ((QFamProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
        (QFamProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
      QFamIdx = i;
      break;
    }
  }
  if (QFamIdx == ~0u) {
    for (uint32_t i = 0; i < QFamCount; ++i) {
      if (QFamProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        QFamIdx = i;
        break;
      }
    }
  }
  if (QFamIdx == ~0u) {
    Fail(std::string("No compute-capable queue family on Vulkan device: ") +
         Dev->Properties_.deviceName);
  }
  Dev->ComputeQueueFamilyIndex_ = QFamIdx;

  // Kernels require VK_KHR_shader_non_semantic_info.
  uint32_t ExtCount = 0;
  vkEnumerateDeviceExtensionProperties(PhysDev, nullptr, &ExtCount, nullptr);
  std::vector<VkExtensionProperties> AvailExts(ExtCount);
  vkEnumerateDeviceExtensionProperties(PhysDev, nullptr, &ExtCount,
                                       AvailExts.data());
  if (!vkPropertyListContains(AvailExts, &VkExtensionProperties::extensionName,
                              VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME)) {
    Fail(std::string("Vulkan device lacks VK_KHR_shader_non_semantic_info: ") +
         Dev->Properties_.deviceName);
  }

  const float QueuePriority = 1.0f;
  VkDeviceQueueCreateInfo QInfo{};
  QInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  QInfo.queueFamilyIndex = QFamIdx;
  QInfo.queueCount = 1;
  QInfo.pQueuePriorities = &QueuePriority;

  // Core 1.2 features still have to be enabled explicitly.
  VkPhysicalDeviceVulkan12Features Enable12{};
  Enable12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  Enable12.shaderInt8 = Dev->HasShaderInt8_ ? VK_TRUE : VK_FALSE;
  Enable12.hostQueryReset = VK_TRUE;
  Enable12.timelineSemaphore = VK_TRUE;
  Enable12.bufferDeviceAddress = VK_TRUE;
  Enable12.bufferDeviceAddressCaptureReplay = VK_FALSE;
  Enable12.storageBuffer8BitAccess = V12Features.storageBuffer8BitAccess;
  Enable12.uniformAndStorageBuffer8BitAccess =
      V12Features.uniformAndStorageBuffer8BitAccess;
  Enable12.storagePushConstant8 = V12Features.storagePushConstant8;
  Enable12.shaderFloat16 = V12Features.shaderFloat16;
  Enable12.shaderBufferInt64Atomics = V12Features.shaderBufferInt64Atomics;
  Enable12.shaderSharedInt64Atomics = V12Features.shaderSharedInt64Atomics;

  // Kernels take their workgroup size through LocalSizeId, which needs
  // maintenance4 (VUID-RuntimeSpirv-LocalSizeId-06434).
  VkPhysicalDeviceMaintenance4Features EnableM4{};
  EnableM4.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES;
  EnableM4.maintenance4 = VK_TRUE;
  // Every supported 1.1 feature, and the float atomics the device has.
  V11Features.pNext = nullptr;
  EnableM4.pNext = &V11Features;

  Enable12.pNext = &EnableM4;

  VkPhysicalDeviceFeatures2 EnableFeat{};
  EnableFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  EnableFeat.pNext = &Enable12;
  EnableFeat.features.shaderInt64 = Dev->HasShaderInt64_ ? VK_TRUE : VK_FALSE;
  EnableFeat.features.shaderInt16 = Feat2.features.shaderInt16;
  EnableFeat.features.shaderFloat64 = Feat2.features.shaderFloat64;

  std::vector<const char *> DevExts = {
      VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
  };
  auto HasExt = [&](const char *Name) {
    return vkPropertyListContains(AvailExts,
                                  &VkExtensionProperties::extensionName, Name);
  };
  void **Tail = &V11Features.pNext;
  if (HasExt(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME)) {
    DevExts.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    AtomicFloat.pNext = nullptr;
    *Tail = &AtomicFloat;
    Tail = &AtomicFloat.pNext;
  }
  if (HasExt(VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME)) {
    DevExts.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME);
    AtomicFloat2.pNext = nullptr;
    *Tail = &AtomicFloat2;
    Tail = &AtomicFloat2.pNext;
  }
  // IPC events: timeline semaphores shared as opaque fds.
  {
    VkSemaphoreTypeCreateInfo TypeInfo{};
    TypeInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    TypeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkPhysicalDeviceExternalSemaphoreInfo ExtInfo{};
    ExtInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO;
    ExtInfo.pNext = &TypeInfo;
    ExtInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkExternalSemaphoreProperties ExtProps{};
    ExtProps.sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES;
    vkGetPhysicalDeviceExternalSemaphoreProperties(PhysDev, &ExtInfo,
                                                   &ExtProps);
    const VkExternalSemaphoreFeatureFlags Need =
        VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT |
        VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT;
    Dev->HasIpcSemaphore_ =
        HasExt(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME) &&
        (ExtProps.externalSemaphoreFeatures & Need) == Need;
    if (Dev->HasIpcSemaphore_)
      DevExts.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
  }
  // For __builtin_readcyclecounter, which reads the subgroup clock.
  if (HasExt(VK_KHR_SHADER_CLOCK_EXTENSION_NAME) && Clock.shaderSubgroupClock) {
    DevExts.push_back(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
    Clock.pNext = nullptr;
    Clock.shaderDeviceClock = VK_FALSE;
    *Tail = &Clock;
  }

  VkDeviceCreateInfo DevInfo{};
  DevInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  DevInfo.pNext = &EnableFeat;
  DevInfo.queueCreateInfoCount = 1;
  DevInfo.pQueueCreateInfos = &QInfo;
  DevInfo.enabledExtensionCount = static_cast<uint32_t>(DevExts.size());
  DevInfo.ppEnabledExtensionNames = DevExts.data();

  VkResult R =
      vkCreateDevice(PhysDev, &DevInfo, nullptr, &Dev->LogicalDevice_);
  if (R != VK_SUCCESS) {
    Fail(std::string("vkCreateDevice failed: ") +
         std::to_string(static_cast<int>(R)));
  }

  vkGetDeviceQueue(Dev->LogicalDevice_, QFamIdx, 0, &Dev->ComputeQueue_);

  if (Dev->HasIpcSemaphore_) {
    Dev->GetSemaphoreFd_ = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
        vkGetDeviceProcAddr(Dev->LogicalDevice_, "vkGetSemaphoreFdKHR"));
    Dev->ImportSemaphoreFd_ = reinterpret_cast<PFN_vkImportSemaphoreFdKHR>(
        vkGetDeviceProcAddr(Dev->LogicalDevice_, "vkImportSemaphoreFdKHR"));
    Dev->HasIpcSemaphore_ = Dev->GetSemaphoreFd_ && Dev->ImportSemaphoreFd_;
  }

  VmaAllocatorCreateInfo AInfo{};
  AInfo.physicalDevice = PhysDev;
  AInfo.device = Dev->LogicalDevice_;
  AInfo.instance = static_cast<CHIPBackendVulkan *>(::Backend)->getInstance();
  AInfo.vulkanApiVersion = VK_API_VERSION_1_3;
  AInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  R = vmaCreateAllocator(&AInfo, &Dev->Allocator_);
  if (R != VK_SUCCESS) {
    Fail(std::string("vmaCreateAllocator failed: ") +
         std::to_string(static_cast<int>(R)));
  }

  VkPipelineCacheCreateInfo PCInfo{};
  PCInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
  R = vkCreatePipelineCache(Dev->LogicalDevice_, &PCInfo, nullptr,
                            &Dev->PipelineCache_);
  if (R != VK_SUCCESS) {
    Fail(std::string("vkCreatePipelineCache failed: ") +
         std::to_string(static_cast<int>(R)));
  }

  Dev->TimestampValidBits_ = QFamProps[QFamIdx].timestampValidBits;
  if (Dev->Properties_.limits.timestampPeriod > 0.0f &&
      (Dev->Properties_.limits.timestampComputeAndGraphics ||
       QFamProps[QFamIdx].timestampValidBits > 0)) {
    VkQueryPoolCreateInfo QPInfo{};
    QPInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    QPInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    QPInfo.queryCount = CHIPDeviceVulkan::TimestampPoolSize_;
    R = vkCreateQueryPool(Dev->LogicalDevice_, &QPInfo, nullptr,
                          &Dev->TimestampQueryPool_);
    if (R != VK_SUCCESS) {
      Fail(std::string("vkCreateQueryPool failed: ") +
           std::to_string(static_cast<int>(R)));
    }
    Dev->TimestampFreeList_.reserve(CHIPDeviceVulkan::TimestampPoolSize_);
    for (int32_t i =
             static_cast<int32_t>(CHIPDeviceVulkan::TimestampPoolSize_) - 1;
         i >= 0; --i) {
      Dev->TimestampFreeList_.push_back(i);
    }
    // Needs the hostQueryReset feature.
    vkResetQueryPool(Dev->LogicalDevice_, Dev->TimestampQueryPool_, 0,
                     CHIPDeviceVulkan::TimestampPoolSize_);
  } else {
    logWarn("Vulkan device '{}' does not support timestamp queries; HIP event "
            "timing will fall back to host clock.",
            Dev->Properties_.deviceName);
  }

  Dev->init();
  return Dev;
}

chipstar::Context *CHIPDeviceVulkan::createContext() {
  // Backend::initializeImpl() creates the context.
  return nullptr;
}

void CHIPDeviceVulkan::populateDevicePropertiesImpl() {
  logTrace("CHIPDeviceVulkan->populateDevicePropertiesImpl()");

  // Total of the DEVICE_LOCAL heaps.
  VkPhysicalDeviceMemoryProperties MemProps{};
  vkGetPhysicalDeviceMemoryProperties(PhysicalDevice_, &MemProps);
  VkDeviceSize TotalDeviceLocal = 0;
  for (uint32_t i = 0; i < MemProps.memoryHeapCount; ++i) {
    if (MemProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
      TotalDeviceLocal += MemProps.memoryHeaps[i].size;
  }

  uint32_t QFamCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(PhysicalDevice_, &QFamCount, nullptr);

  const VkPhysicalDeviceLimits &Limits = Properties_.limits;

  std::strncpy(HipDeviceProps_.name, Properties_.deviceName,
               sizeof(HipDeviceProps_.name) - 1);
  HipDeviceProps_.name[sizeof(HipDeviceProps_.name) - 1] = 0;

  MaxMallocSize_ = Limits.maxStorageBufferRange;
  HipDeviceProps_.totalGlobalMem = TotalDeviceLocal;
  // Dynamic shared memory arrays are at most ChipVulkanDynSharedBytes.
  HipDeviceProps_.sharedMemPerBlock = std::min<size_t>(
      Limits.maxComputeSharedMemorySize, ChipVulkanDynSharedBytes);
  HipDeviceProps_.maxThreadsPerBlock = Limits.maxComputeWorkGroupInvocations;
  HipDeviceProps_.maxThreadsDim[0] = Limits.maxComputeWorkGroupSize[0];
  HipDeviceProps_.maxThreadsDim[1] = Limits.maxComputeWorkGroupSize[1];
  HipDeviceProps_.maxThreadsDim[2] = Limits.maxComputeWorkGroupSize[2];
  HipDeviceProps_.maxGridSize[0] = Limits.maxComputeWorkGroupCount[0];
  HipDeviceProps_.maxGridSize[1] = Limits.maxComputeWorkGroupCount[1];
  HipDeviceProps_.maxGridSize[2] = Limits.maxComputeWorkGroupCount[2];

  // Vulkan does not report clock rates; use placeholders.
  HipDeviceProps_.clockRate = 1000 * 1000; // 1 GHz, kHz units.
  HipDeviceProps_.memoryClockRate = 1000;
  HipDeviceProps_.memoryBusWidth = 256;
  HipDeviceProps_.clockInstructionRate = 1000 * 1000;

  HipDeviceProps_.totalConstMem = Limits.maxStorageBufferRange;
  HipDeviceProps_.l2CacheSize = 0;
  HipDeviceProps_.regsPerBlock = 0; // No Vulkan analogue.

  HipDeviceProps_.warpSize =
      SubgroupProperties_.subgroupSize > 0
          ? static_cast<int>(SubgroupProperties_.subgroupSize)
          : CHIP_DEFAULT_WARP_SIZE;

  // Vulkan does not report a compute unit count; use the queue family count.
  HipDeviceProps_.multiProcessorCount = static_cast<int>(QFamCount);

  HipDeviceProps_.major = 2;
  HipDeviceProps_.minor = 0;
  HipDeviceProps_.computeMode = hipComputeModeDefault;
  HipDeviceProps_.arch = {};
  HipDeviceProps_.arch.hasGlobalInt32Atomics = 1;
  HipDeviceProps_.arch.hasSharedInt32Atomics = 1;
  HipDeviceProps_.arch.hasGlobalInt64Atomics = HasShaderInt64_ ? 1 : 0;
  HipDeviceProps_.arch.hasSharedInt64Atomics = HasShaderInt64_ ? 1 : 0;
  HipDeviceProps_.arch.hasDoubles = Features_.shaderFloat64 ? 1 : 0;
  HipDeviceProps_.arch.hasWarpBallot =
      (SubgroupProperties_.supportedOperations &
       VK_SUBGROUP_FEATURE_BALLOT_BIT)
          ? 1
          : 0;

  HipDeviceProps_.maxThreadsPerMultiProcessor =
      Limits.maxComputeWorkGroupInvocations;

  HipDeviceProps_.concurrentKernels = 1;
  HipDeviceProps_.pciDomainID = 0;
  HipDeviceProps_.pciBusID = 0x10;
  HipDeviceProps_.pciDeviceID = 0x40 + getDeviceId();
  HipDeviceProps_.isMultiGpuBoard = 0;
  HipDeviceProps_.canMapHostMemory = 1;
  HipDeviceProps_.integrated =
      Properties_.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 1 : 0;
  HipDeviceProps_.maxSharedMemoryPerMultiProcessor =
      Limits.maxComputeSharedMemorySize;

  HipDeviceProps_.managedMemory = HipDeviceProps_.integrated ? 1 : 0;
  HipDeviceProps_.directManagedMemAccessFromHost = 0;
  HipDeviceProps_.concurrentManagedAccess = 0;
  HipDeviceProps_.pageableMemoryAccess = 0;
  HipDeviceProps_.pageableMemoryAccessUsesHostPageTables = 0;

  HipDeviceProps_.cooperativeLaunch = 0;
  HipDeviceProps_.cooperativeMultiDeviceLaunch = 0;
  HipDeviceProps_.cooperativeMultiDeviceUnmatchedFunc = 0;
  HipDeviceProps_.cooperativeMultiDeviceUnmatchedGridDim = 0;
  HipDeviceProps_.cooperativeMultiDeviceUnmatchedBlockDim = 0;
  HipDeviceProps_.cooperativeMultiDeviceUnmatchedSharedMem = 0;

  HipDeviceProps_.memPitch = 1;
  HipDeviceProps_.textureAlignment = 1;
  HipDeviceProps_.texturePitchAlignment = 1;
  HipDeviceProps_.kernelExecTimeoutEnabled = 0;
  HipDeviceProps_.ECCEnabled = 0;
  HipDeviceProps_.asicRevision = 1;
  HipDeviceProps_.unifiedAddressing = 1;


  constexpr char ArchName[] = "vulkan";
  static_assert(sizeof(ArchName) <= sizeof(HipDeviceProps_.gcnArchName),
                "gcnArchName overflow");
  std::strncpy(HipDeviceProps_.gcnArchName, ArchName, sizeof(ArchName));
}

chipstar::Queue *CHIPDeviceVulkan::createQueue(chipstar::QueueFlags Flags,
                                               int Priority) {
  return new CHIPQueueVulkan(this, Flags, Priority);
}

chipstar::Queue *CHIPDeviceVulkan::createQueue(const uintptr_t * /*NH*/,
                                               int /*NumHandles*/) {
  CHIPERR_LOG_AND_THROW(
      "CHIPDeviceVulkan::createQueue(native): not implemented",
      hipErrorNotSupported);
}

chipstar::Texture *
CHIPDeviceVulkan::createTexture(const hipResourceDesc * /*ResDesc*/,
                                const hipTextureDesc * /*TexDesc*/,
                                const struct hipResourceViewDesc * /*RVD*/) {
  CHIPERR_LOG_AND_THROW(
      "Texture support not implemented in Vulkan backend yet",
      hipErrorNotSupported);
  return nullptr; // Unreachable; CHIPERR_LOG_AND_THROW always throws.
}

void CHIPDeviceVulkan::destroyTexture(chipstar::Texture *TextureObject) {
  delete TextureObject;
}

void CHIPDeviceVulkan::resetImpl() {
  // Waits for the device and empties the pipeline cache; the VkDevice stays.
  if (LogicalDevice_ != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(LogicalDevice_);
    if (PipelineCache_ != VK_NULL_HANDLE) {
      vkDestroyPipelineCache(LogicalDevice_, PipelineCache_, nullptr);
      VkPipelineCacheCreateInfo PCInfo{};
      PCInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
      VkResult R = vkCreatePipelineCache(LogicalDevice_, &PCInfo, nullptr,
                                         &PipelineCache_);
      if (R != VK_SUCCESS) {
        PipelineCache_ = VK_NULL_HANDLE;
        logError("vkCreatePipelineCache failed in resetImpl: {}",
                 static_cast<int>(R));
      }
    }
  }
}

chipstar::Module *CHIPDeviceVulkan::compile(const SPVModule &Src) {
  auto *Mod = new CHIPModuleVulkan(Src);
  Mod->compile(this);
  return Mod;
}

VkFence CHIPDeviceVulkan::acquireFence() {
  std::lock_guard<std::mutex> Lock(FencePoolMtx_);
  if (!FencePool_.empty()) {
    VkFence F = FencePool_.back();
    FencePool_.pop_back();
    vkResetFences(LogicalDevice_, 1, &F);
    return F;
  }
  VkFenceCreateInfo FInfo{};
  FInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence F = VK_NULL_HANDLE;
  VkResult R = vkCreateFence(LogicalDevice_, &FInfo, nullptr, &F);
  if (R != VK_SUCCESS) {
    CHIPERR_LOG_AND_THROW(std::string("vkCreateFence failed: ") +
                              std::to_string(static_cast<int>(R)),
                          hipErrorOutOfMemory);
  }
  return F;
}

void CHIPDeviceVulkan::releaseFence(VkFence F) {
  if (F == VK_NULL_HANDLE)
    return;
  std::lock_guard<std::mutex> Lock(FencePoolMtx_);
  vkResetFences(LogicalDevice_, 1, &F);
  FencePool_.push_back(F);
}

int32_t CHIPDeviceVulkan::acquireTimestampSlot() {
  if (TimestampQueryPool_ == VK_NULL_HANDLE)
    return -1;
  std::lock_guard<std::mutex> Lock(TimestampMtx_);
  if (TimestampFreeList_.empty())
    return -1;
  int32_t Slot = TimestampFreeList_.back();
  TimestampFreeList_.pop_back();
  vkResetQueryPool(LogicalDevice_, TimestampQueryPool_, static_cast<uint32_t>(Slot), 1);
  return Slot;
}

void CHIPDeviceVulkan::releaseTimestampSlot(int32_t Slot) {
  if (Slot < 0 || TimestampQueryPool_ == VK_NULL_HANDLE)
    return;
  std::lock_guard<std::mutex> Lock(TimestampMtx_);
  TimestampFreeList_.push_back(Slot);
}

// ============================================================================
// CHIPQueueVulkan
// ============================================================================

CHIPQueueVulkan::CHIPQueueVulkan(chipstar::Device *ChipDevice,
                                 chipstar::QueueFlags Flags, int Priority)
    : chipstar::Queue(ChipDevice, Flags, Priority),
      ChipDevice_(static_cast<CHIPDeviceVulkan *>(ChipDevice)),
      CmdBufferRing_(RingCapacity_, VK_NULL_HANDLE) {
  VkDevice Dev = ChipDevice_->getLogicalDevice();
  if (Dev == VK_NULL_HANDLE) {
    CHIPERR_LOG_AND_THROW(
        "CHIPQueueVulkan ctor called before CHIPDeviceVulkan logical device "
        "was created",
        hipErrorTbd);
  }

  VkCommandPoolCreateInfo PoolCI{};
  PoolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  PoolCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                 VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  PoolCI.queueFamilyIndex = ChipDevice_->getComputeQueueFamilyIndex();
  checkVk(vkCreateCommandPool(Dev, &PoolCI, nullptr, &CommandPool_),
          "CHIPQueueVulkan: vkCreateCommandPool failed", hipErrorTbd);

  VkCommandBufferAllocateInfo CBAI{};
  CBAI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  CBAI.commandPool = CommandPool_;
  CBAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  CBAI.commandBufferCount = RingCapacity_;
  VkResult AllocR =
      vkAllocateCommandBuffers(Dev, &CBAI, CmdBufferRing_.data());
  if (AllocR != VK_SUCCESS) {
    vkDestroyCommandPool(Dev, CommandPool_, nullptr);
    CommandPool_ = VK_NULL_HANDLE;
    checkVk(AllocR, "CHIPQueueVulkan: vkAllocateCommandBuffers failed",
            hipErrorTbd);
  }

  VkSemaphoreTypeCreateInfo SemTypeCI{};
  SemTypeCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  SemTypeCI.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  SemTypeCI.initialValue = 0;
  VkSemaphoreCreateInfo SemCI{};
  SemCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  SemCI.pNext = &SemTypeCI;
  VkResult SemR = vkCreateSemaphore(Dev, &SemCI, nullptr, &TimelineSemaphore_);
  if (SemR != VK_SUCCESS) {
    vkFreeCommandBuffers(Dev, CommandPool_, RingCapacity_,
                         CmdBufferRing_.data());
    vkDestroyCommandPool(Dev, CommandPool_, nullptr);
    CommandPool_ = VK_NULL_HANDLE;
    checkVk(SemR, "CHIPQueueVulkan: vkCreateSemaphore (timeline) failed",
            hipErrorTbd);
  }

  VkFenceCreateInfo FenceCI{};
  FenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkResult FenceR = vkCreateFence(Dev, &FenceCI, nullptr, &FinishFence_);
  if (FenceR != VK_SUCCESS) {
    vkDestroySemaphore(Dev, TimelineSemaphore_, nullptr);
    TimelineSemaphore_ = VK_NULL_HANDLE;
    vkFreeCommandBuffers(Dev, CommandPool_, RingCapacity_,
                         CmdBufferRing_.data());
    vkDestroyCommandPool(Dev, CommandPool_, nullptr);
    CommandPool_ = VK_NULL_HANDLE;
    checkVk(FenceR, "CHIPQueueVulkan: vkCreateFence (FinishFence_) failed",
            hipErrorTbd);
  }

  // Every stream shares one VkQueue, so the priority is only reported.
  (void)Priority;
}

CHIPQueueVulkan::~CHIPQueueVulkan() {
  // A detached thread's queue can be destroyed after uninitialize() has
  // destroyed the VkDevice; then leak the handles to process exit.
  std::lock_guard<std::recursive_mutex> TeardownLock(
      CHIPBackendVulkan::TeardownMtx_);

  VkDevice Dev =
      ChipDevice_ ? ChipDevice_->getLogicalDevice() : VK_NULL_HANDLE;
  if (Dev == VK_NULL_HANDLE)
    return;
  if (CHIPBackendVulkan::ShuttingDown_.load(std::memory_order_acquire)) {
    FinishFence_ = VK_NULL_HANDLE;
    TimelineSemaphore_ = VK_NULL_HANDLE;
    CommandPool_ = VK_NULL_HANDLE;
    DescPool_ = VK_NULL_HANDLE;
    return;
  }

  VkQueue Q = ChipDevice_->getComputeQueue();
  if (Q != VK_NULL_HANDLE) {
    // Ignore the result; the device may already be lost.
    (void)vkQueueWaitIdle(Q);
  }

  if (FinishFence_ != VK_NULL_HANDLE) {
    vkDestroyFence(Dev, FinishFence_, nullptr);
    FinishFence_ = VK_NULL_HANDLE;
  }
  if (TimelineSemaphore_ != VK_NULL_HANDLE) {
    vkDestroySemaphore(Dev, TimelineSemaphore_, nullptr);
    TimelineSemaphore_ = VK_NULL_HANDLE;
  }
  if (CommandPool_ != VK_NULL_HANDLE) {
    vkDestroyCommandPool(Dev, CommandPool_, nullptr);
    CommandPool_ = VK_NULL_HANDLE;
  }
  if (DescPool_ != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(Dev, DescPool_, nullptr);
    DescPool_ = VK_NULL_HANDLE;
  }
}

void CHIPQueueVulkan::recordEvent(chipstar::Event *Event) {
  auto *EvVk = static_cast<CHIPEventVulkan *>(Event);
  if (EvVk == nullptr)
    CHIPERR_LOG_AND_THROW("CHIPQueueVulkan::recordEvent: null event",
                          hipErrorInvalidValue);
  if (EvVk->isIpcOpened())
    CHIPERR_LOG_AND_THROW("Recording an opened IPC event is not supported",
                          hipErrorNotSupported);
  auto CmdLock = lockCmdRecord();

  // Re-recording: the previous recording's fence and slot must be free.
  if (EvVk->getEventStatus() != EVENT_STATUS_INIT) {
    EvVk->wait();
    if (VkFence F = EvVk->getFence())
      vkResetFences(ChipDevice_->getLogicalDevice(), 1, &F);
    ChipDevice_->releaseTimestampSlot(EvVk->getTimestampSlot());
  }

  int32_t Slot = ChipDevice_->acquireTimestampSlot();
  EvVk->setTimestampSlot(Slot);

  // getElapsedTime() falls back to this without a timestamp slot.
  EvVk->getHostTimestamp() = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());

  VkCommandBuffer Cb = acquireCmdBuffer();
  VkCommandBufferBeginInfo BI{};
  BI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  BI.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  checkVk(vkBeginCommandBuffer(Cb, &BI),
          "CHIPQueueVulkan::recordEvent: vkBeginCommandBuffer failed",
          hipErrorTbd);

  if (Slot >= 0 && ChipDevice_->getTimestampQueryPool() != VK_NULL_HANDLE) {
    // A query must be reset before vkCmdWriteTimestamp writes it.
    vkCmdResetQueryPool(Cb, ChipDevice_->getTimestampQueryPool(),
                        static_cast<uint32_t>(Slot), 1);
    vkCmdWriteTimestamp(Cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        ChipDevice_->getTimestampQueryPool(),
                        static_cast<uint32_t>(Slot));
  }

  checkVk(vkEndCommandBuffer(Cb),
          "CHIPQueueVulkan::recordEvent: vkEndCommandBuffer failed",
          hipErrorTbd);

  uint64_t WaitTimelineVal, SignalTimelineVal;
  {
    std::lock_guard<std::mutex> Lock(QueueOpMtx_);
    WaitTimelineVal = TimelineValue_;
    SignalTimelineVal = ++TimelineValue_;
    noteRingSubmit(Cb, SignalTimelineVal);
  }
  // An exported IPC event also signals its own semaphore.
  VkSemaphore SignalSems[2] = {TimelineSemaphore_, EvVk->getIpcSemaphore()};
  uint64_t SignalVals[2] = {SignalTimelineVal, 0};
  uint32_t NumSignals = 1;
  if (SignalSems[1] != VK_NULL_HANDLE) {
    SignalVals[1] = EvVk->nextIpcValue();
    NumSignals = 2;
  }

  // Waits for the previous value, which may be a pending stream callback's.
  VkPipelineStageFlags WaitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  uint32_t NumWaits = WaitTimelineVal > 0 ? 1 : 0;
  VkTimelineSemaphoreSubmitInfo TsSubmit{};
  TsSubmit.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  TsSubmit.waitSemaphoreValueCount = NumWaits;
  TsSubmit.pWaitSemaphoreValues = &WaitTimelineVal;
  TsSubmit.signalSemaphoreValueCount = NumSignals;
  TsSubmit.pSignalSemaphoreValues = SignalVals;

  VkSubmitInfo Submit{};
  Submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  Submit.pNext = &TsSubmit;
  Submit.waitSemaphoreCount = NumWaits;
  Submit.pWaitSemaphores = &TimelineSemaphore_;
  Submit.pWaitDstStageMask = &WaitStage;
  Submit.commandBufferCount = 1;
  Submit.pCommandBuffers = &Cb;
  Submit.signalSemaphoreCount = NumSignals;
  Submit.pSignalSemaphores = SignalSems;

  VkFence SignalFence = EvVk->getFence();
  if (SignalFence == VK_NULL_HANDLE) {
    SignalFence = FinishFence_;
    (void)vkResetFences(ChipDevice_->getLogicalDevice(), 1, &SignalFence);
  }

  {
    // vkQueueSubmit requires external synchronization of the VkQueue.
    std::lock_guard<std::mutex> SubmitLock(ChipDevice_->getSubmitMtx());
    checkVk(vkQueueSubmit(ChipDevice_->getComputeQueue(), 1, &Submit,
                          SignalFence),
            "CHIPQueueVulkan::recordEvent: vkQueueSubmit failed", hipErrorTbd);
  }
  if (NumSignals == 2)
    EvVk->publishIpcValue(SignalVals[1]);

  IsEmptyQueue_.store(false);
  EvVk->setRecording();
  // hipStreamWaitEvent waits on an event's dependencies; the marker signals
  // once everything submitted before the record has completed.
  CmdLock.unlock();
  Event->addDependency(enqueueMarkerImpl());
}

// Staging buffers of host transfers, freed once their event's fence signals.
namespace {

struct VulkanPendingStaging {
  VmaAllocator Allocator = VK_NULL_HANDLE;
  VkBuffer Buffer = VK_NULL_HANDLE;
  VmaAllocation Allocation = VK_NULL_HANDLE;
  std::shared_ptr<chipstar::Event> SignalEvent;
};

inline std::mutex &i8StagingMtx() {
  static std::mutex Mtx;
  return Mtx;
}

inline std::vector<VulkanPendingStaging> &i8PendingStagings() {
  // Never destroyed: the device destructor drains it from exit handlers.
  static auto *List = new std::vector<VulkanPendingStaging>;
  return *List;
}

// Frees the staging buffers whose fence has signalled.
inline void i8DrainCompletedStagings(VkDevice Dev) {
  std::lock_guard<std::mutex> Lock(i8StagingMtx());
  auto &List = i8PendingStagings();
  auto It = List.begin();
  while (It != List.end()) {
    auto *Ev = static_cast<CHIPEventVulkan *>(It->SignalEvent.get());
    VkFence F = Ev ? Ev->getFence() : VK_NULL_HANDLE;
    bool Done = (F == VK_NULL_HANDLE) || Dev == VK_NULL_HANDLE ||
                vkGetFenceStatus(Dev, F) == VK_SUCCESS;
    if (Done) {
      if (It->Allocator != VK_NULL_HANDLE && It->Buffer != VK_NULL_HANDLE)
        vmaDestroyBuffer(It->Allocator, It->Buffer, It->Allocation);
      It = List.erase(It);
    } else {
      ++It;
    }
  }
}

inline void i8RecordPendingStaging(VmaAllocator Allocator, VkBuffer Buffer,
                                   VmaAllocation Allocation,
                                   std::shared_ptr<chipstar::Event> Event) {
  std::lock_guard<std::mutex> Lock(i8StagingMtx());
  i8PendingStagings().push_back(
      {Allocator, Buffer, Allocation, std::move(Event)});
}

// A persistently mapped, host coherent staging buffer.
inline bool i8AllocateStagingBuffer(VmaAllocator Allocator, size_t Size,
                                    VkBufferUsageFlags UsageFlags,
                                    VkBuffer &BufOut, VmaAllocation &AllocOut,
                                    void *&MappedOut) {
  VkBufferCreateInfo Info{};
  Info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  Info.size = Size;
  Info.usage = UsageFlags;
  Info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo AllocInfo{};
  AllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
  AllocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  AllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

  VmaAllocationInfo Result{};
  VkResult Status = vmaCreateBuffer(Allocator, &Info, &AllocInfo, &BufOut,
                                    &AllocOut, &Result);
  if (Status != VK_SUCCESS || BufOut == VK_NULL_HANDLE ||
      Result.pMappedData == nullptr)
    return false;
  MappedOut = Result.pMappedData;
  return true;
}

// Repeats Pattern over Dst; vkCmdFillBuffer only takes 4-byte patterns.
inline void i8TilePattern(void *Dst, size_t TotalSize, const void *Pattern,
                          size_t PatternSize) {
  auto *DstBytes = static_cast<uint8_t *>(Dst);
  const auto *PatBytes = static_cast<const uint8_t *>(Pattern);
  for (size_t i = 0; i < TotalSize; ++i)
    DstBytes[i] = PatBytes[i % PatternSize];
}

// The VkBuffer containing Ptr and Ptr's offset in it, or VK_NULL_HANDLE for
// host memory; IsMappedAlloc when the host can access it directly.
inline VkBuffer i8LookupVkBuffer(CHIPContextVulkan *Ctx, const void *Ptr,
                                 bool &IsMappedAlloc, VkDeviceSize &OffsetOut) {
  IsMappedAlloc = false;
  OffsetOut = 0;
  if (!Ctx || !Ptr)
    return VK_NULL_HANDLE;
  size_t Offset = 0;
  const auto *Entry = Ctx->getDevPtrEntryContaining(Ptr, Offset);
  if (!Entry)
    return VK_NULL_HANDLE;
  OffsetOut = static_cast<VkDeviceSize>(Offset);
  IsMappedAlloc = Entry->AllocInfo.pMappedData != nullptr;
  return Entry->Buffer;
}

// Throw hipErrorInvalidValue when [Ptr, Ptr + Size) runs past the end of the
// registered allocation containing Ptr (unregistered host memory is fine).
inline void i8CheckFitsAllocation(CHIPContextVulkan *Ctx, const void *Ptr,
                                  size_t Size) {
  size_t Offset = 0;
  const auto *Entry = Ctx ? Ctx->getDevPtrEntryContaining(Ptr, Offset) : nullptr;
  if (Entry && Size > Entry->Size - Offset)
    CHIPERR_LOG_AND_THROW("Access exceeds the allocation", hipErrorInvalidValue);
}

inline VkCommandBuffer i8BeginCmdBuffer(CHIPQueueVulkan *Q) {
  VkCommandBuffer Cmd = Q->acquireCmdBuffer();
  VkCommandBufferBeginInfo BI{};
  BI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  BI.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VkResult R = vkBeginCommandBuffer(Cmd, &BI);
  if (R != VK_SUCCESS)
    CHIPERR_LOG_AND_THROW(
        std::string("vkBeginCommandBuffer failed VkResult=") +
            std::to_string(static_cast<int>(R)),
        hipErrorRuntimeMemory);
  return Cmd;
}

inline void i8EndCmdBuffer(VkCommandBuffer Cmd) {
  VkResult R = vkEndCommandBuffer(Cmd);
  if (R != VK_SUCCESS)
    CHIPERR_LOG_AND_THROW(
        std::string("vkEndCommandBuffer failed VkResult=") +
            std::to_string(static_cast<int>(R)),
        hipErrorRuntimeMemory);
}

} // namespace

// ----------------------------------------------------------------------------
// memCopyAsyncImpl: host memory goes through a staging buffer, except when the
// allocation is host mapped, which a plain memcpy reads or writes directly.
// ----------------------------------------------------------------------------
void CHIPQueueVulkan::waitSubmitted() {
  // Through submitWithEvent, so the other streams' work this one must follow
  // (the legacy default stream) completes too.
  VkCommandBuffer Cmd = i8BeginCmdBuffer(this);
  i8EndCmdBuffer(Cmd);
  submitWithEvent(Cmd, {})->wait();
}

std::shared_ptr<chipstar::Event>
CHIPQueueVulkan::memCopyAsyncImpl(void *Dst, const void *Src, size_t Size,
                                  hipMemcpyKind Kind) {
  auto CmdLock = lockCmdRecord();
  CHIPContextVulkan *Ctx = getContext();
  CHIPDeviceVulkan *Dev = getVulkanDevice();
  VkDevice VkDev = Dev ? Dev->getLogicalDevice() : VK_NULL_HANDLE;
  VmaAllocator Allocator = Dev ? Dev->getAllocator() : VK_NULL_HANDLE;
  i8DrainCompletedStagings(VkDev);

  IsEmptyQueue_.store(false);

  if (Dst == Src || Size == 0) {
    logTrace("CHIPQueueVulkan::memCopyAsync no-op (Dst==Src or Size==0)");
    {
      VkCommandBuffer EmptyCmd = i8BeginCmdBuffer(this);
      i8EndCmdBuffer(EmptyCmd);
      return submitWithEvent(EmptyCmd, {});
    }
  }

  bool DstIsMapped = false, SrcIsMapped = false;
  VkDeviceSize DstOffset = 0, SrcOffset = 0;
  VkBuffer DstBuf = i8LookupVkBuffer(Ctx, Dst, DstIsMapped, DstOffset);
  VkBuffer SrcBuf = i8LookupVkBuffer(Ctx, Src, SrcIsMapped, SrcOffset);
  i8CheckFitsAllocation(Ctx, Dst, Size);
  i8CheckFitsAllocation(Ctx, Src, Size);

  if (Kind == hipMemcpyDefault) {
    if (DstBuf != VK_NULL_HANDLE && SrcBuf != VK_NULL_HANDLE)
      Kind = hipMemcpyDeviceToDevice;
    else if (DstBuf != VK_NULL_HANDLE)
      Kind = hipMemcpyHostToDevice;
    else if (SrcBuf != VK_NULL_HANDLE)
      Kind = hipMemcpyDeviceToHost;
    else
      Kind = hipMemcpyHostToHost;
  }

  // Copy on the host, and submit an empty command buffer for the event.
  if (Kind == hipMemcpyHostToHost ||
      (Kind == hipMemcpyHostToDevice && DstIsMapped) ||
      (Kind == hipMemcpyDeviceToHost && SrcIsMapped) ||
      (Kind == hipMemcpyDeviceToDevice && DstIsMapped && SrcIsMapped)) {
    logTrace("CHIPQueueVulkan::memCopyAsync host-side memcpy {} -> {} / {} B",
             Src, Dst, Size);
    waitSubmitted();
    std::memcpy(Dst, Src, Size);
    {
      VkCommandBuffer EmptyCmd = i8BeginCmdBuffer(this);
      i8EndCmdBuffer(EmptyCmd);
      return submitWithEvent(EmptyCmd, {});
    }
  }

  switch (Kind) {
  case hipMemcpyDeviceToDevice:
    if (DstBuf == VK_NULL_HANDLE || SrcBuf == VK_NULL_HANDLE)
      CHIPERR_LOG_AND_THROW("memCopyAsync D2D: both pointers must be device",
                            hipErrorInvalidValue);
    break;
  case hipMemcpyHostToDevice:
    if (DstBuf == VK_NULL_HANDLE)
      CHIPERR_LOG_AND_THROW("memCopyAsync H2D: destination is not a device "
                            "pointer registered with this context",
                            hipErrorInvalidValue);
    break;
  case hipMemcpyDeviceToHost:
    if (SrcBuf == VK_NULL_HANDLE)
      CHIPERR_LOG_AND_THROW("memCopyAsync D2H: source is not a device "
                            "pointer registered with this context",
                            hipErrorInvalidValue);
    break;
  default:
    CHIPERR_LOG_AND_THROW("memCopyAsync: unsupported hipMemcpyKind",
                          hipErrorInvalidValue);
  }

  VkCommandBuffer Cmd = i8BeginCmdBuffer(this);

  if (Kind == hipMemcpyDeviceToDevice) {
    VkBufferCopy Region{};
    Region.srcOffset = SrcOffset;
    Region.dstOffset = DstOffset;
    Region.size = Size;
    vkCmdCopyBuffer(Cmd, SrcBuf, DstBuf, 1, &Region);
    i8EndCmdBuffer(Cmd);
    auto Ev = submitWithEvent(Cmd, {});
    logTrace("CHIPQueueVulkan::memCopyAsync D2D {} -> {} / {} B", Src, Dst,
             Size);
    return Ev;
  }

  if (Allocator == VK_NULL_HANDLE)
    CHIPERR_LOG_AND_THROW("memCopyAsync: VMA allocator not initialized",
                          hipErrorRuntimeMemory);

  VkBuffer Staging = VK_NULL_HANDLE;
  VmaAllocation StagingAlloc = VK_NULL_HANDLE;
  void *StagingMapped = nullptr;
  VkBufferUsageFlags UsageFlags = (Kind == hipMemcpyHostToDevice)
                                      ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                      : VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (!i8AllocateStagingBuffer(Allocator, Size, UsageFlags, Staging,
                               StagingAlloc, StagingMapped))
    CHIPERR_LOG_AND_THROW(
        "memCopyAsync: failed to allocate host-visible staging buffer",
        hipErrorOutOfMemory);

  if (Kind == hipMemcpyHostToDevice) {
    std::memcpy(StagingMapped, Src, Size);
    VkBufferCopy Region{};
    Region.srcOffset = 0;
    Region.dstOffset = DstOffset;
    Region.size = Size;
    vkCmdCopyBuffer(Cmd, Staging, DstBuf, 1, &Region);
    i8EndCmdBuffer(Cmd);
    auto Ev = submitWithEvent(Cmd, {});
    i8RecordPendingStaging(Allocator, Staging, StagingAlloc, Ev);
    logTrace("CHIPQueueVulkan::memCopyAsync H2D {} -> {} / {} B (staged)", Src,
             Dst, Size);
    return Ev;
  }

  // Device to host waits: only the host can copy out of the staging buffer.
  {
    VkBufferCopy Region{};
    Region.srcOffset = SrcOffset;
    Region.dstOffset = 0;
    Region.size = Size;
    vkCmdCopyBuffer(Cmd, SrcBuf, Staging, 1, &Region);
    i8EndCmdBuffer(Cmd);
    auto Ev = submitWithEvent(Cmd, {});

    auto *EvVk = static_cast<CHIPEventVulkan *>(Ev.get());
    VkFence F = EvVk ? EvVk->getFence() : VK_NULL_HANDLE;
    if (F != VK_NULL_HANDLE && VkDev != VK_NULL_HANDLE) {
      VkResult WaitRes = vkWaitForFences(VkDev, 1, &F, VK_TRUE, UINT64_MAX);
      if (WaitRes != VK_SUCCESS)
        CHIPERR_LOG_AND_THROW(
            std::string("memCopyAsync D2H: vkWaitForFences failed VkResult=") +
                std::to_string(static_cast<int>(WaitRes)),
            hipErrorRuntimeMemory);
    }
    std::memcpy(Dst, StagingMapped, Size);
    vmaDestroyBuffer(Allocator, Staging, StagingAlloc);
    logTrace("CHIPQueueVulkan::memCopyAsync D2H {} -> {} / {} B (staged)", Src,
             Dst, Size);
    return Ev;
  }
}

// ----------------------------------------------------------------------------
// memFillAsyncImpl: vkCmdFillBuffer for an aligned 4-byte pattern, else a copy
// from a staging buffer holding the repeated pattern.
// ----------------------------------------------------------------------------
std::shared_ptr<chipstar::Event>
CHIPQueueVulkan::memFillAsyncImpl(void *Dst, size_t Size, const void *Pattern,
                                  size_t PatternSize) {
  auto CmdLock = lockCmdRecord();
  CHIPContextVulkan *Ctx = getContext();
  CHIPDeviceVulkan *Dev = getVulkanDevice();
  VkDevice VkDev = Dev ? Dev->getLogicalDevice() : VK_NULL_HANDLE;
  VmaAllocator Allocator = Dev ? Dev->getAllocator() : VK_NULL_HANDLE;
  i8DrainCompletedStagings(VkDev);

  IsEmptyQueue_.store(false);

  if (PatternSize == 0)
    CHIPERR_LOG_AND_THROW("memFillAsync: PatternSize must be > 0",
                          hipErrorInvalidValue);

  bool DstIsMapped = false;
  VkDeviceSize DstOffset = 0;
  VkBuffer DstBuf = i8LookupVkBuffer(Ctx, Dst, DstIsMapped, DstOffset);
  i8CheckFitsAllocation(Ctx, Dst, Size);

  // A host mapped allocation's pointer is its host address.
  if (DstBuf == VK_NULL_HANDLE || DstIsMapped) {
    waitSubmitted();
    logTrace("CHIPQueueVulkan::memFillAsync host-side fill {} / {} B (pat {})",
             Dst, Size, PatternSize);
    i8TilePattern(Dst, Size, Pattern, PatternSize);
    {
      VkCommandBuffer EmptyCmd = i8BeginCmdBuffer(this);
      i8EndCmdBuffer(EmptyCmd);
      return submitWithEvent(EmptyCmd, {});
    }
  }

  if (Size == 0)
    {
      VkCommandBuffer EmptyCmd = i8BeginCmdBuffer(this);
      i8EndCmdBuffer(EmptyCmd);
      return submitWithEvent(EmptyCmd, {});
    }

  VkCommandBuffer Cmd = i8BeginCmdBuffer(this);

  // vkCmdFillBuffer requires dstOffset and size to be multiples of 4.
  if (PatternSize == 4 && (Size % 4) == 0 && (DstOffset % 4) == 0) {
    uint32_t Value = 0;
    std::memcpy(&Value, Pattern, sizeof(Value));
    vkCmdFillBuffer(Cmd, DstBuf, DstOffset, Size, Value);
    i8EndCmdBuffer(Cmd);
    auto Ev = submitWithEvent(Cmd, {});
    logTrace(
        "CHIPQueueVulkan::memFillAsync vkCmdFillBuffer {} / {} B (value=0x{:x})",
        Dst, Size, Value);
    return Ev;
  }

  if (Allocator == VK_NULL_HANDLE)
    CHIPERR_LOG_AND_THROW("memFillAsync: VMA allocator not initialized",
                          hipErrorRuntimeMemory);

  VkBuffer Staging = VK_NULL_HANDLE;
  VmaAllocation StagingAlloc = VK_NULL_HANDLE;
  void *StagingMapped = nullptr;
  if (!i8AllocateStagingBuffer(Allocator, Size,
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT, Staging,
                               StagingAlloc, StagingMapped))
    CHIPERR_LOG_AND_THROW("memFillAsync: failed to allocate staging buffer",
                          hipErrorOutOfMemory);

  i8TilePattern(StagingMapped, Size, Pattern, PatternSize);
  VkBufferCopy Region{};
  Region.srcOffset = 0;
  Region.dstOffset = DstOffset;
  Region.size = Size;
  vkCmdCopyBuffer(Cmd, Staging, DstBuf, 1, &Region);
  i8EndCmdBuffer(Cmd);
  auto Ev = submitWithEvent(Cmd, {});
  i8RecordPendingStaging(Allocator, Staging, StagingAlloc, Ev);
  logTrace(
      "CHIPQueueVulkan::memFillAsync staged-tile {} / {} B (PatternSize={})",
      Dst, Size, PatternSize);
  return Ev;
}

std::shared_ptr<chipstar::Event> CHIPQueueVulkan::memCopy2DAsyncImpl(
    void *Dst, size_t DPitch, const void *Src, size_t SPitch, size_t Width,
    size_t Height, hipMemcpyKind Kind) {
  UNIMPLEMENTED(nullptr);
}

std::shared_ptr<chipstar::Event> CHIPQueueVulkan::memCopy3DAsyncImpl(
    void *Dst, size_t DPitch, size_t DSPitch, const void *Src, size_t SPitch,
    size_t SSPitch, size_t Width, size_t Height, size_t Depth,
    hipMemcpyKind Kind) {
  UNIMPLEMENTED(nullptr);
}

// ----------------------------------------------------------------------------
// memFillAsync2D: one submit for all rows.
// ----------------------------------------------------------------------------
void CHIPQueueVulkan::memFillAsync2D(void *Dst, size_t Pitch, int Value,
                                     size_t Width, size_t Height) {
  if (Width == 0 || Height == 0)
    return;

  auto CmdLock = lockCmdRecord();
  CHIPContextVulkan *Ctx = getContext();
  CHIPDeviceVulkan *Dev = getVulkanDevice();
  VkDevice VkDev = Dev ? Dev->getLogicalDevice() : VK_NULL_HANDLE;
  VmaAllocator Allocator = Dev ? Dev->getAllocator() : VK_NULL_HANDLE;
  i8DrainCompletedStagings(VkDev);
  IsEmptyQueue_.store(false);

  bool DstIsMapped = false;
  VkDeviceSize DstOffset = 0;
  VkBuffer DstBuf = i8LookupVkBuffer(Ctx, Dst, DstIsMapped, DstOffset);

  if (DstBuf == VK_NULL_HANDLE || DstIsMapped) {
    waitSubmitted();
    unsigned char Byte = static_cast<unsigned char>(Value);
    for (size_t R = 0; R < Height; ++R) {
      void *Row = static_cast<uint8_t *>(Dst) + R * Pitch;
      std::memset(Row, Byte, Width);
    }
    VkCommandBuffer EmptyCmd = i8BeginCmdBuffer(this);
    i8EndCmdBuffer(EmptyCmd);
    auto Ev = submitWithEvent(EmptyCmd, {});
    Ev->Msg = "memFillAsync2D";
    return;
  }

  VkCommandBuffer Cmd = i8BeginCmdBuffer(this);

  // vkCmdFillBuffer requires dstOffset and size to be multiples of 4.
  if ((Width % 4) == 0 && (Pitch % 4) == 0 && (DstOffset % 4) == 0) {
    unsigned char Byte = static_cast<unsigned char>(Value);
    uint32_t Word = static_cast<uint32_t>(Byte);
    Word |= Word << 8;
    Word |= Word << 16;
    for (size_t R = 0; R < Height; ++R)
      vkCmdFillBuffer(Cmd, DstBuf, DstOffset + R * Pitch, Width, Word);
    i8EndCmdBuffer(Cmd);
    auto Ev = submitWithEvent(Cmd, {});
    Ev->Msg = "memFillAsync2D";
    return;
  }

  // Otherwise copy one staged row into every row.
  if (Allocator == VK_NULL_HANDLE)
    CHIPERR_LOG_AND_THROW("memFillAsync2D: VMA allocator not initialized",
                          hipErrorRuntimeMemory);

  VkBuffer Staging = VK_NULL_HANDLE;
  VmaAllocation StagingAlloc = VK_NULL_HANDLE;
  void *StagingMapped = nullptr;
  if (!i8AllocateStagingBuffer(Allocator, Width,
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT, Staging,
                               StagingAlloc, StagingMapped))
    CHIPERR_LOG_AND_THROW("memFillAsync2D: failed to allocate staging buffer",
                          hipErrorOutOfMemory);
  unsigned char Byte = static_cast<unsigned char>(Value);
  std::memset(StagingMapped, Byte, Width);

  std::vector<VkBufferCopy> Regions;
  Regions.reserve(Height);
  for (size_t R = 0; R < Height; ++R) {
    VkBufferCopy Reg{};
    Reg.srcOffset = 0;
    Reg.dstOffset = DstOffset + R * Pitch;
    Reg.size = Width;
    Regions.push_back(Reg);
  }
  vkCmdCopyBuffer(Cmd, Staging, DstBuf, static_cast<uint32_t>(Regions.size()),
                  Regions.data());
  i8EndCmdBuffer(Cmd);
  auto Ev = submitWithEvent(Cmd, {});
  i8RecordPendingStaging(Allocator, Staging, StagingAlloc, Ev);
  Ev->Msg = "memFillAsync2D";
}

// ----------------------------------------------------------------------------
// memFillAsync3D: as memFillAsync2D, over Depth * Height rows.
// ----------------------------------------------------------------------------
void CHIPQueueVulkan::memFillAsync3D(hipPitchedPtr PitchedDevPtr, int Value,
                                     hipExtent Extent) {
  const size_t Width = Extent.width;
  const size_t Height = Extent.height;
  const size_t Depth = Extent.depth;
  if (Width == 0 || Height == 0 || Depth == 0)
    return;

  auto CmdLock = lockCmdRecord();

  void *Dst = PitchedDevPtr.ptr;
  const size_t Pitch = PitchedDevPtr.pitch;
  const size_t SlicePitch = Pitch * PitchedDevPtr.ysize;

  CHIPContextVulkan *Ctx = getContext();
  CHIPDeviceVulkan *Dev = getVulkanDevice();
  VkDevice VkDev = Dev ? Dev->getLogicalDevice() : VK_NULL_HANDLE;
  VmaAllocator Allocator = Dev ? Dev->getAllocator() : VK_NULL_HANDLE;
  i8DrainCompletedStagings(VkDev);
  IsEmptyQueue_.store(false);

  bool DstIsMapped = false;
  VkDeviceSize DstOffset = 0;
  VkBuffer DstBuf = i8LookupVkBuffer(Ctx, Dst, DstIsMapped, DstOffset);

  if (DstBuf == VK_NULL_HANDLE || DstIsMapped) {
    waitSubmitted();
    unsigned char Byte = static_cast<unsigned char>(Value);
    for (size_t S = 0; S < Depth; ++S)
      for (size_t R = 0; R < Height; ++R) {
        void *Row = static_cast<uint8_t *>(Dst) + S * SlicePitch + R * Pitch;
        std::memset(Row, Byte, Width);
      }
    VkCommandBuffer EmptyCmd = i8BeginCmdBuffer(this);
    i8EndCmdBuffer(EmptyCmd);
    auto Ev = submitWithEvent(EmptyCmd, {});
    Ev->Msg = "memFillAsync3D";
    return;
  }

  VkCommandBuffer Cmd = i8BeginCmdBuffer(this);

  if ((Width % 4) == 0 && (Pitch % 4) == 0 && (DstOffset % 4) == 0 &&
      (SlicePitch % 4) == 0) {
    unsigned char Byte = static_cast<unsigned char>(Value);
    uint32_t Word = static_cast<uint32_t>(Byte);
    Word |= Word << 8;
    Word |= Word << 16;
    // One fill per contiguous slice, or for the whole contiguous volume.
    const bool RowsContig = (Pitch == Width);
    const bool SlicesContig = RowsContig && (SlicePitch == Height * Pitch);
    if (SlicesContig) {
      vkCmdFillBuffer(Cmd, DstBuf, DstOffset, Width * Height * Depth, Word);
    } else if (RowsContig) {
      for (size_t S = 0; S < Depth; ++S)
        vkCmdFillBuffer(Cmd, DstBuf, DstOffset + S * SlicePitch,
                        Width * Height, Word);
    } else {
      for (size_t S = 0; S < Depth; ++S)
        for (size_t R = 0; R < Height; ++R)
          vkCmdFillBuffer(Cmd, DstBuf,
                          DstOffset + S * SlicePitch + R * Pitch, Width, Word);
    }
    i8EndCmdBuffer(Cmd);
    auto Ev = submitWithEvent(Cmd, {});
    Ev->Msg = "memFillAsync3D";
    return;
  }

  if (Allocator == VK_NULL_HANDLE)
    CHIPERR_LOG_AND_THROW("memFillAsync3D: VMA allocator not initialized",
                          hipErrorRuntimeMemory);

  VkBuffer Staging = VK_NULL_HANDLE;
  VmaAllocation StagingAlloc = VK_NULL_HANDLE;
  void *StagingMapped = nullptr;
  if (!i8AllocateStagingBuffer(Allocator, Width,
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT, Staging,
                               StagingAlloc, StagingMapped))
    CHIPERR_LOG_AND_THROW("memFillAsync3D: failed to allocate staging buffer",
                          hipErrorOutOfMemory);
  unsigned char Byte = static_cast<unsigned char>(Value);
  std::memset(StagingMapped, Byte, Width);

  std::vector<VkBufferCopy> Regions;
  Regions.reserve(Depth * Height);
  for (size_t S = 0; S < Depth; ++S)
    for (size_t R = 0; R < Height; ++R) {
      VkBufferCopy Reg{};
      Reg.srcOffset = 0;
      Reg.dstOffset = DstOffset + S * SlicePitch + R * Pitch;
      Reg.size = Width;
      Regions.push_back(Reg);
    }
  vkCmdCopyBuffer(Cmd, Staging, DstBuf, static_cast<uint32_t>(Regions.size()),
                  Regions.data());
  i8EndCmdBuffer(Cmd);
  auto Ev = submitWithEvent(Cmd, {});
  i8RecordPendingStaging(Allocator, Staging, StagingAlloc, Ev);
  Ev->Msg = "memFillAsync3D";
}

// ============================================================================
// launchImpl
// ============================================================================

std::shared_ptr<chipstar::Event>
CHIPQueueVulkan::launchImpl(chipstar::ExecItem *ExecItem) {
  auto CmdLock = lockCmdRecord();
  if (!ExecItem)
    CHIPERR_LOG_AND_THROW("CHIPQueueVulkan::launchImpl received null ExecItem",
                          hipErrorInvalidValue);

  auto *VkExecItem = static_cast<CHIPExecItemVulkan *>(ExecItem);
  auto *Kernel =
      static_cast<CHIPKernelVulkan *>(VkExecItem->getVulkanKernel());
  if (!Kernel)
    CHIPERR_LOG_AND_THROW(
        "CHIPQueueVulkan::launchImpl: ExecItem has no kernel bound",
        hipErrorInvalidValue);

  auto *Module = Kernel->getVulkanModule();
  if (!Module)
    CHIPERR_LOG_AND_THROW(
        "CHIPQueueVulkan::launchImpl: kernel has no parent module",
        hipErrorInvalidValue);

  const std::string KernelName = Kernel->getName();
  const VulkanKernelReflection *Refl = Module->getReflection(KernelName);
  if (!Refl)
    CHIPERR_LOG_AND_THROW(
        "CHIPQueueVulkan::launchImpl: no reflection record for kernel",
        hipErrorInvalidValue);

  dim3 Grid = VkExecItem->getGrid();
  dim3 Block = VkExecItem->getBlock();
  VkPipeline Pipeline = Module->getOrCreatePipeline(
      KernelName, Block, VkExecItem->getSharedMem());
  VkDescriptorSetLayout DSLayout =
      Module->getOrCreateDescriptorSetLayout(KernelName);
  VkPipelineLayout PLLayout = Module->getOrCreatePipelineLayout(KernelName);
  if (Pipeline == VK_NULL_HANDLE || PLLayout == VK_NULL_HANDLE)
    CHIPERR_LOG_AND_THROW(
        "CHIPQueueVulkan::launchImpl: failed to build pipeline / layout",
        hipErrorInvalidValue);

  VkDevice Device = ChipDevice_->getLogicalDevice();

  if (DescPool_ == VK_NULL_HANDLE) {
    VkDescriptorPoolSize PoolSize{};
    PoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    PoolSize.descriptorCount = 32 * 1024;
    VkDescriptorPoolCreateInfo PoolInfo{};
    PoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    PoolInfo.maxSets = 1024;
    PoolInfo.poolSizeCount = 1;
    PoolInfo.pPoolSizes = &PoolSize;
    checkVk(vkCreateDescriptorPool(Device, &PoolInfo, nullptr, &DescPool_),
            "vkCreateDescriptorPool failed in launchImpl", hipErrorOutOfMemory);
  }
  VkDescriptorPool DescPool = DescPool_;
  VkDescriptorSet DescSet = VK_NULL_HANDLE;
  {
    VkDescriptorSetAllocateInfo AllocInfo{};
    AllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    AllocInfo.descriptorPool = DescPool;
    AllocInfo.descriptorSetCount = 1;
    AllocInfo.pSetLayouts = &DSLayout;
    VkResult R = vkAllocateDescriptorSets(Device, &AllocInfo, &DescSet);
    if (R == VK_ERROR_OUT_OF_POOL_MEMORY || R == VK_ERROR_FRAGMENTED_POOL) {
      // Pool exhausted: once this queue's submitted work (the only user of
      // its pool) has finished, every set can be recycled.
      uint64_t Target;
      {
        std::lock_guard<std::mutex> Lock(QueueOpMtx_);
        Target = TimelineValue_;
      }
      VkSemaphoreWaitInfo WI{};
      WI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
      WI.semaphoreCount = 1;
      WI.pSemaphores = &TimelineSemaphore_;
      WI.pValues = &Target;
      vkWaitSemaphores(Device, &WI, UINT64_MAX);
      vkResetDescriptorPool(Device, DescPool, 0);
      DescSet = VK_NULL_HANDLE;
      R = vkAllocateDescriptorSets(Device, &AllocInfo, &DescSet);
    }
    if (R != VK_SUCCESS || DescSet == VK_NULL_HANDLE)
      CHIPERR_LOG_AND_THROW("vkAllocateDescriptorSets failed in launchImpl",
                            hipErrorOutOfMemory);
  }

  const auto &Bindings = VkExecItem->getBufferBindings();
  const auto &Ranges = VkExecItem->getBufferRanges();
  if (Bindings.size() != Ranges.size())
    CHIPERR_LOG_AND_THROW(
        "CHIPQueueVulkan::launchImpl: BufferBindings/Ranges size mismatch",
        hipErrorInvalidValue);

  std::vector<VkDescriptorBufferInfo> BufInfos;
  std::vector<VkWriteDescriptorSet> Writes;
  BufInfos.reserve(Bindings.size() + 1); // + the POD argument buffer
  Writes.reserve(Bindings.size());
  for (uint32_t I = 0; I < Bindings.size(); ++I) {
    VkBuffer Buf = Bindings[I];
    if (Buf == VK_NULL_HANDLE)
      continue; // A binding no argument uses.
    VkDescriptorBufferInfo &BI = BufInfos.emplace_back();
    BI.buffer = Buf;
    BI.offset = 0;
    BI.range = Ranges[I] == 0 ? VK_WHOLE_SIZE : Ranges[I];

    VkWriteDescriptorSet &W = Writes.emplace_back();
    W = {};
    W.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    W.dstSet = DescSet;
    W.dstBinding = I;
    W.dstArrayElement = 0;
    W.descriptorCount = 1;
    W.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    W.pBufferInfo = &BI;
  }
  const auto &PCBlob = VkExecItem->getPushConstantBlob();
  VkBuffer PodBuf = VK_NULL_HANDLE;
  if (Refl->PodBufferBinding >= 0) {
    auto *Ctx = static_cast<CHIPContextVulkan *>(ChipDevice_->getContext());
    if (const auto *E = Ctx->getDevPtrEntry(Ctx->getPodArgBuffer()))
      PodBuf = E->Buffer;
    if (PodBuf == VK_NULL_HANDLE ||
        PCBlob.size() > CHIPContextVulkan::PodArgBufferSize)
      CHIPERR_LOG_AND_THROW("CHIPQueueVulkan::launchImpl: kernel arguments "
                            "exceed the POD argument buffer",
                            hipErrorLaunchFailure);
    VkDescriptorBufferInfo &BI = BufInfos.emplace_back();
    BI = {PodBuf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet &W = Writes.emplace_back();
    W = {};
    W.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    W.dstSet = DescSet;
    W.dstBinding = static_cast<uint32_t>(Refl->PodBufferBinding);
    W.descriptorCount = 1;
    W.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    W.pBufferInfo = &BI;
  }
  if (!Writes.empty())
    vkUpdateDescriptorSets(Device, static_cast<uint32_t>(Writes.size()),
                           Writes.data(), 0, nullptr);

  VkCommandBuffer Cmd = acquireCmdBuffer();
  if (Cmd == VK_NULL_HANDLE)
    CHIPERR_LOG_AND_THROW(
        "CHIPQueueVulkan::launchImpl: acquireCmdBuffer returned null",
        hipErrorTbd);

  VkCommandBufferBeginInfo BeginInfo{};
  BeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  BeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(Cmd, &BeginInfo) != VK_SUCCESS)
    CHIPERR_LOG_AND_THROW("vkBeginCommandBuffer failed in launchImpl",
                          hipErrorTbd);

  vkCmdBindPipeline(Cmd, VK_PIPELINE_BIND_POINT_COMPUTE, Pipeline);
  vkCmdBindDescriptorSets(Cmd, VK_PIPELINE_BIND_POINT_COMPUTE, PLLayout,
                          /*firstSet=*/0, 1, &DescSet,
                          /*dynamicOffsetCount=*/0, nullptr);

  if (PodBuf != VK_NULL_HANDLE && !PCBlob.empty()) {
    // All queues share one VkQueue, so the barriers order this update after
    // every earlier dispatch reading the buffer and before this one.
    VkMemoryBarrier MB{};
    MB.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    MB.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    MB.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(Cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &MB, 0, nullptr,
                         0, nullptr);
    vkCmdUpdateBuffer(Cmd, PodBuf, 0, PCBlob.size(), PCBlob.data());
    MB.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    MB.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(Cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &MB, 0,
                         nullptr, 0, nullptr);
  } else if (Refl->PushConstantBlockSize > 0 && !PCBlob.empty()) {
    const uint32_t PCSize =
        std::min<uint32_t>(Refl->PushConstantBlockSize,
                           static_cast<uint32_t>(PCBlob.size()));
    vkCmdPushConstants(Cmd, PLLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       /*offset=*/0, PCSize, PCBlob.data());
  }

  vkCmdDispatch(Cmd, Grid.x, Grid.y, Grid.z);

  if (vkEndCommandBuffer(Cmd) != VK_SUCCESS)
    CHIPERR_LOG_AND_THROW("vkEndCommandBuffer failed in launchImpl",
                          hipErrorTbd);

  IsEmptyQueue_.store(false);

  std::vector<std::shared_ptr<chipstar::Event>> NoWaits;
  std::shared_ptr<chipstar::Event> LaunchEvent =
      submitWithEvent(Cmd, NoWaits);
  if (LaunchEvent)
    LaunchEvent->Msg = "KernelLaunch";
  return LaunchEvent;
}

std::vector<chipstar::DeviceVar *> CHIPDeviceVulkan::getDevicePrintfBuffers() {
  std::vector<chipstar::DeviceVar *> Out;
  // An uninitialized buffer holds whatever its allocation held before.
  for (auto *Mod : getCompiledModules())
    for (auto *V : Mod->getDeviceVariables())
      if (Mod->deviceVariablesInitialized() &&
          V->getName() == "__hipspv_printf_buf" && V->getDevAddr())
        Out.push_back(V);
  return Out;
}

// Format one record written by HIPSPVLowerToHLSLShape's device printf:
// [nwords, fmt_bytes | stderr<<31, fmt words, {tag, lo, hi}...].
static void printDevicePrintfRecord(const uint32_t *W, uint32_t NW) {
  uint32_t FmtBytes = W[1] & 0x7fffffffu;
  size_t FmtWords = (FmtBytes + 3) / 4;
  if (2 + FmtWords > NW)
    return;
  std::string Fmt(reinterpret_cast<const char *>(&W[2]), FmtBytes);
  struct Arg {
    uint32_t Tag;
    uint64_t V;
    std::string S;
  };
  std::vector<Arg> Args;
  for (size_t I = 2 + FmtWords; I + 3 <= NW;) {
    Arg A{W[I], W[I + 1] | (uint64_t(W[I + 2]) << 32), {}};
    I += 3;
    if (A.Tag == 4) {
      // The high word, when set, is the padded word count of a string
      // chosen at run time among several literals.
      size_t Bytes = A.V & 0xffffffffu,
             Words = (A.V >> 32) ? (A.V >> 32) : (Bytes + 3) / 4;
      if (I + Words > NW)
        break;
      A.S.assign(reinterpret_cast<const char *>(&W[I]), Bytes);
      I += Words;
    }
    Args.push_back(std::move(A));
  }
  std::string Out;
  size_t AI = 0;
  for (size_t P = 0; P < Fmt.size();) {
    if (Fmt[P] != '%') {
      Out += Fmt[P++];
      continue;
    }
    size_t Q = P + 1;
    if (Q < Fmt.size() && Fmt[Q] == '%') {
      Out += '%';
      P = Q + 1;
      continue;
    }
    std::string Spec = "%";
    for (; Q < Fmt.size() && strchr("-+ #0123456789.*", Fmt[Q]); ++Q)
      Spec += Fmt[Q] != '*' ? std::string(1, Fmt[Q])
              : AI < Args.size() ? std::to_string(int(Args[AI++].V)) : "0";
    while (Q < Fmt.size() && strchr("hlLqjzt", Fmt[Q]))
      ++Q;
    if (Q >= Fmt.size())
      break;
    char Conv = Fmt[Q];
    P = Q + 1;
    if (AI >= Args.size())
      continue;
    const Arg &A = Args[AI++];
    char Buf[512] = {0};
    uint64_t U = A.Tag == 1 ? uint64_t(uint32_t(A.V)) : A.V;
    double D;
    std::memcpy(&D, &A.V, sizeof(D));
    switch (Conv) {
    case 'd':
    case 'i':
      snprintf(Buf, sizeof(Buf), (Spec + "lld").c_str(), (long long)A.V);
      break;
    case 'u':
    case 'x':
    case 'X':
    case 'o':
      snprintf(Buf, sizeof(Buf), (Spec + "ll" + Conv).c_str(),
               (unsigned long long)U);
      break;
    case 'c':
      snprintf(Buf, sizeof(Buf), (Spec + "c").c_str(), (int)A.V);
      break;
    case 's':
      snprintf(Buf, sizeof(Buf), (Spec + "s").c_str(), A.S.c_str());
      break;
    case 'p':
      // Tag 3: a buffer pointer, whose address the device cannot record.
      snprintf(Buf, sizeof(Buf), A.Tag == 3 ? "(unknown)" : "0x%llx",
               (unsigned long long)A.V);
      break;
    default:
      if (Conv && strchr("fFeEgGaA", Conv))
        snprintf(Buf, sizeof(Buf), (Spec + Conv).c_str(), D);
      break;
    }
    Out += Buf;
  }
  fputs(Out.c_str(), (W[1] >> 31) ? stderr : stdout);
}

void CHIPQueueVulkan::drainDevicePrintf() {
  if (Draining_)
    return;
  auto *Dev = static_cast<CHIPDeviceVulkan *>(ChipDevice_);
  auto Buffers = Dev->getDevicePrintfBuffers();
  // A callback waiting for the queue would wait on itself.
  if (Buffers.empty() || InCallbackThread)
    return;
  {
    // Kernels of every stream write the same buffers.
    std::lock_guard<std::mutex> SubmitLock(Dev->getSubmitMtx());
    checkVk(vkQueueWaitIdle(Dev->getComputeQueue()),
            "CHIPQueueVulkan::drainDevicePrintf: vkQueueWaitIdle failed",
            hipErrorTbd);
  }
  Draining_ = true;
  bool Abort = false;
  for (auto *V : Buffers) {
    uint32_t Hdr[2] = {0, 0};
    memCopyAsyncImpl(Hdr, V->getDevAddr(), sizeof(Hdr),
                     hipMemcpyDeviceToHost);
    if (!Hdr[0] && !Hdr[1])
      continue;
    size_t Cap = V->getSize() / sizeof(uint32_t) - 3;
    size_t Used = std::min<size_t>(Hdr[0], Cap);
    std::vector<uint32_t> Data(Used);
    if (Used)
      memCopyAsyncImpl(Data.data(),
                       static_cast<char *>(V->getDevAddr()) + sizeof(Hdr),
                       Used * sizeof(uint32_t), hipMemcpyDeviceToHost);
    for (size_t I = 0; I < Used;) {
      uint32_t NW = Data[I];
      if (NW < 2 || I + NW > Used)
        break;
      printDevicePrintfRecord(&Data[I], NW);
      I += NW;
    }
    Abort |= Hdr[1] != 0;
    static const uint32_t Zero[2] = {0, 0};
    auto Ev = memCopyAsyncImpl(V->getDevAddr(), Zero, sizeof(Zero),
                               hipMemcpyHostToDevice);
    if (auto *EvVk = static_cast<CHIPEventVulkan *>(Ev.get()))
      if (VkFence F = EvVk->getFence())
        vkWaitForFences(ChipDevice_->getLogicalDevice(), 1, &F, VK_TRUE,
                        UINT64_MAX);
  }
  fflush(stdout);
  fflush(stderr);
  Draining_ = false;
  if (Abort && !getenv("CHIP_HOST_IGNORES_DEVICE_ABORT"))
    abort();
}

void CHIPQueueVulkan::finish() {
  // Waits for this queue's submits and stream callbacks, unless called from a
  // callback, which would wait on itself.
  if (!InCallbackThread && TimelineSemaphore_ != VK_NULL_HANDLE) {
    uint64_t Target;
    {
      std::lock_guard<std::mutex> Lock(QueueOpMtx_);
      Target = TimelineValue_;
    }
    VkSemaphoreWaitInfo WI{};
    WI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    WI.semaphoreCount = 1;
    WI.pSemaphores = &TimelineSemaphore_;
    WI.pValues = &Target;
    checkVk(vkWaitSemaphores(ChipDevice_->getLogicalDevice(), &WI, UINT64_MAX),
            "CHIPQueueVulkan::finish: vkWaitSemaphores failed", hipErrorTbd);
  }
  drainDevicePrintf();
  IsEmptyQueue_.store(true);
}

uint64_t CHIPQueueVulkan::reserveTimelineValue() {
  std::lock_guard<std::mutex> Lock(QueueOpMtx_);
  return ++TimelineValue_;
}

void CHIPQueueVulkan::signalTimelineValue(uint64_t Value) {
  VkSemaphoreSignalInfo SI{};
  SI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
  SI.semaphore = TimelineSemaphore_;
  SI.value = Value;
  checkVk(vkSignalSemaphore(ChipDevice_->getLogicalDevice(), &SI),
          "CHIPQueueVulkan: vkSignalSemaphore failed", hipErrorTbd);
}

bool CHIPQueueVulkan::query() {
  if (TimelineSemaphore_ == VK_NULL_HANDLE)
    return true;
  uint64_t Current = 0;
  checkVk(vkGetSemaphoreCounterValue(ChipDevice_->getLogicalDevice(),
                                     TimelineSemaphore_, &Current),
          "CHIPQueueVulkan::query: vkGetSemaphoreCounterValue failed",
          hipErrorTbd);

  uint64_t Target;
  {
    std::lock_guard<std::mutex> Lock(QueueOpMtx_);
    Target = TimelineValue_;
  }
  return Current >= Target;
}

std::shared_ptr<chipstar::Event> CHIPQueueVulkan::enqueueBarrierImpl(
    const std::vector<std::shared_ptr<chipstar::Event>> &EventsToWaitFor) {
  // A VkFence can only be waited on by the host.
  auto CmdLock = lockCmdRecord();
  for (auto &E : EventsToWaitFor) {
    auto *Ev = static_cast<CHIPEventVulkan *>(E.get());
    if (Ev && Ev->getFence() != VK_NULL_HANDLE) {
      VkFence Wf = Ev->getFence();
      (void)vkWaitForFences(ChipDevice_->getLogicalDevice(), 1, &Wf,
                            VK_TRUE, UINT64_MAX);
    }
  }

  VkCommandBuffer Cb = acquireCmdBuffer();
  VkCommandBufferBeginInfo BI{};
  BI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  BI.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  checkVk(vkBeginCommandBuffer(Cb, &BI),
          "CHIPQueueVulkan::enqueueBarrierImpl: vkBeginCommandBuffer failed",
          hipErrorTbd);

  VkMemoryBarrier MemBar{};
  MemBar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  MemBar.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
  MemBar.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  vkCmdPipelineBarrier(Cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       /*dependencyFlags=*/0, 1, &MemBar, 0, nullptr, 0,
                       nullptr);

  checkVk(vkEndCommandBuffer(Cb),
          "CHIPQueueVulkan::enqueueBarrierImpl: vkEndCommandBuffer failed",
          hipErrorTbd);

  auto Marker = submitWithEvent(Cb, {});
  // Keeps the awaited events alive until the marker completes.
  for (auto &E : EventsToWaitFor)
    if (E)
      Marker->addDependency(E);
  return Marker;
}

std::shared_ptr<chipstar::Event> CHIPQueueVulkan::enqueueMarkerImpl() {
  auto CmdLock = lockCmdRecord();
  VkCommandBuffer Cb = acquireCmdBuffer();
  VkCommandBufferBeginInfo BI{};
  BI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  BI.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  checkVk(vkBeginCommandBuffer(Cb, &BI),
          "CHIPQueueVulkan::enqueueMarkerImpl: vkBeginCommandBuffer failed",
          hipErrorTbd);
  checkVk(vkEndCommandBuffer(Cb),
          "CHIPQueueVulkan::enqueueMarkerImpl: vkEndCommandBuffer failed",
          hipErrorTbd);
  return submitWithEvent(Cb, {});
}

std::shared_ptr<chipstar::Event>
CHIPQueueVulkan::memPrefetchImpl(const void * /*Ptr*/, size_t /*Count*/,
                                 int /*DstDevId*/) {
  // Vulkan has no prefetch.
  return enqueueMarkerImpl();
}

hipError_t CHIPQueueVulkan::getBackendHandles(uintptr_t *NativeHandles,
                                              int *NumHandles) {
  // Backend name, VkInstance, VkPhysicalDevice, VkDevice, VkQueue.
  constexpr int VulkanNumHandles = 5;
  if (NumHandles) {
    *NumHandles = VulkanNumHandles;
    return hipSuccess;
  }

  if (NativeHandles == nullptr)
    return hipErrorInvalidValue;

  NativeHandles[0] = reinterpret_cast<uintptr_t>("vulkan");
  auto *BVk = static_cast<CHIPBackendVulkan *>(Backend);
  NativeHandles[1] = reinterpret_cast<uintptr_t>(BVk->getInstance());
  NativeHandles[2] =
      reinterpret_cast<uintptr_t>(ChipDevice_->getPhysicalDevice());
  NativeHandles[3] =
      reinterpret_cast<uintptr_t>(ChipDevice_->getLogicalDevice());
  NativeHandles[4] =
      reinterpret_cast<uintptr_t>(ChipDevice_->getComputeQueue());
  return hipSuccess;
}

VkCommandBuffer CHIPQueueVulkan::acquireCmdBuffer() {
  VkCommandBuffer Cb;
  uint64_t Pending;
  {
    std::lock_guard<std::mutex> Lock(QueueOpMtx_);
    Cb = CmdBufferRing_[RingHead_];
    Pending = RingSlotValue_[RingHead_];
    RingHead_ = (RingHead_ + 1) % RingCapacity_;
  }
  // Resetting a command buffer that is still pending drops its commands.
  if (Pending > 0 && TimelineSemaphore_ != VK_NULL_HANDLE) {
    VkSemaphoreWaitInfo WI{};
    WI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    WI.semaphoreCount = 1;
    WI.pSemaphores = &TimelineSemaphore_;
    WI.pValues = &Pending;
    (void)vkWaitSemaphores(ChipDevice_->getLogicalDevice(), &WI, UINT64_MAX);
  }
  if (Cb != VK_NULL_HANDLE) {
    (void)vkResetCommandBuffer(Cb, /*flags=*/0);
  }
  return Cb;
}

void CHIPQueueVulkan::noteRingSubmit(VkCommandBuffer Cb, uint64_t Val) {
  for (uint32_t I = 0; I < RingCapacity_; ++I)
    if (CmdBufferRing_[I] == Cb)
      RingSlotValue_[I] = Val;
}

std::shared_ptr<chipstar::Event> CHIPQueueVulkan::submitWithEvent(
    VkCommandBuffer Buf,
    const std::vector<std::shared_ptr<chipstar::Event>> &EventsToWaitFor) {
  std::shared_ptr<chipstar::Event> SignalEvent = Backend->createEventShared(
      ChipContext_, chipstar::EventFlags(), "submitWithEvent");
  auto *SignalVk = static_cast<CHIPEventVulkan *>(SignalEvent.get());

  // A VkFence can only be waited on by the host.
  for (auto &E : EventsToWaitFor) {
    auto *Ev = static_cast<CHIPEventVulkan *>(E.get());
    if (Ev && Ev->getFence() != VK_NULL_HANDLE) {
      VkFence Wf = Ev->getFence();
      (void)vkWaitForFences(ChipDevice_->getLogicalDevice(), 1, &Wf,
                            VK_TRUE, UINT64_MAX);
    }
  }

  uint64_t WaitTimelineVal;
  uint64_t SignalTimelineVal;
  {
    std::lock_guard<std::mutex> Lock(QueueOpMtx_);
    WaitTimelineVal = TimelineValue_;
    SignalTimelineVal = ++TimelineValue_;
    noteRingSubmit(Buf, SignalTimelineVal);
  }

  // Waiting on the previous submit's value also makes its writes visible.
  std::vector<VkSemaphore> WaitSems;
  std::vector<uint64_t> WaitVals;
  if (WaitTimelineVal > 0) {
    WaitSems.push_back(TimelineSemaphore_);
    WaitVals.push_back(WaitTimelineVal);
  }
  // Legacy default stream semantics: the default stream waits for every
  // blocking stream, and a blocking stream waits for the default stream.
  // Draining runs inside finish(), whose callers may hold
  // QueueAddRemoveMtx; its copies need no cross-stream ordering.
  if (getQueueFlags().isBlocking() && !Draining_) {
    std::lock_guard<std::mutex> QLock(ChipDevice_->QueueAddRemoveMtx);
    std::vector<chipstar::Queue *> Others;
    if (isDefaultLegacyQueue()) {
      for (auto *Q : ChipDevice_->getQueuesNoLock())
        if (Q != this && Q->getQueueFlags().isBlocking())
          Others.push_back(Q);
      // The calling thread's per-thread default stream, which is not listed.
      auto *PT = chipstar::Device::PerThreadDefaultQueue.get();
      if (PT && PT->getDevice() == ChipDevice_)
        Others.push_back(PT);
    } else if (auto *Def = ChipDevice_->getLegacyDefaultQueue()) {
      if (Def != this)
        Others.push_back(Def);
    }
    for (auto *Q : Others) {
      auto *QV = static_cast<CHIPQueueVulkan *>(Q);
      uint64_t V;
      {
        std::lock_guard<std::mutex> Lock(QV->QueueOpMtx_);
        V = QV->TimelineValue_;
      }
      if (V > 0 && QV->TimelineSemaphore_ != VK_NULL_HANDLE) {
        WaitSems.push_back(QV->TimelineSemaphore_);
        WaitVals.push_back(V);
      }
    }
  }
  std::vector<VkPipelineStageFlags> WaitStages(
      WaitSems.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

  VkTimelineSemaphoreSubmitInfo TsSubmit{};
  TsSubmit.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  TsSubmit.waitSemaphoreValueCount = static_cast<uint32_t>(WaitVals.size());
  TsSubmit.pWaitSemaphoreValues = WaitVals.data();
  TsSubmit.signalSemaphoreValueCount = 1;
  TsSubmit.pSignalSemaphoreValues = &SignalTimelineVal;

  VkSubmitInfo Submit{};
  Submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  Submit.pNext = &TsSubmit;
  Submit.waitSemaphoreCount = static_cast<uint32_t>(WaitSems.size());
  Submit.pWaitSemaphores = WaitSems.data();
  Submit.pWaitDstStageMask = WaitStages.data();
  Submit.commandBufferCount = (Buf != VK_NULL_HANDLE) ? 1u : 0u;
  Submit.pCommandBuffers = (Buf != VK_NULL_HANDLE) ? &Buf : nullptr;
  Submit.signalSemaphoreCount = 1;
  Submit.pSignalSemaphores = &TimelineSemaphore_;

  VkFence SignalFence = SignalVk ? SignalVk->getFence() : VK_NULL_HANDLE;
  if (SignalFence == VK_NULL_HANDLE) {
    SignalFence = FinishFence_;
    (void)vkResetFences(ChipDevice_->getLogicalDevice(), 1, &SignalFence);
  }

  {
    // vkQueueSubmit requires external synchronization of the VkQueue.
    std::lock_guard<std::mutex> SubmitLock(ChipDevice_->getSubmitMtx());
    checkVk(vkQueueSubmit(ChipDevice_->getComputeQueue(), 1, &Submit,
                          SignalFence),
            "CHIPQueueVulkan::submitWithEvent: vkQueueSubmit failed",
            hipErrorTbd);
  }

  IsEmptyQueue_.store(false);
  if (SignalVk)
    SignalVk->setRecording();
  // The event monitor frees it once its fence signals.
  {
    LOCK(Backend->EventsMtx);
    Backend->trackEvent(SignalEvent);
  }
  return SignalEvent;
}


// ============================================================================
// CHIPBackendVulkan
// ============================================================================

std::atomic<bool> CHIPBackendVulkan::ShuttingDown_{false};
std::recursive_mutex CHIPBackendVulkan::TeardownMtx_;
CHIPBackendVulkan::CHIPBackendVulkan() = default;

CHIPBackendVulkan::~CHIPBackendVulkan() {
  // In case uninitialize() was not called.
  if (DebugMessenger_ != VK_NULL_HANDLE && Instance_ != VK_NULL_HANDLE) {
    auto Destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(Instance_, "vkDestroyDebugUtilsMessengerEXT"));
    if (Destroy)
      Destroy(Instance_, DebugMessenger_, nullptr);
    DebugMessenger_ = VK_NULL_HANDLE;
  }
  if (Instance_ != VK_NULL_HANDLE) {
    vkDestroyInstance(Instance_, nullptr);
    Instance_ = VK_NULL_HANDLE;
  }
}

chipstar::ExecItem *CHIPBackendVulkan::createExecItem(dim3 GridDim,
                                                      dim3 BlockDim,
                                                      size_t SharedMem,
                                                      hipStream_t ChipQueue) {
  return new CHIPExecItemVulkan(GridDim, BlockDim, SharedMem, ChipQueue);
}

std::string CHIPBackendVulkan::getDefaultJitFlags() { return std::string(); }

int CHIPBackendVulkan::ReqNumHandles() { return 4; }

void CHIPBackendVulkan::initializeImpl() {
  logTrace("CHIPBackendVulkan Initialize");

  VkApplicationInfo App{};
  App.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  App.pApplicationName = "chipStar";
  App.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
  App.pEngineName = "chipStar";
  App.engineVersion = VK_MAKE_VERSION(0, 1, 0);
  App.apiVersion = VK_API_VERSION_1_3;

  ValidationEnabled_ = false;
  DebugUtilsEnabled_ = false;

  std::vector<const char *> WantLayers;
  std::vector<const char *> WantExts;

  if (chipVkValidationRequested()) {
    // Without the layer installed, run unvalidated rather than fail.
    uint32_t LayerCount = 0;
    vkEnumerateInstanceLayerProperties(&LayerCount, nullptr);
    std::vector<VkLayerProperties> Layers(LayerCount);
    vkEnumerateInstanceLayerProperties(&LayerCount, Layers.data());
    if (vkPropertyListContains(Layers, &VkLayerProperties::layerName,
                                "VK_LAYER_KHRONOS_validation")) {
      WantLayers.push_back("VK_LAYER_KHRONOS_validation");
      ValidationEnabled_ = true;
    } else {
      logWarn("CHIP_VK_VALIDATION requested but VK_LAYER_KHRONOS_validation "
              "is not installed; running without validation.");
    }

    uint32_t ExtCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &ExtCount, nullptr);
    std::vector<VkExtensionProperties> Exts(ExtCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &ExtCount, Exts.data());
    if (vkPropertyListContains(Exts, &VkExtensionProperties::extensionName,
                               VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
      WantExts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
      DebugUtilsEnabled_ = true;
    } else if (ValidationEnabled_) {
      logWarn("VK_EXT_debug_utils not present; validation layer messages "
              "will not be routed through the chipStar logger.");
    }
  }

  // In pNext, it also reports vkCreateInstance and vkDestroyInstance.
  VkDebugUtilsMessengerCreateInfoEXT BootMsgInfo{};
  if (DebugUtilsEnabled_) {
    BootMsgInfo.sType =
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    BootMsgInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    BootMsgInfo.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    BootMsgInfo.pfnUserCallback = chipVkDebugCallback;
  }

  VkInstanceCreateInfo InstInfo{};
  InstInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  InstInfo.pApplicationInfo = &App;
  InstInfo.enabledLayerCount = static_cast<uint32_t>(WantLayers.size());
  InstInfo.ppEnabledLayerNames =
      WantLayers.empty() ? nullptr : WantLayers.data();
  InstInfo.enabledExtensionCount = static_cast<uint32_t>(WantExts.size());
  InstInfo.ppEnabledExtensionNames =
      WantExts.empty() ? nullptr : WantExts.data();
  if (DebugUtilsEnabled_)
    InstInfo.pNext = &BootMsgInfo;

  VkResult R = vkCreateInstance(&InstInfo, nullptr, &Instance_);
  if (R != VK_SUCCESS) {
    CHIPERR_LOG_AND_THROW(std::string("vkCreateInstance failed: ") +
                              std::to_string(static_cast<int>(R)),
                          hipErrorInitializationError);
  }
  if (DebugUtilsEnabled_) {
    auto Create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(Instance_, "vkCreateDebugUtilsMessengerEXT"));
    if (Create) {
      R = Create(Instance_, &BootMsgInfo, nullptr, &DebugMessenger_);
      if (R != VK_SUCCESS) {
        logWarn("vkCreateDebugUtilsMessengerEXT failed: {}; continuing "
                "without persistent debug messenger.",
                static_cast<int>(R));
        DebugMessenger_ = VK_NULL_HANDLE;
      }
    }
  }

  uint32_t PhysCount = 0;
  R = vkEnumeratePhysicalDevices(Instance_, &PhysCount, nullptr);
  if (R != VK_SUCCESS || PhysCount == 0) {
    CHIPERR_LOG_AND_THROW("No Vulkan physical devices found",
                          hipErrorInitializationError);
  }
  std::vector<VkPhysicalDevice> Phys(PhysCount);
  vkEnumeratePhysicalDevices(Instance_, &PhysCount, Phys.data());

  // One context with one device.
  CHIPContextVulkan *Ctx = new CHIPContextVulkan();
  addContext(Ctx);

  int Accepted = 0;
  for (uint32_t i = 0; i < PhysCount; ++i) {
    VkPhysicalDeviceProperties Props{};
    vkGetPhysicalDeviceProperties(Phys[i], &Props);

    if (Props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
        Props.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU &&
        Props.deviceType != VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) {
      logTrace("Skipping non-GPU Vulkan device '{}'", Props.deviceName);
      continue;
    }
    if (Props.apiVersion < VK_API_VERSION_1_3) {
      logTrace("Skipping Vulkan device '{}' with API < 1.3",
               Props.deviceName);
      continue;
    }

    VkPhysicalDeviceShaderFloat16Int8Features F16I8{};
    F16I8.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    VkPhysicalDeviceVulkan12Features V12{};
    V12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    V12.pNext = &F16I8;
    VkPhysicalDeviceFeatures2 Feat2{};
    Feat2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    Feat2.pNext = &V12;
    vkGetPhysicalDeviceFeatures2(Phys[i], &Feat2);
    if (V12.shaderInt8 != VK_TRUE && F16I8.shaderInt8 != VK_TRUE) {
      logTrace("Skipping Vulkan device '{}' (no shaderInt8)",
               Props.deviceName);
      continue;
    }
    // HIP device pointers are buffer device addresses.
    if (V12.bufferDeviceAddress != VK_TRUE) {
      logTrace("Skipping Vulkan device '{}' (no bufferDeviceAddress)",
               Props.deviceName);
      continue;
    }

    // Kernels use subgroup operations.
    VkPhysicalDeviceSubgroupProperties SgProps{};
    SgProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 Props2{};
    Props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    Props2.pNext = &SgProps;
    vkGetPhysicalDeviceProperties2(Phys[i], &Props2);
    if (!(SgProps.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT)) {
      logTrace(
          "Skipping Vulkan device '{}' (no subgroup ops in compute stage)",
          Props.deviceName);
      continue;
    }

    logInfo("Vulkan device {}: '{}' (apiVersion {}.{}.{})", Accepted,
            Props.deviceName, VK_API_VERSION_MAJOR(Props.apiVersion),
            VK_API_VERSION_MINOR(Props.apiVersion),
            VK_API_VERSION_PATCH(Props.apiVersion));

    CHIPDeviceVulkan *Dev = CHIPDeviceVulkan::create(Ctx, Phys[i], Accepted);
    (void)Dev; // create() registers it with the context.
    ++Accepted;
    // Only the first suitable device is used.
    break;
  }
  if (Accepted == 0) {
    CHIPERR_LOG_AND_THROW(
        "No Vulkan physical device meets the chipStar requirements "
        "(GPU, Vulkan 1.3, shaderInt8, bufferDeviceAddress, compute-stage "
        "subgroup ops)",
        hipErrorInitializationError);
  }

  EventMonitor_ = ::Backend->createEventMonitor_();
}

void CHIPBackendVulkan::initializeFromNative(const uintptr_t * /*NH*/,
                                              int /*NumHandles*/) {
  CHIPERR_LOG_AND_THROW(
      "CHIPBackendVulkan::initializeFromNative not supported",
      hipErrorNotSupported);
}

void CHIPBackendVulkan::uninitialize() {
  logTrace("CHIPBackendVulkan::uninitialize");

  // Detached threads may still be in HIP calls using the VkDevice.
  waitForThreadExit();

  // Flag before locking, so a queue destructor waiting on the lock sees it.
  ShuttingDown_.store(true, std::memory_order_release);
  std::lock_guard<std::recursive_mutex> TeardownLock(TeardownMtx_);
  if (EventMonitor_) {
    {
      LOCK(EventMonitor_->EventMonitorMtx);
      EventMonitor_->Stop = true;
    }
    EventMonitor_->join();
    EventMonitor_ = nullptr;
  }

  // Devices go before the VkInstance; ~Backend deletes the contexts.
  for (auto *Ctx : ChipContexts) {
    if (auto *Dev = Ctx ? Ctx->getDevice() : nullptr) {
      delete Dev;
      Ctx->setDevice(nullptr);
    }
  }

  if (DebugMessenger_ != VK_NULL_HANDLE && Instance_ != VK_NULL_HANDLE) {
    auto Destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(Instance_, "vkDestroyDebugUtilsMessengerEXT"));
    if (Destroy)
      Destroy(Instance_, DebugMessenger_, nullptr);
    DebugMessenger_ = VK_NULL_HANDLE;
  }
  if (Instance_ != VK_NULL_HANDLE) {
    vkDestroyInstance(Instance_, nullptr);
    Instance_ = VK_NULL_HANDLE;
  }
}

chipstar::Queue *
CHIPBackendVulkan::createCHIPQueue(chipstar::Device *ChipDev) {
  return new CHIPQueueVulkan(ChipDev, chipstar::QueueFlags(), /*Priority=*/0);
}

std::shared_ptr<chipstar::Event>
CHIPBackendVulkan::createEventShared(chipstar::Context *ChipCtx,
                                     chipstar::EventFlags Flags,
                                     std::string Msg) {
  auto Event = std::make_shared<CHIPEventVulkan>(ChipCtx, Flags);
  if (!Msg.empty())
    Event->Msg = std::move(Msg);
  return Event;
}

chipstar::Event *
CHIPBackendVulkan::createEvent(chipstar::Context *ChipCtx,
                               chipstar::EventFlags Flags) {
  auto *Event = new CHIPEventVulkan(ChipCtx, Flags);
  Event->setUserEvent(true);
  return Event;
}

chipstar::CallbackData *
CHIPBackendVulkan::createCallbackData(hipStreamCallback_t Callback,
                                      void *UserData,
                                      chipstar::Queue *ChipQ) {
  if (Callback == nullptr || ChipQ == nullptr)
    return nullptr;
  return new CHIPCallbackDataVulkan(Callback, UserData, ChipQ);
}

chipstar::EventMonitor *CHIPBackendVulkan::createEventMonitor_() {
  auto *Monitor = new EventMonitorVulkan();
  Monitor->start();
  return Monitor;
}

hipEvent_t CHIPBackendVulkan::getHipEvent(void * /*NativeEvent*/) {
  CHIPERR_LOG_AND_THROW(
      "CHIPBackendVulkan::getHipEvent: native event interop not implemented",
      hipErrorNotSupported);
}

void *CHIPBackendVulkan::getNativeEvent(hipEvent_t /*HipEvent*/) {
  CHIPERR_LOG_AND_THROW(
      "CHIPBackendVulkan::getNativeEvent: native event interop not implemented",
      hipErrorNotSupported);
}

