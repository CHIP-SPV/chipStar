/*
 * Copyright (c) 2021-23 CHIP-SPV developers
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
 * @brief Backend class definition. CHIP backends are to inherit from this
 * base class and override desired virtual functions. Overrides for this class
 * are expected to be minimal with primary overrides being done on lower-level
 * classes such as Context consturctors, etc.
 * @version 0.1
 * @date 2021-08-19
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef CHIP_BACKEND_H
#define CHIP_BACKEND_H

#include "spirv.hh"
#include "common.hh"
#include "hip/hip_runtime_api.h"
#include "hip/spirv_hip.hh"

#include "CHIPDriver.hh"
#include "logging.hh"
#include "macros.hh"
#include "CHIPException.hh"

#include "SPVRegister.hh"

#define DEFAULT_QUEUE_PRIORITY 1

inline std::string hipMemcpyKindToString(hipMemcpyKind Kind) {
  switch (Kind) {
  case hipMemcpyHostToHost:
    return "hipMemcpyHostToHost";
  case hipMemcpyHostToDevice:
    return "hipMemcpyHostToDevice";
  case hipMemcpyDeviceToHost:
    return "hipMemcpyDeviceToHost";
  case hipMemcpyDeviceToDevice:
    return "hipMemcpyDeviceToDevice";
  case hipMemcpyDefault:
    return "hipMemcpyDefault";
  default:
    return "hipMemcpyUnknown";
  }
}

static inline size_t getChannelByteSize(hipChannelFormatDesc Desc) {
  unsigned TotalNumBits = Desc.x + Desc.y + Desc.z + Desc.w;
  return ((TotalNumBits + 7u) / 8u); // Round upwards.
}

template <class T> std::string resultToString(T Err);

// class CHIPGraph;
// class CHIPGraphNode;
#include "CHIPGraph.hh"

/// Describes a memory region to copy from/to.
namespace chipstar {

class Queue;
class Backend;
class Event;
class Kernel;
class Device;
class EventMonitor;
class Texture;
class Context;

class RegionDesc {
public:
  static constexpr unsigned MaxNumDims = 3;
  // The number of dimensions.
  unsigned char NumDims = 1;
  // The size of element measured in bytes.
  size_t ElementSize = 0;
  // Measured in elements. The minimum size allowed is one for the first
  // NumDim elements.
  size_t Size[MaxNumDims] = {1, 1, 1};
  // Measured in elements.
  size_t Offset[MaxNumDims] = {0, 0, 0};
  // Row and slice pitch. Measured in bytes.
  size_t Pitch[MaxNumDims - 1] = {1, 1};

  std::string dumpAsString() const {
    std::string Result =
        "Size=(" + std::to_string(Size[0]) + ", " + std::to_string(Size[1]) +
        ", " + std::to_string(Size[2]) + "), Offset=(" +
        std::to_string(Offset[0]) + ", " + std::to_string(Offset[1]) + ", " +
        std::to_string(Offset[2]) + "), Pitch=(" + std::to_string(Pitch[0]) +
        "," + std::to_string(Pitch[1]) + ")";
    return Result;
  }

  unsigned getNumDims() const {
    CHIPASSERT(NumDims > 0 && NumDims <= MaxNumDims);
    return NumDims;
  }

  bool isPitched() const {
    switch (getNumDims()) {
    default:
      CHIPASSERT(false && "Unexpected dimension count.");
      return false;
    case 1:
      return false;
    case 2:
      return Pitch[0] > Size[0] * ElementSize;
    case 3:
      return Pitch[0] > Size[0] * ElementSize &&
             Pitch[1] > Size[1] * Size[0] * ElementSize;
    }
  }

  static chipstar::RegionDesc get3DRegion(size_t TheWidth, size_t TheHeight,
                                          size_t TheDepth,
                                          size_t ElementByteSize = 1) {
    chipstar::RegionDesc Result;
    Result.NumDims = 3;
    Result.ElementSize = ElementByteSize;
    Result.Size[0] = TheWidth;
    Result.Size[1] = TheHeight;
    Result.Size[2] = TheDepth;
    Result.Pitch[0] = TheWidth * ElementByteSize;
    Result.Pitch[1] = TheWidth * TheHeight * ElementByteSize;
    return Result;
  }

  static chipstar::RegionDesc get2DRegion(size_t TheWidth, size_t TheHeight,
                                          size_t ElementByteSize = 1) {
    auto R = get3DRegion(TheWidth, TheHeight, 1, ElementByteSize);
    R.NumDims = 2;
    return R;
  }

  static chipstar::RegionDesc get1DRegion(size_t TheWidth, size_t TheHeight,
                                          size_t ElementByteSize = 1) {
    auto R = get2DRegion(TheWidth, 1, ElementByteSize);
    R.NumDims = 1;
    return R;
  }

  static chipstar::RegionDesc from(const hipArray &Array) {
    auto TexelByteSize = getChannelByteSize(Array.desc);
    switch (Array.textureType) {
    default:
      assert(false && "Unkown texture type.");
      return chipstar::RegionDesc();
    case hipTextureType1D:
      return chipstar::RegionDesc::get1DRegion(Array.width, TexelByteSize);
    case hipTextureType2D:
      return chipstar::RegionDesc::get2DRegion(Array.width, Array.height,
                                               TexelByteSize);
    case hipTextureType3D:
      return chipstar::RegionDesc::get3DRegion(Array.width, Array.height,
                                               Array.depth, TexelByteSize);
    }
  }

  static chipstar::RegionDesc from(const hipResourceDesc &ResDesc) {
    switch (ResDesc.resType) {
    default:
      CHIPASSERT(false && "Unknown resource type");
      return chipstar::RegionDesc();
    case hipResourceTypePitch2D: {
      auto &Res = ResDesc.res.pitch2D;
      auto R = get2DRegion(Res.width, Res.height, getChannelByteSize(Res.desc));
      R.Pitch[0] = Res.pitchInBytes;
      CHIPASSERT(Res.pitchInBytes >= Res.width * getChannelByteSize(Res.desc) &&
                 "Invalid pitch.");
      return R;
    }
    case hipResourceTypeArray: {
      const auto *Array = ResDesc.res.array.array;
      assert(Array);
      return from(*Array);
    }
    } // switch
  }
};

class QueueFlags {
  unsigned int FlagsRaw_;
  bool Default_ = true;
  bool NonBlocking_ = false;

public:
  QueueFlags() : QueueFlags(hipStreamDefault) {}
  QueueFlags(unsigned int FlagsRaw) : FlagsRaw_(FlagsRaw) {

    if (FlagsRaw & hipStreamDefault) {
      Default_ = true;
      FlagsRaw = FlagsRaw & (~hipStreamDefault);
    }

    if (FlagsRaw & hipStreamNonBlocking) {
      NonBlocking_ = true;
      FlagsRaw = FlagsRaw & (~hipStreamNonBlocking);
    }

    if (FlagsRaw > 0)
      CHIPERR_LOG_AND_THROW("Invalid QueueFlags", hipErrorInvalidValue);
  }

  bool isDefault() { return Default_; }
  bool isNonBlocking() { return NonBlocking_; }
  bool isBlocking() { return !NonBlocking_; }
  unsigned int getRaw() { return FlagsRaw_; }
};

enum class ManagedMemFlags : unsigned int {
  AttachHost = hipMemAttachHost,
  AttachGlobal = hipMemAttachGlobal
};

class HostAllocFlags {
  bool Default_ = true;
  bool Portable_ = false;
  bool Mapped_ = false;
  bool WriteCombined_ = false;
  bool NumaUser_ = false;
  bool Coherent_ = false;
  bool NonCoherent_ = false;
  unsigned int FlagsRaw_;

public:
  HostAllocFlags() : FlagsRaw_(hipHostMallocDefault){};
  HostAllocFlags(unsigned int FlagsRaw) : FlagsRaw_(FlagsRaw) {
    if (FlagsRaw & hipHostMallocDefault) {
      Default_ = true;
      FlagsRaw = FlagsRaw & (~hipHostMallocDefault);
    }

    if (FlagsRaw & hipHostMallocPortable) {
      Portable_ = true;
      Default_ = false;
      FlagsRaw = FlagsRaw & (~hipHostMallocPortable);
    }

    if (FlagsRaw & hipHostMallocMapped) {
      Mapped_ = true;
      Default_ = false;
      FlagsRaw = FlagsRaw & (~hipHostMallocMapped);
    }

    if (FlagsRaw & hipHostMallocWriteCombined) {
      WriteCombined_ = true;
      Default_ = false;
      FlagsRaw = FlagsRaw & (~hipHostMallocWriteCombined);
    }

    if (FlagsRaw & hipHostMallocNumaUser) {
      NumaUser_ = true;
      Default_ = false;
      FlagsRaw = FlagsRaw & (~hipHostMallocNumaUser);
    }

    if (FlagsRaw & hipHostMallocCoherent) {
      Coherent_ = true;
      Default_ = false;
      FlagsRaw = FlagsRaw & (~hipHostMallocCoherent);
    }

    if (FlagsRaw & hipHostMallocNonCoherent) {
      NonCoherent_ = true;
      Default_ = false;
      FlagsRaw = FlagsRaw & (~hipHostMallocNonCoherent);
    }

    if (FlagsRaw > 0)
      CHIPERR_LOG_AND_THROW("Invalid CHIPHostAllocFlag", hipErrorInvalidValue);

    if (Coherent_ && NonCoherent_)
      CHIPERR_LOG_AND_THROW("Invalid CHIPHostAllocFlag", hipErrorInvalidValue);
  }
  unsigned int getRaw() { return FlagsRaw_; }
  bool isDefault() { return Default_; }
  bool isPortable() { return Portable_; }
  bool isMapped() { return Mapped_; }
  bool isWriteCombined() { return WriteCombined_; }
  bool isNumaUser() { return NumaUser_; }
  bool isCoherent() { return Coherent_; }
  bool isNonCoherent() { return NonCoherent_; }
};

/**
 * @brief This object gets created when a callback is requested. Once created,
 * it gets placed on the Backend callback queue. A Callback monitor thread
 * gets created and executes these callback objects. This object stores all the
 * necessary data to execute a callback function:
 * - Events for synching
 * - Callback function
 * - Arguments for the callback function
 */
class CallbackData {
protected:
  virtual ~CallbackData() = default;

public:
  chipstar::Queue *ChipQueue;
  std::shared_ptr<chipstar::Event> GpuReady;
  std::shared_ptr<chipstar::Event> CpuCallbackComplete;
  std::shared_ptr<chipstar::Event> GpuAck;

  hipError_t Status;
  void *CallbackArgs;
  hipStreamCallback_t CallbackF;

  CallbackData(hipStreamCallback_t CallbackF, void *CallbackArgs,
               chipstar::Queue *ChipQueue);

  void execute(hipError_t ResultFromDependency);
};

class EventMonitor {
  typedef void *(*THREADFUNCPTR)(void *);

protected:
  EventMonitor() = default;
  virtual ~EventMonitor() = default;
  pthread_t Thread_;

public:
  std::mutex EventMonitorMtx;
  volatile bool Stop = false;

  void join() {
    assert(Thread_);
    logDebug("Joining chipstar::Event Monitor Thread {}", Thread_);
    int Status = pthread_join(Thread_, nullptr);
    if (Status != 0) {
      logError("Failed to call join() {}", Status);
    }
  }

  static void *monitorWrapper(void *Arg) {
    auto Monitor = (chipstar::EventMonitor *)Arg;
    Monitor->monitor();
    return 0;
  }
  virtual void monitor(){};

  void start() {
    logDebug("Starting chipstar::Event Monitor Thread");
    auto Res = pthread_create(&Thread_, 0, monitorWrapper, (void *)this);
    if (Res)
      CHIPERR_LOG_AND_THROW("Failed to create thread", hipErrorTbd);
    logDebug("Thread Created with ID : {}", Thread_);
  }

  void stop() {
    LOCK(EventMonitorMtx) // Lock the mutex to ensure that the thread is not
                          // executing the monitor function
    logDebug("Stopping chipstar::Event Monitor Thread");
    Stop = true;
    join();
  }
};

class Texture {
  /// Resource description used to create this texture.
  hipResourceDesc ResourceDesc;

public:
  Texture() = delete;
  Texture(const hipResourceDesc &ResDesc) : ResourceDesc(ResDesc) {}
  virtual ~Texture() {}

  const hipResourceDesc &getResourceDesc() const { return ResourceDesc; }
};

class EventFlags {
  bool BlockingSync_ = false;
  bool DisableTiming_ = false;
  bool Interprocess_ = false;
  bool ReleaseToDevice_ = false;
  bool ReleaseToSystem_ = false;

public:
  EventFlags() = default;
  EventFlags(unsigned Flags) {

    if (Flags & hipEventBlockingSync) {
      Flags = Flags & (~hipEventBlockingSync);
      BlockingSync_ = true;
    }
    if (Flags & hipEventDisableTiming) {
      Flags = Flags & (~hipEventDisableTiming);
      DisableTiming_ = true;
    }
    if (Flags & hipEventInterprocess) {
      Flags = Flags & (~hipEventInterprocess);
      logWarn("hipEventInterprocess is not supported on CHIP-SPV");
      Interprocess_ = true;
    }
    if (Flags & hipEventReleaseToDevice) {
      Flags = Flags & (~hipEventReleaseToDevice);
      logWarn("hipEventReleaseToDevice is not supported on CHIP-SPV");
      ReleaseToDevice_ = true;
    }
    if (Flags & hipEventReleaseToSystem) {
      Flags = Flags & (~hipEventReleaseToSystem);
      logWarn("hipEventReleaseToSystem is not supported on CHIP-SPV");
      ReleaseToSystem_ = true;
    }

    if (Interprocess_ && !DisableTiming_) {
      CHIPERR_LOG_AND_THROW(
          "hipEventInterprocess requires hipEventDisableTiming",
          hipErrorInvalidValue);
    }
    if (Flags > 0)
      CHIPERR_LOG_AND_THROW("Invalid hipEvent flag combination",
                            hipErrorInvalidValue);
  }

  bool isDefault() {
    return !BlockingSync_ && !DisableTiming_ && !Interprocess_;
  };
  bool isBlockingSync() { return BlockingSync_; };
  bool isDisableTiming() { return DisableTiming_; };
  bool isInterprocess() { return Interprocess_; };
};

/**
 * @brief  Structure describing an allocation
 *
 */
struct AllocationInfo {
  // TODO make this into a class
  void *DevPtr;
  void *HostPtr;
  size_t Size;
  chipstar::HostAllocFlags Flags;
  hipDevice_t Device;
  bool Managed = false;
  enum hipMemoryType MemoryType;
  bool RequiresMapUnmap = false;
  bool IsHostRegistered = false; ///< True if registered via hipHostRegister().
};

/**
 * @brief Class for keeping track of device allocations.
 *
 */
class AllocationTracker {
private:
  std::string Name_;

  std::unordered_set<chipstar::AllocationInfo *> AllocInfos_;
  std::unordered_map<void *, chipstar::AllocationInfo *> PtrToAllocInfo_;

public:
  mutable std::mutex AllocationTrackerMtx;

  /**
   * @brief Associate a host pointer with a device pointer. @see hipHostRegister
   *
   * @param HostPtr
   */
  void registerHostPointer(void *HostPtr, void *DevPtr) {
    CHIPASSERT(HostPtr && "HostPtr is null");
    CHIPASSERT(DevPtr && "DevPtr is null");
    auto AllocInfo = this->getAllocInfo(DevPtr);
    if (AllocInfo->HostPtr)
      // HIP test suite expects hipErrorInvalidValue, HIP API does not
      // meantion it. If we followed CUDA Runtime API, we'd return
      // hipErrorHostMemoryAlreadyRegistered.
      CHIPERR_LOG_AND_THROW("Host memory is already registered!",
                            hipErrorInvalidValue);
    AllocInfo->HostPtr = HostPtr;
    this->PtrToAllocInfo_[HostPtr] = AllocInfo;
    AllocInfo->MemoryType = hipMemoryTypeManaged;
    AllocInfo->IsHostRegistered = true;
  }

  size_t GlobalMemSize, TotalMemSize, MaxMemUsed;
  /**
   * @brief Construct a new chipstar::AllocationTracker object
   *
   * @param GlobalMemSize Total available global memory on the device
   * @param Name name for this allocation tracker for logging. Normally device
   * name
   */
  AllocationTracker(size_t GlobalMemSize, std::string Name);

  /**
   * @brief Destroy the AllocationTracker object
   *
   */
  ~AllocationTracker();

  /**
   * @brief Get the Name object
   *
   * @return std::string
   */
  std::string getName();

  /**
   * @brief Get Allocation Info associated with this pointer
   *
   * @return chipstar::AllocationInfo contains the base pointer and allocation
   * size;
   */
  chipstar::AllocationInfo *getAllocInfo(const void *);

  /**
   * @brief Reserve memory for an allocation.
   * This method is run prior to allocations to keep track of how much memory is
   * available on the device
   *
   * @param bytes
   * @return true Reservation successful
   * @return false Not enough available memory for reservation of this size.
   */
  bool reserveMem(size_t Bytes);

  /**
   * @brief Release some of the reserved memory. Called by free()
   *
   * @param bytes
   * @return true
   * @return false
   */
  bool releaseMemReservation(size_t Bytes);

  /**
   * @brief Record the pointer received from Context::allocate_()
   *
   * @param dev_ptr
   */
  void recordAllocation(void *DevPtr, void *HostPtr, hipDevice_t Device,
                        size_t Size, chipstar::HostAllocFlags Flags,
                        hipMemoryType MemoryType);

  /**
   * @brief Check if a given pointer belongs to any of the existing allocations
   *
   * @param DevPtr device side pointer
   * @return chipstar::AllocationInfo* pointer to allocation info. Nullptr if
   * this pointer does not belong to any existing allocations
   */
  chipstar::AllocationInfo *getAllocInfoCheckPtrRanges(void *DevPtr);

  /**
   * @brief Delete an chipstar::AllocationInfo item
   *
   * @param AllocInfo
   */
  void eraseRecord(chipstar::AllocationInfo *AllocInfo) {
    LOCK(AllocationTrackerMtx); // CHIPAllocationTracker::PtrToAllocInfo_
                                // CHIPAllocationTracker::AllocInfos_
    assert(AllocInfo && "Null pointer passed to eraseRecord");
    assert(AllocInfos_.count(AllocInfo) &&
           "Not a member of the allocation tracker!");
    PtrToAllocInfo_.erase(AllocInfo->DevPtr);
    if (AllocInfo->HostPtr)
      PtrToAllocInfo_.erase(AllocInfo->HostPtr);
    AllocInfos_.erase(AllocInfo);
    delete AllocInfo;
  }

  /**
   * @brief Visit tracked allocations.
   *
   * The visitor is called with 'const chipstar::AllocationInfo&' argument.
   */
  template <typename VisitorT> void visitAllocations(VisitorT Visitor) const {
    LOCK(AllocationTrackerMtx); // chipstar::AllocationTracker::AllocInfos_
    for (const auto *Info : AllocInfos_)
      Visitor(*Info);
  }

  size_t getNumAllocations() const { return AllocInfos_.size(); }
};

class DeviceVar {
private:
  const SPVVariable *SrcVar_ = nullptr;
  void *DevAddr_ = nullptr;
  /// The alignment requirement of the variable.
  // NOTE: The alignment infromation is not carried in __hipRegisterVar() calls
  // It have to be queried via shadow kernels.
  size_t Alignment_ = 0;
  /// Tells if the variable has an initializer. NOTE: Variables are
  /// initialized via a shadow kernel.
  bool HasInitializer_ = false;

public:
  DeviceVar(const SPVVariable *SrcVar) : SrcVar_(SrcVar) {}
  ~DeviceVar();

  void *getDevAddr() const { return DevAddr_; }
  void setDevAddr(void *Addr) { DevAddr_ = Addr; }
  std::string_view getName() const { return SrcVar_->Name; }
  size_t getSize() const { return SrcVar_->Size; }
  size_t getAlignment() const { return Alignment_; }
  void setAlignment(size_t TheAlignment) {
    assert(Alignment_ && "Invalid alignment");
    Alignment_ = TheAlignment;
  }
  bool hasInitializer() const { return HasInitializer_; }
  void markHasInitializer(bool State = true) { HasInitializer_ = State; }
};

class Event : public ihipEvent_t {
protected:
  bool TrackCalled_ = false;
  bool UserEvent_ = false;
  event_status_e EventStatus_;
  chipstar::EventFlags Flags_;
  std::vector<std::shared_ptr<chipstar::Event>> DependsOnList;

#ifndef NDEBUG
  // A debug flag for cathing use-after-delete.
  bool Deleted_ = false;
#endif

  /**
   * @brief Events are always created with a context
   *
   */
  chipstar::Context *ChipContext_;

  /**
   * @brief hidden default constructor for Event. Only derived class
   * constructor should be called.
   *
   */
  Event() : TrackCalled_(false), UserEvent_(false) {}
  virtual ~Event(){};

public:
  void markTracked() { TrackCalled_ = true; }
  bool isTrackCalled() { return TrackCalled_; }
  void setTrackCalled(bool Val) { TrackCalled_ = Val; }
  bool isUserEvent() { return UserEvent_; }
  void setUserEvent(bool Val) { UserEvent_ = Val; }
  void addDependency(const std::shared_ptr<chipstar::Event> &Event) {
    assert(!Deleted_ && "Event use after delete!");
    DependsOnList.push_back(Event);
  }
  void releaseDependencies();
  chipstar::EventFlags getFlags() { return Flags_; }
  std::mutex EventMtx;
  std::string Msg;
  // Optionally provide a field for origin of this event
  /**
   * @brief chipstar::Event constructor. Must always be created with some
   * context.
   *
   */
  Event(chipstar::Context *Ctx,
        chipstar::EventFlags Flags = chipstar::EventFlags());
  /**
   * @brief Get the Context object
   *
   * @return Context* pointer to context on which this event was created
   */
  chipstar::Context *getContext() {
    assert(!Deleted_ && "chipstar::Event use after delete!");
    return ChipContext_;
  }

  /**
   * @brief Query the state of this event and update it's status
   * Each backend must override this method with implementation specific calls
   * e.x. clGetEventInfo()
   *
   * @return true event was in recording state, state might have changed
   * @return false event was not in recording state
   */
  virtual bool updateFinishStatus(bool ThrowErrorIfNotReady = true) = 0;

  /**
   * @brief Check if this event is recording or already recorded
   *
   * @return true event is recording/recorded
   * @return false event is in init or invalid state
   */
  bool isRecordingOrRecorded() {
    assert(!Deleted_ && "chipstar::Event use after delete!");
    return EventStatus_ >= EVENT_STATUS_RECORDING;
  }

  /**
   * @brief check if this event is done recording
   *
   * @return true recoded
   * @return false not recorded
   */
  bool isFinished() {
    assert(!Deleted_ && "chipstar::Event use after delete!");
    return (EventStatus_ == EVENT_STATUS_RECORDED);
  }

  /**
   * @brief Get the chipstar::Event Status object
   *
   * @return event_status_e current event status
   */
  event_status_e getEventStatus() { return EventStatus_; }

  std::string getEventStatusStr() {
    switch (EventStatus_) {
    case EVENT_STATUS_INIT:
      return "EVENT_STATUS_INIT";
    case EVENT_STATUS_RECORDING:
      return "EVENT_STATUS_RECORDING";
    case EVENT_STATUS_RECORDED:
      return "EVENT_STATUS_RECORDED";
    default:
      return "INVALID_EVENT_STATUS";
    };
  }

  /**
   * @brief Enqueue this event in a given Queue
   *
   * @param chip_queue_ Queue in which to enque this event
   * @return true
   * @return false
   */
  virtual void recordStream(chipstar::Queue *ChipQueue) = 0;
  /**
   * @brief Wait for this event to complete
   *
   * @return true
   * @return false
   */
  virtual bool wait() = 0;

  /**
   * @brief Calculate absolute difference between completion timestamps of this
   * event and other
   *
   * @param other
   * @return float
   */
  virtual float getElapsedTime(chipstar::Event *Other) = 0;

  /**
   * @brief Toggle this event from the host.
   *
   */
  virtual void hostSignal() = 0;

#ifndef NDEBUG
  void markDeleted(bool State = true) {
    LOCK(EventMtx); // Deleted_
    Deleted_ = State;
  }
  bool isDeleted() {
    LOCK(EventMtx); // Deleted_
    return Deleted_;
  }
#endif
};

class Program {
  std::string ProgramName_;   ///< Program name.
  std::string ProgramSource_; ///< Program source code.

  /// Include headers.
  std::map<std::string, std::string> Headers_;

  /// Name expressions added before compilation as key to the
  /// map. After compilation they point to their lowered/mangled
  /// names. The map value may also be empty meaning the lowered name
  /// is unknown.
  std::map<std::string, std::string> NameExpressions_;

  std::string ProgramLog_; ///< Captured compilation log.
  std::string Code_;       ///< Compiled program.

public:
  Program() = delete;
  Program(const std::string &ProgramName) : ProgramName_(ProgramName) {}

  void setSource(const char *Source) {
    if (Source)
      ProgramSource_.assign(Source);
  }

  void addHeader(std::string_view IncludeName, std::string_view Contents) {
    assert(!IncludeName.empty() && "Nameless include header!");
    Headers_[std::string(IncludeName)].assign(Contents);
  }

  void appendToLog(const std::string &Log) { ProgramLog_.append(Log); }

  void addCode(std::string_view Code) { Code_.append(Code); }

  void addNameExpression(std::string_view NameExpr) {
    assert(!isAfterCompilation() &&
           "Must not add name expressions after compilation!");
    NameExpressions_.emplace(std::make_pair(NameExpr, ""));
  }

  const std::map<std::string, std::string> &getHeaders() const {
    return Headers_;
  }
  const std::string &getSource() const { return ProgramSource_; }
  const std::string &getProgramLog() const { return ProgramLog_; }
  const std::string &getCode() const { return Code_; }

  const std::map<std::string, std::string> &getNameExpressionMap() const {
    return NameExpressions_;
  }
  std::map<std::string, std::string> &getNameExpressionMap() {
    return NameExpressions_;
  }

  /// Return true if the program has been compiled.
  bool isAfterCompilation() const { return !Code_.empty(); }
};

/**
 * @brief Module abstraction. Contains global variables and kernels. Can be
 * extracted from FatBinary or loaded at runtime.
 * OpenCL - ClProgram
 * Level Zero - zeModule
 * ROCclr - amd::Program
 * CUDA - CUmodule
 */
class Module : public ihipModule_t {
  /// Flag for the allocation state of the device variables. True if
  /// all variables have space allocated for this module for the
  /// device this module is attached to. False implies that
  /// DeviceVariablesAllocated false.
  bool DeviceVariablesAllocated_ = false;
  /// Flag for the initialization state of the device variables. True
  /// if all variables are initialized for this module for the device
  /// this module is attached to.
  bool DeviceVariablesInitialized_ = false;

  OpenCLFunctionInfoMap FuncInfos_;

protected:
  uint8_t *FuncIL_;
  size_t IlSize_;
  std::mutex Mtx_;
  // Global variables
  std::vector<chipstar::DeviceVar *> ChipVars_;
  // Kernels
  std::vector<chipstar::Kernel *> ChipKernels_;
  /// Binary representation extracted from FatBinary.
  const SPVModule *Src_;
  // Kernel JIT compilation can be lazy
  std::once_flag Compiled_;

  uint32_t *BinaryData_;

  /**
   * @brief hidden default constuctor. Only derived type constructor should be
   * called.
   *
   */
  Module() = default;

public:
  /**
   * @brief Destroy the Module object
   *
   */
  virtual ~Module();
  /**
   * @brief Construct a new Module object.
   * This constructor should be implemented by the derived class (specific
   * backend implementation). Call to this constructor should result in a
   * populated chip_kernels vector.
   *
   * @param module_str string prepresenting the binary extracted from FatBinary
   */
  Module(const SPVModule &Src) : Src_(&Src) {}

  /**
   * @brief Add a chipstar::Kernel to this module.
   * During initialization when the FatBinary is consumed, a Module is
   * constructed for every device. SPIR-V kernels reside in this module. This
   * method is called called via the constructor during this initialization
   * phase. Modules can also be loaded from a file during runtime, however.
   *
   * @param kernel chipstar::Kernel to be added to this module.
   */
  void addKernel(chipstar::Kernel *Kernel);

  /**
   * @brief Wrapper around compile() called via std::call_once
   *
   * @param chip_dev device for which to compile the kernels
   */
  void compileOnce(chipstar::Device *ChipDev);
  /**
   * @brief chipstar::Kernel JIT compilation can be lazy. This is configured via
   * Cmake LAZY_JIT option. If LAZY_JIT is set to true then this module won't be
   * compiled until the first call to one of its kernels. If LAZY_JIT is set to
   * false(default) then this method should be called in the constructor;
   *
   * This method should populate this modules chip_kernels vector. These
   * kernels would have a name extracted from the kernel but no associated host
   * function pointers.
   *
   */
  virtual void compile(chipstar::Device *ChipDev) = 0;
  /**
   * @brief Get the Global Var object
   * A module, along with device kernels, can also contain global variables.
   *
   * @param name global variable name
   * @return DeviceVar*
   */
  virtual chipstar::DeviceVar *getGlobalVar(const char *VarName);

  /**
   * @brief Get the chipstar::Kernel object
   *
   * @param name name of the corresponding host function
   * @return Kernel* if found and nullptr otherwise.
   */
  chipstar::Kernel *findKernel(const std::string &Name);

  /**
   * @brief Get the chipstar::Kernel object
   *
   * @param name name of the corresponding host function
   * @return Kernel*
   */
  chipstar::Kernel *getKernelByName(const std::string &Name);

  /**
   * @brief Checks if the module has a kernel with the given name.
   *
   * @param name the name of the kernel
   * @return true in case the kernels is found
   */
  bool hasKernel(std::string Name);

  /**
   * @brief Get the Kernels object
   *
   * @return std::vector<Kernel*>&
   */
  std::vector<chipstar::Kernel *> &getKernels();

  /**
   * @brief Get the chipstar::Kernel object
   *
   * @param host_f_ptr host-side function pointer
   * @return Kernel*
   */
  chipstar::Kernel *getKernel(const void *HostFPtr);

  /**
   * @brief consume SPIRV and fill in SPVFuncINFO
   *
   */
  void consumeSPIRV();

  /**
   * @brief Record a device variable
   *
   * Takes ownership of the variable.
   */
  void addDeviceVariable(chipstar::DeviceVar *DevVar) {
    ChipVars_.push_back(DevVar);
  }

  std::vector<chipstar::DeviceVar *> &getDeviceVariables() { return ChipVars_; }

  hipError_t allocateDeviceVariablesNoLock(chipstar::Device *Device,
                                           chipstar::Queue *Queue);
  void prepareDeviceVariablesNoLock(chipstar::Device *Device,
                                    chipstar::Queue *Queue);
  void invalidateDeviceVariablesNoLock();
  void deallocateDeviceVariablesNoLock(chipstar::Device *Device);

  SPVFuncInfo *findFunctionInfo(const std::string &FName);

  const SPVModule &getSourceModule() const { return *Src_; }
};

/**
 * @brief Contains information about the function on the host and device
 */
class Kernel : public ihipModuleSymbol_t {
protected:
  /**
   * @brief hidden default constructor. Only derived type constructor should be
   * called.
   *
   */
  Kernel(std::string HostFName, SPVFuncInfo *FuncInfo);
  /// Name of the function
  std::string HostFName_;
  /// Pointer to the host function
  const void *HostFPtr_ = nullptr;
  /// Pointer to the device function
  const void *DevFPtr_;

  SPVFuncInfo *FuncInfo_;

public:
  virtual ~Kernel();

  /**
   * @brief Get the Name object
   *
   * @return std::string
   */
  std::string getName();

  /**
   * @brief Get the Func Info object
   *
   * @return SPVFuncInfo&
   */
  SPVFuncInfo *getFuncInfo();
  /**
   * @brief Get the associated host pointer to a host function
   *
   * @return const void*
   */
  const void *getHostPtr();
  /**
   * @brief Get the associated funciton pointer on the device
   *
   * @return const void*
   */
  const void *getDevPtr();

  /**
   * @brief Get the Name object
   *
   * @return std::string
   */
  void setName(std::string HostFName);
  /**
   * @brief Get the associated host pointer to a host function
   *
   * @return const void*
   */
  void setHostPtr(const void *HostFPtr);
  /**
   * @brief Get the associated funciton pointer on the device
   *
   * @return const void*
   */
  void setDevPtr(const void *DevFPtr);

  /**
   * @brief Get the associated kernel information (max threads and so on)
   *
   * @return hipError_t
   * @return hipFuncAttributes
   */
  virtual hipError_t getAttributes(hipFuncAttributes *Attr) = 0;

  /**
   * @brief Return the parent module of the kernel.
   */
  virtual chipstar::Module *getModule() = 0;
  virtual const chipstar::Module *getModule() const = 0;
};

class ArgSpillBuffer {
  chipstar::Context *Ctx_; ///< A context to allocate device space from.
  std::unique_ptr<char[]> HostBuffer_;
  char *DeviceBuffer_ = nullptr;
  std::map<size_t, size_t> ArgIndexToOffset_;
  size_t Size_ = 0;

public:
  ArgSpillBuffer() = delete;
  ArgSpillBuffer(chipstar::Context *Ctx) : Ctx_(Ctx) {}
  ~ArgSpillBuffer();
  void computeAndReserveSpace(const SPVFuncInfo &KernelInfo);
  void *allocate(const SPVFuncInfo::Arg &Arg);
  size_t getSize() const { return Size_; }
  const void *getHostBuffer() const {
    assert(HostBuffer_.get());
    return HostBuffer_.get();
  }
  void *getDeviceBuffer() {
    assert(DeviceBuffer_);
    return DeviceBuffer_;
  }
};

/**
 * @brief Contains kernel arguments and a queue on which to execute.
 * Prior to kernel launch, the arguments are setup via
 * Backend::configureCall(). Because of this, we get the kernel last so the
 * kernel so the launch() takes a kernel argument as opposed to queue receiving
 * a chipstar::ExecItem containing the kernel and arguments
 *
 */
class ExecItem {
protected:
  bool ArgsSetup = false;
  size_t SharedMem_;

  dim3 GridDim_;
  dim3 BlockDim_;

  chipstar::Queue *ChipQueue_;

  std::vector<void *> Args_;

  std::shared_ptr<chipstar::ArgSpillBuffer> ArgSpillBuffer_;

public:
  void copyArgs(void **Args);
  void setQueue(chipstar::Queue *Queue) { ChipQueue_ = Queue; }
  std::mutex ExecItemMtx;
  size_t getNumArgs() {
    assert(getKernel() &&
           "chipstar::Kernel was not set! (call  setKernel() first)");
    return getKernel()->getFuncInfo()->getNumClientArgs();
  }

  /**
   * @brief Return argument list.
   */
  const std::vector<void *> &getArgs() const { return Args_; }

  /**
   * @brief Deleted default constructor
   * Doesn't make sense for ExecItem to exist without arguments
   *
   */
  ExecItem() = delete;

  /**
   * @brief Deleted copy constructor
   * Since this is an abstract class and derived classes might add their own
   * members and overrides, we must request an implementation for clone()
   *
   * @param Other
   */
  ExecItem(const ExecItem &Other) = delete;

  /**
   * @brief Destroy the ExecItem object
   *
   */
  virtual ~ExecItem() {}

  virtual ExecItem *clone() const = 0;

  /**
   * @brief Construct a new ExecItem object
   *
   * @param grid_dim_
   * @param block_dim_
   * @param shared_mem_
   * @param chip_queue_
   */
  ExecItem(dim3 GirdDim, dim3 BlockDim, size_t SharedMem,
           hipStream_t ChipQueue);

  /**
   * @brief Set the chipstar::Kernel object
   *
   * @return Kernel* chipstar::Kernel to be executed
   */
  virtual void setKernel(chipstar::Kernel *Kernel) = 0;

  /**
   * @brief Get the chipstar::Kernel object
   *
   * @return Kernel* chipstar::Kernel to be executed
   */
  virtual chipstar::Kernel *getKernel() = 0;

  /**
   * @brief Get the Queue object
   *
   * @return Queue*
   */

  chipstar::Queue *getQueue();

  /**
   * @brief Get the Grid object
   *
   * @return dim3
   */
  dim3 getGrid();

  /**
   * @brief Get the Block object
   *
   * @return dim3
   */
  dim3 getBlock();

  /**
   * @brief Get the SharedMem
   *
   * @return size_t
   */
  size_t getSharedMem();

  /**
   * @brief Sets up the kernel arguments via backend API calls.
   * Called after all the arugments are setup either via hipSetupArg() (old HIP
   * kernel launch API)
   * Or after hipLaunchKernel (new HIP kernel launch API)
   *
   */
  virtual void setupAllArgs() = 0;

  std::shared_ptr<chipstar::ArgSpillBuffer> getArgSpillBuffer() const {
    return ArgSpillBuffer_;
  };
};

/**
 * @brief Compute device class
 */
class Device {

  // A bundle for CHIPReinitialize.
  class ModuleState {
    friend class Device;
    std::unordered_map<const SPVModule *, chipstar::Module *>
        SrcModToCompiledMod_;
    std::unordered_map<const void *, chipstar::Module *> HostPtrToCompiledMod_;
  };

  /// Modules compiled so far.
  std::unordered_map<const SPVModule *, chipstar::Module *>
      SrcModToCompiledMod_;
  /// Host pointer mapping to modules.
  std::unordered_map<const void *, chipstar::Module *> HostPtrToCompiledMod_;

protected:
  std::string DeviceName_;
  chipstar::Context *Ctx_;
  std::vector<chipstar::Queue *> ChipQueues_;
  std::once_flag PropsPopulated_;

  hipDeviceAttribute_t Attrs_;
  hipDeviceProp_t HipDeviceProps_;

  size_t TotalUsedMem_;
  size_t MaxUsedMem_;
  size_t MaxMallocSize_ = 0;

  /// Maps host-side shadow variables to the corresponding device variables.
  std::unordered_map<const void *, chipstar::DeviceVar *> DeviceVarLookup_;

  int Idx_ = -1; // Initialized with a value indicating unset ID.

  // only callable from derived classes, because we need to call also init()
  Device(chipstar::Context *Ctx, int DeviceIdx);
  // initializer. may call virtual methods
  void init();
  bool PerThreadStreamUsed_ = false;

public:
  hipDeviceProp_t getDeviceProps() { return HipDeviceProps_; }
  std::mutex DeviceVarMtx;
  std::mutex DeviceMtx;

  std::vector<chipstar::Queue *> getQueuesNoLock() { return ChipQueues_; }

  chipstar::Queue *LegacyDefaultQueue;
  inline static thread_local std::unique_ptr<chipstar::Queue>
      PerThreadDefaultQueue;

  /**
   * @brief Get the Legacy Default Queue object.
   *
   * @return Queue* default legacy queue
   */
  chipstar::Queue *getLegacyDefaultQueue();
  /**
   * @brief Get the Per Thread Default Queue object. If it was not initialized,
   * initialize it and set PerThreadStreamUsed to true
   * @see Device::PerThreadStreamUsed
   *
   * @return Queue*
   */
  chipstar::Queue *getPerThreadDefaultQueue();
  chipstar::Queue *getPerThreadDefaultQueueNoLock();

  bool isPerThreadStreamUsed();
  bool isPerThreadStreamUsedNoLock();
  void setPerThreadStreamUsed(bool Status);

  /**
   * @brief Get the Default Queue object. If HIP_API_PER_THREAD_DEFAULT_STREAM
   * was set during compilation, return PerThreadStream, otherwise return legacy
   * stream
   *
   * @return Queue*
   */
  chipstar::Queue *getDefaultQueue();

  /**
   * @brief Create a Queue object
   *
   * @param Flags
   * @param Priority
   * @return Queue*
   */
  chipstar::Queue *
  createQueueAndRegister(chipstar::QueueFlags Flags = chipstar::QueueFlags(),
                         int Priority = DEFAULT_QUEUE_PRIORITY);

  chipstar::Queue *createQueueAndRegister(const uintptr_t *NativeHandles,
                                          const size_t NumHandles);

  void removeContext(chipstar::Context *Ctx);
  virtual chipstar::Context *createContext() = 0;
  chipstar::Context *createContextAndRegister() {
    Ctx_ = createContext();
    return Ctx_;
  }

  size_t getMaxMallocSize() {
    if (MaxMallocSize_ < 1)
      CHIPERR_LOG_AND_THROW("MaxMallocSize was not set", hipErrorTbd);
    return MaxMallocSize_;
  }

  chipstar::AllocationTracker *AllocTracker = nullptr;

  virtual ~Device();

  /// Return kernel the host-pointer 'Ptr' is associated with, if
  /// found. Otherwise return nullptr.
  chipstar::Kernel *findKernel(HostPtr Ptr) {
    if (auto *Mod = getOrCreateModule(Ptr))
      return Mod->getKernel(Ptr);
    return nullptr;
  }

  chipstar::Module *getOrCreateModule(HostPtr Ptr);
  chipstar::Module *getOrCreateModule(const SPVModule &SrcMod);

  /// Return the number of currently compiled modules on this device.
  size_t getNumCompiledModules() const { return SrcModToCompiledMod_.size(); }

  /**
   * @brief Get the Kernels object
   *
   * @return std::vector<Kernel*>&
   */
  std::vector<chipstar::Kernel *> getKernels();

  ModuleState getModuleState() const {
    ModuleState State;
    State.SrcModToCompiledMod_ = SrcModToCompiledMod_;
    State.HostPtrToCompiledMod_ = HostPtrToCompiledMod_;
    return State;
  }

  void addFromModuleState(ModuleState &State) {
    for (auto &kv : State.SrcModToCompiledMod_)
      SrcModToCompiledMod_.insert(kv);
    for (auto &kv : State.HostPtrToCompiledMod_)
      HostPtrToCompiledMod_.insert(kv);
  }

  /**
   * @brief Use a backend to populate device properties such as memory
   * available, frequencies, etc.
   */
  virtual void populateDevicePropertiesImpl() = 0;

  /**
   * @brief Query the device for properties
   *
   * @param prop
   */
  void copyDeviceProperties(hipDeviceProp_t *Prop);

  /**
   * @brief Get the context object
   *
   * @return Context* pointer to the Context object this Device
   * was created with
   */
  chipstar::Context *getContext();

  /**
   * @brief Construct an additional queue for this device
   *
   * @param flags
   * @param priority
   * @return Queue* pointer to the newly created queue (can also be found
   * in chip_queues vector)
   */
  virtual chipstar::Queue *createQueue(chipstar::QueueFlags Flags,
                                       int Priority) = 0;
  virtual chipstar::Queue *createQueue(const uintptr_t *NativeHandles,
                                       int NumHandles) = 0;

  /**
   * @brief Add a queue to this device and the backend
   *
   * @param chip_queue_  chipstar::Queue to be added
   */
  void addQueue(chipstar::Queue *ChipQueue);
  /**
   * @brief Get the Queues object
   *
   * @return std::vector<chipstar::Queue*>
   */
  std::vector<chipstar::Queue *> &getQueues();

  /**
   * @brief Remove a queue from this device's queue vector
   *
   * @param q
   * @return true
   * @return false
   */
  bool removeQueue(chipstar::Queue *ChipQueue);

  /**
   * @brief Get the integer ID of this device as it appears in the Backend's
   * chip_devices list
   *
   * @return int
   */
  int getDeviceId();
  /**
   * @brief Get the device name
   *
   * @return std::string
   */
  std::string getName();

  /**
   * @brief Destroy all allocations and reset all state on the current device in
   the current process.
   *
   */
  void reset() {
    invalidateDeviceVariables();
    // resetImpl();
  }

  /**
   * @brief Query for a specific device attribute. Implementation copied from
   * HIPAMD.
   *
   * @param attr attribute to query
   * @return int attribute value. In case invalid query returns -1;
   */
  int getAttr(hipDeviceAttribute_t Attr);

  /**
   * @brief Get the total global memory available for this device.
   *
   * @return size_t
   */
  size_t getGlobalMemSize();

  /**
   * @brief Set the Cache Config object
   *
   * @param cfg configuration
   */
  virtual void setCacheConfig(hipFuncCache_t Cfg);

  /**
   * @brief Get the cache configuration for this device
   *
   * @return hipFuncCache_t
   */
  virtual hipFuncCache_t getCacheConfig();

  /**
   * @brief Configure shared memory for this device
   *
   * @param config
   */
  virtual void setSharedMemConfig(hipSharedMemConfig Cfg);

  /**
   * @brief Get the shared memory configuration for this device
   *
   * @return hipSharedMemConfig
   */
  virtual hipSharedMemConfig getSharedMemConfig();

  /**
   * @brief Setup the cache configuration for the device to use when executing
   * this function
   *
   * @param func
   * @param config
   */
  virtual void setFuncCacheConfig(const void *Func, hipFuncCache_t Cfg);

  /**
   * @brief Check if the current device has same PCI bus ID as the one given by
   * input
   *
   * @param pciDomainID
   * @param pciBusID
   * @param pciDeviceID
   * @return true
   * @return false
   */
  bool hasPCIBusId(int PciDomainID, int PciBusID, int PciDeviceID);

  /**
   * @brief Get peer-accesability between this and another device
   *
   * @param peerDevice
   * @return int
   */
  int getPeerAccess(chipstar::Device *PeerDevice);

  /**
   * @brief Set access between this and another device
   *
   * @param peer
   * @param flags
   * @param canAccessPeer
   * @return hipError_t
   */
  hipError_t setPeerAccess(chipstar::Device *Peer, int Flags,
                           bool CanAccessPeer);

  /**
   * @brief Get the total used global memory
   *
   * @return size_t
   */
  size_t getUsedGlobalMem();

  /**
   * @brief Get the global variable that came from a FatBinary module
   *
   * @param var host pointer to the variable
   * @return DeviceVar* if not found returns nullptr
   */
  chipstar::DeviceVar *getDynGlobalVar(const void *Var) { return nullptr; }

  /**
   * @brief Get the global variable that came from a FatBinary module
   *
   * @param var Pointer to host side shadow variable.
   * @return DeviceVar* if not found returns nullptr
   */
  chipstar::DeviceVar *getStatGlobalVar(const void *Var);

  /**
   * @brief Get the global variable
   *
   * @param var Pointer to host side shadow variable.
   * @return DeviceVar* if not found returns nullptr
   */
  chipstar::DeviceVar *getGlobalVar(const void *Var);

  void eraseModule(chipstar::Module *Module);

  virtual chipstar::Texture *
  createTexture(const hipResourceDesc *ResDesc, const hipTextureDesc *TexDesc,
                const struct hipResourceViewDesc *ResViewDesc) = 0;

  virtual void destroyTexture(chipstar::Texture *TextureObject) = 0;

  void prepareDeviceVariables(HostPtr Ptr);
  void invalidateDeviceVariables();
  void deallocateDeviceVariables();

protected:
  /**
   * @brief The backend hook for reset().
   */
  virtual void resetImpl() = 0;

  /// Compile the source (SPIR-V) to native/backend code.
  virtual chipstar::Module *compile(const SPVModule &Src) = 0;
};

/**
 * @brief Context class
 * Contexts contain execution queues and are created on top of a single or
 * multiple devices. Provides for creation of additional queues, events, and
 * interaction with devices.
 */
class Context : public ihipCtx_t {
protected:
  int RefCount_;
  chipstar::Device *ChipDevice_;
  std::vector<void *> AllocatedPtrs_;

  unsigned int Flags_;

  /**
   * @brief Construct a new Context object
   *
   */
  Context();

public:
  mutable std::mutex ContextMtx;

  /**
   * @brief Destroy the Context object
   *
   */
  virtual ~Context();

  virtual void syncQueues(chipstar::Queue *TargetQueue);

  void setDevice(chipstar::Device *Device) { ChipDevice_ = Device; }

  /**
   * @brief Get this context's CHIPDevices
   *
   * @return chipstar::Device *
   */
  chipstar::Device *getDevice();

  /**
   * @brief Allocate data.
   * Calls reserveMem() to keep track memory used on the device.
   * Calls Context::allocate_(size_t size, size_t alignment,
   * hipMemoryType mem_type) with allignment = 0
   *
   * @param size size of the allocation
   * @param mem_type type of the allocation: Host, Device, Shared
   * @return void* pointer to allocated memory
   */
  void *allocate(size_t Size, hipMemoryType MemType);

  /**
   * @brief Allocate data.
   * Calls reserveMem() to keep track memory used on the device.
   * Calls Context::allocate_(size_t size, size_t alignment,
   * hipMemoryType mem_type)
   *
   * @param size size of the allocation
   * @param alignment allocation alignment in bytes
   * @param mem_type type of the allocation: Host, Device, Shared
   * @return void* pointer to allocated memory
   */
  void *allocate(size_t Size, size_t Alignment, hipMemoryType MemType);

  /**
   * @brief Allocate data.
   * Calls reserveMem() to keep track memory used on the device.
   * Calls Context::allocate_(size_t size, size_t alignment,
   * hipMemoryType mem_type)
   *
   * @param size size of the allocation
   * @param alignment allocation alignment in bytes
   * @param mem_type type of the allocation: Host, Device, Shared
   * @param Flags flags
   * @return void* pointer to allocated memory
   */
  void *allocate(size_t Size, size_t Alignment, hipMemoryType MemType,
                 chipstar::HostAllocFlags Flags);

  /**
   * @brief Allocate data. Pure virtual function - to be overriden by each
   * backend. This member function is the one that's called by all the
   * publically visible Context::allocate() variants
   *
   * @param size size of the allocation.
   * @param alignment allocation alignment in bytes
   * @param mem_type type of the allocation: Host, Device, Shared
   * @return void*
   */
  virtual void *
  allocateImpl(size_t Size, size_t Alignment, hipMemoryType MemType,
               chipstar::HostAllocFlags Flags = chipstar::HostAllocFlags()) = 0;

  /**
   * @brief Returns true if the pointer is mapped to virtual memory with
   * updates synchronized to it automatically at synchronization points.
   *
   * @param Ptr pointer to memory allocated by allocate().
   * @return true/false
   */

  virtual bool isAllocatedPtrMappedToVM(void *Ptr) = 0;

  /**
   * @brief Free memory
   *
   * @param ptr pointer to the memory location to be deallocated. Internally
   * calls Context::free_()
   * @return true Success
   * @return false Failure
   */
  hipError_t free(void *Ptr);

  /**
   * @brief Free memory
   * To be overriden by the backend
   *
   * @param ptr
   * @return true
   * @return false
   */
  virtual void freeImpl(void *Ptr) = 0;

  /**
   * @brief Get the flags set on this context
   *
   * @return unsigned int context flags
   */
  unsigned int getFlags();

  /**
   * @brief Set the flags for this context
   *
   * @param flags flags to set on this context
   */
  void setFlags(unsigned int Flags);

  /**
   * @brief Reset this context.
   *
   */
  void reset();

  /**
   * @brief Retain this context.
   * TODO: What does it mean to retain a context?
   *
   * @return Context*
   */
  void retain() { ++RefCount_; }

  void release() {
    --RefCount_;
    if (RefCount_ == 0) {
      // TODO hipCtx - shuold call overlaoded destructor
    }
  }
};

/**
 * @brief Primary object to interact with the backend
 */
class Backend {
protected:
  chipstar::EventMonitor *CallbackEventMonitor_ = nullptr;
  chipstar::EventMonitor *StaleEventMonitor_ = nullptr;

  int MinQueuePriority_;
  int MaxQueuePriority_ = 0;

  chipstar::Context *ActiveCtx_;
  chipstar::Device *ActiveDev_;

  // Keep hold on the default logger instance to make sure that it is
  // not destructed before the backend finishes uninitialization.
  std::shared_ptr<spdlog::logger> Logger;

public:
  std::shared_ptr<chipstar::Event> userEventLookup(chipstar::Event *EventPtr) {
    std::lock_guard<std::mutex> Lock(UserEventsMtx);
    for (auto &UserEvent : UserEvents) {
      if (UserEvent.get() == EventPtr) {
        return UserEvent;
      }
    }
    return nullptr;
  }
  void trackEvent(const std::shared_ptr<chipstar::Event> &Event);

#ifdef DUBIOUS_LOCKS
  std::mutex DubiousLockOpenCL;
  std::mutex DubiousLockLevel0;
#endif

  virtual chipstar::ExecItem *createExecItem(dim3 GirdDim, dim3 BlockDim,
                                             size_t SharedMem,
                                             hipStream_t ChipQueue) = 0;

  int getPerThreadQueuesActive();
  std::mutex SetActiveMtx;
  std::mutex QueueCreateDestroyMtx;
  mutable std::mutex BackendMtx;
  std::mutex CallbackQueueMtx;
  std::vector<std::shared_ptr<chipstar::Event>> Events;
  std::vector<std::shared_ptr<chipstar::Event>> UserEvents;
  std::mutex EventsMtx;
  std::mutex UserEventsMtx;

  std::queue<chipstar::CallbackData *> CallbackQueue;

  std::vector<chipstar::Context *> ChipContexts;

  /**
   * @brief User defined compiler options to pass to the JIT compiler
   *
   */
  std::string CustomJitFlags;

  int getQueuePriorityRange();

  /**
   * @brief Get the default compiler flags for the JIT compiler
   *
   * @return std::string flags to pass to JIT compiler
   */
  virtual std::string getDefaultJitFlags() = 0;

  virtual int ReqNumHandles() = 0;
  /**
   * @brief Get the jit options object
   * return CHIP_JIT_FLAGS if it is set, otherwise return default options as
   * defined by Backend<implementation>::getDefaultJitFlags()
   *
   * @return std::string flags to pass to JIT compiler
   */
  std::string getJitFlags();

  // TODO
  // key for caching compiled modules. To get a cached compiled module on a
  // particular device you must make sure that you have a module which matches
  // the host funciton pointer and also that this module was compiled for the
  // same device model.
  // typedef  std::pair<const void*, std::string> ptr_dev;
  // /**
  //  * @brief
  //  *
  //  */
  // std::unordered_map<ptr_dev, Module*> host_f_ptr_to_chipmodule_map;

  /**
   * @brief Construct a new Backend object
   *
   */
  Backend();
  /**
   * @brief Destroy the Backend objectk
   *
   */
  virtual ~Backend();

  /**
   * @brief Initialize this backend with given environment flags
   *
   * @param platform_str
   * @param device_type_str
   * @param device_ids_str
   */
  void initialize(std::string PlatformStr, std::string DeviceTypeStr,
                  std::string DeviceIdStr);

  /**
   * @brief Initialize this backend with given environment flags
   *
   * @param platform_str
   * @param device_type_str
   * @param device_ids_str
   */
  virtual void initializeImpl(std::string PlatformStr,
                              std::string DeviceTypeStr,
                              std::string DeviceIdStr) = 0;

  /**
   * @brief Initialize this backend with given Native handles
   *
   * @param platform_str
   * @param device_type_str
   * @param device_ids_str
   */
  virtual void initializeFromNative(const uintptr_t *NativeHandles,
                                    int NumHandles) = 0;

  /**
   * @brief
   *
   */
  virtual void uninitialize() = 0;

  /**
   * @brief Wait for all per-thread queues to finish
   *
   */
  void waitForThreadExit();
  /**
   * @brief Get the Active Context object. Returns the context of the active
   * queue.
   *
   * @return Context*
   */
  chipstar::Context *getActiveContext();
  /**
   * @brief Get the Active Device object. Returns the device of the active
   * queue.
   *
   * @return Device*
   */
  chipstar::Device *getActiveDevice();
  /**
   * @brief Set the active device. Sets the active queue to this device's
   * first/default/primary queue.
   *
   * @param chip_dev
   */
  void setActiveContext(chipstar::Context *ChipContext);
  void setActiveDevice(chipstar::Device *ChipDevice);

  std::vector<chipstar::Device *> getDevices();
  /**
   * @brief Get the Num Devices object
   *
   * @return size_t
   */
  size_t getNumDevices();
  /**
   * @brief Add a context to this backend.
   *
   * @param ctx_in
   */
  void addContext(chipstar::Context *ChipContext);
  void removeContext(chipstar::Context *ChipContext);

  /**
   * @brief Configure an upcoming kernel call
   *
   * @param grid
   * @param block
   * @param shared
   * @param q
   * @return hipError_t
   */
  hipError_t configureCall(dim3 GridDim, dim3 BlockDim, size_t SharedMem,
                           hipStream_t ChipQueue);

  /**
   * @brief Return a device which meets or exceeds the requirements
   *
   * @param props
   * @return Device*
   */
  chipstar::Device *findDeviceMatchingProps(const hipDeviceProp_t *Props);

  /**
   * @brief Find a given queue in this backend.
   *
   * @param q queue to find
   * @return Queue* return queue or nullptr if not found
   */
  chipstar::Queue *findQueue(chipstar::Queue *ChipQueue);

  /**
   * @brief Add a chipstar::Module to every initialized device
   *
   * @param chip_module pointer to chipstar::Module object
   * @return hipError_t
   */
  // Module* addModule(std::string* module_src);
  /**
   * @brief Remove this module from every device
   *
   * @param chip_module pointer to the module which is to be removed
   * @return hipError_t
   */
  // hipError_t removeModule(Module* chip_module);

  /************Factories***************/

  virtual chipstar::Queue *createCHIPQueue(chipstar::Device *ChipDev) = 0;

  /**
   * @brief Create an chipstar::Event, adding it to the Backend chipstar::Event
   * list.
   *
   * @param ChipCtx Context in which to create the event in
   * @param Flags Events falgs
   * @param UserEvent Is this a user event? If so, increase refcount to 2 to
   * prevent it from being garbage collected.
   * @return chipstar::Event* chipstar::Event
   */
  virtual std::shared_ptr<chipstar::Event>
  createCHIPEvent(chipstar::Context *ChipCtx,
                  chipstar::EventFlags Flags = chipstar::EventFlags(),
                  bool UserEvent = false) = 0;
  /**
   * @brief Create a Callback Obj object
   * Each backend must implement this function which calls a derived
   * chipstar::CallbackData constructor.
   * @return chipstar::CallbackData* pointer to newly allocated
   * chipstar::CallbackData object.
   */
  virtual chipstar::CallbackData *
  createCallbackData(hipStreamCallback_t Callback, void *UserData,
                     chipstar::Queue *ChipQ) = 0;

  virtual chipstar::EventMonitor *createCallbackEventMonitor_() = 0;
  virtual chipstar::EventMonitor *createStaleEventMonitor_() = 0;

  /* event interop */
  virtual hipEvent_t getHipEvent(void *NativeEvent) = 0;
  virtual void *getNativeEvent(hipEvent_t HipEvent) = 0;
};

/**
 * @brief Queue class for submitting kernels to for execution
 */
class Queue : public ihipStream_t {
protected:
  hipStreamCaptureStatus CaptureStatus_ = hipStreamCaptureStatusNone;
  hipStreamCaptureMode CaptureMode_ = hipStreamCaptureModeGlobal;
  hipGraph_t CaptureGraph_;
  std::mutex LastEventMtx;
  /// @brief  node for creating a dependency chain between subsequent record
  /// events when in graph capture mode
  CHIPGraphNode *LastNode_ = nullptr;
  int Priority_;
  /**
   * @brief Maximum priority that can be had by a queue is 0; Priority range is
   * defined to be [0, MinPriority]
   */
  const int MaxPriority = 0;
  int MinPriority;
  chipstar::QueueFlags QueueFlags_;
  /// Device on which this queue will execute
  chipstar::Device *ChipDevice_;
  /// Context to which device belongs to
  chipstar::Context *ChipContext_;

  /** Keep track of what was the last event submitted to this queue. Required
   * for enforcing proper queue syncronization as per HIP/CUDA API. */
  std::shared_ptr<chipstar::Event> LastEvent_ = nullptr;

  enum class MANAGED_MEM_STATE { PRE_KERNEL, POST_KERNEL };

  std::shared_ptr<chipstar::Event>
  RegisteredVarCopy(chipstar::ExecItem *ExecItem, MANAGED_MEM_STATE ExecState);

public:
  enum MEM_MAP_TYPE { HOST_READ, HOST_WRITE, HOST_READ_WRITE };
  virtual void MemMap(const chipstar::AllocationInfo *AllocInfo,
                      MEM_MAP_TYPE MapType) {}
  virtual void MemUnmap(const chipstar::AllocationInfo *AllocInfo) {}

  /**
   * @brief Check the stream to see if it's in capture mode and if so, capture.
   *
   * @tparam GraphNodeType the type of graph node to create
   * @tparam ArgTypes variadic template parameter
   * @param ArgsPack graph node type constructor arguments
   * @return true stream was in capture mode and a graph node was created -
   * caller should return from whatever HIP API function was invoking this
   * @return false stream was not in capture mode, proceed with executing the
   * HIP API call.
   */
  template <class GraphNodeType, class... ArgTypes>
  bool captureIntoGraph(ArgTypes... ArgsPack) {
    if (getCaptureStatus() == hipStreamCaptureStatusActive) {
      auto Graph = getCaptureGraph();
      auto Node = new GraphNodeType(ArgsPack...);
      updateLastNode(Node);
      Graph->addNode(Node);
      return true;
    }
    return false;
  }

  void updateLastNode(CHIPGraphNode *NewNode);
  void initCaptureGraph();

  hipStreamCaptureStatus getCaptureStatus() const { return CaptureStatus_; }
  void setCaptureStatus(hipStreamCaptureStatus CaptureMode) {
    CaptureStatus_ = CaptureMode;
  }
  hipStreamCaptureMode getCaptureMode() const { return CaptureMode_; }
  void setCaptureMode(hipStreamCaptureMode CaptureMode) {
    CaptureMode_ = CaptureMode;
  }
  CHIPGraph *getCaptureGraph() const;

  chipstar::Device *PerThreadQueueForDevice = nullptr;

  // I want others to be able to lock this queue?
  std::mutex QueueMtx;

  virtual std::shared_ptr<chipstar::Event> getLastEvent() {
    LOCK(LastEventMtx); // Queue::LastEvent_
    return LastEvent_;
  }

  /**
   * @brief Construct a new Queue object
   *
   * @param chip_dev
   * @param flags
   */
  Queue(chipstar::Device *ChipDev, chipstar::QueueFlags Flags);
  /**
   * @brief Construct a new Queue object
   *
   * @param chip_dev
   * @param flags
   * @param priority
   */
  Queue(chipstar::Device *ChipDev, chipstar::QueueFlags Flags, int Priority);
  /**
   * @brief Destroy the Queue object
   *
   */
  virtual ~Queue();

  chipstar::QueueFlags getQueueFlags() { return QueueFlags_; }
  virtual void
  updateLastEvent(const std::shared_ptr<chipstar::Event> &NewEvent) {
    LOCK(LastEventMtx); // CHIPQueue::LastEvent_
    logDebug("Setting LastEvent for {} {} -> {}", (void *)this,
             (void *)LastEvent_.get(), (void *)NewEvent.get());
    if (NewEvent == LastEvent_) // TODO: should I compare NewEvent.get()
      return;

    LastEvent_ = NewEvent;
  }

  /**
   * @brief Blocking memory copy
   *
   * @param dst Destination
   * @param src Source
   * @param size Transfer size
   * @return hipError_t
   */
  hipError_t memCopy(void *Dst, const void *Src, size_t Size);

  /**
   * @brief Non-blocking memory copy
   *
   * @param dst Destination
   * @param src Source
   * @param size Transfer size
   * @return hipError_t
   */
  virtual std::shared_ptr<chipstar::Event>
  memCopyAsyncImpl(void *Dst, const void *Src, size_t Size) = 0;
  void memCopyAsync(void *Dst, const void *Src, size_t Size);

  /**
   * @brief Blocking memset
   *
   * @param dst
   * @param size
   * @param pattern
   * @param pattern_size
   */
  virtual void memFill(void *Dst, size_t Size, const void *Pattern,
                       size_t PatternSize);

  /**
   * @brief Non-blocking mem set
   *
   * @param dst
   * @param size
   * @param pattern
   * @param pattern_size
   */
  virtual std::shared_ptr<chipstar::Event>
  memFillAsyncImpl(void *Dst, size_t Size, const void *Pattern,
                   size_t PatternSize) = 0;
  virtual void memFillAsync(void *Dst, size_t Size, const void *Pattern,
                            size_t PatternSize);

  // The memory copy 2D support
  virtual void memCopy2D(void *Dst, size_t DPitch, const void *Src,
                         size_t SPitch, size_t Width, size_t Height);

  virtual std::shared_ptr<chipstar::Event>
  memCopy2DAsyncImpl(void *Dst, size_t DPitch, const void *Src, size_t SPitch,
                     size_t Width, size_t Height) = 0;
  virtual void memCopy2DAsync(void *Dst, size_t DPitch, const void *Src,
                              size_t SPitch, size_t Width, size_t Height);

  // The memory copy 3D support
  virtual void memCopy3D(void *Dst, size_t DPitch, size_t DSPitch,
                         const void *Src, size_t SPitch, size_t SSPitch,
                         size_t Width, size_t Height, size_t Depth);

  virtual std::shared_ptr<chipstar::Event>
  memCopy3DAsyncImpl(void *Dst, size_t DPitch, size_t DSPitch, const void *Src,
                     size_t SPitch, size_t SSPitch, size_t Width, size_t Height,
                     size_t Depth) = 0;
  virtual void memCopy3DAsync(void *Dst, size_t DPitch, size_t DSPitch,
                              const void *Src, size_t SPitch, size_t SSPitch,
                              size_t Width, size_t Height, size_t Depth);

  /**
   * @brief Submit a chipstar::ExecItem to this queue for execution. ExecItem
   * needs to be complete - contain the kernel and arguments
   *
   * @param exec_item
   * @return hipError_t
   */
  virtual std::shared_ptr<chipstar::Event>
  launchImpl(chipstar::ExecItem *ExecItem) = 0;
  virtual void launch(chipstar::ExecItem *ExecItem);

  /**
   * @brief Get the Device obj
   *
   * @return Device*
   */

  chipstar::Device *getDevice();
  /**
   * @brief Wait for this queue to finish.
   *
   */

  virtual void finish() = 0;
  /**
   * @brief Check if the queue is still actively executing
   *
   * @return true
   * @return false
   */

  bool query() {
    if (!LastEvent_)
      return true;

    if (LastEvent_->updateFinishStatus(false))
      if (LastEvent_->isFinished())
        return true;

    return false;
  };

  /**
   * @brief Insert an event into this queue
   *
   * @param e
   * @return true
   * @return false
   */
  virtual std::shared_ptr<chipstar::Event> enqueueBarrierImpl(
      const std::vector<std::shared_ptr<chipstar::Event>> &EventsToWaitFor) = 0;
  virtual std::shared_ptr<chipstar::Event> enqueueBarrier(
      const std::vector<std::shared_ptr<chipstar::Event>> &EventsToWaitFor);

  virtual std::shared_ptr<chipstar::Event> enqueueMarkerImpl() = 0;
  std::shared_ptr<chipstar::Event> enqueueMarker();

  /**
   * @brief Get the Flags object with which this queue was created.
   *
   * @return unsigned int
   */

  chipstar::QueueFlags getFlags();
  /**
   * @brief Get the Priority object with which this queue was created.
   *
   * @return int
   */

  int getPriority();
  /**
   * @brief Add a callback funciton to be called on the host after the specified
   * stream is done
   *
   * @param callback function pointer for a ballback function
   * @param userData
   * @return true
   * @return false
   */

  virtual void addCallback(hipStreamCallback_t Callback, void *UserData);
  /**
   * @brief Insert a memory prefetch
   *
   * @param ptr
   * @param count
   * @return true
   * @return false
   */

  virtual std::shared_ptr<chipstar::Event> memPrefetchImpl(const void *Ptr,
                                                           size_t Count) = 0;
  void memPrefetch(const void *Ptr, size_t Count);

  /**
   * @brief Launch a kernel on this queue given a host pointer and arguments
   *
   * @param hostFunction
   * @param numBlocks
   * @param dimBlocks
   * @param args
   * @param sharedMemBytes
   */
  void launchKernel(chipstar::Kernel *ChipKernel, dim3 NumBlocks,
                    dim3 DimBlocks, void **Args, size_t SharedMemBytes);

  /**
   * @brief returns Native backend handles for a stream
   *
   * @param NativeHandles storage for handles
   * @param NumHandles variable to hold number of returned handles
   * @return for Level0 backend, returns { ze_driver_handle_t,
   * ze_device_handle_t, ze_context_handle_t, ze_command_queue_handle_t }
   * @return for OpenCL backend, returns { cl_platform_id, cl_device_id,
   * cl_context, cl_command_queue }
   */
  virtual hipError_t getBackendHandles(uintptr_t *NativeHandles,
                                       int *NumHandles) = 0;

  chipstar::Context *getContext() { return ChipContext_; }
  void setFlags(chipstar::QueueFlags TheFlags) { QueueFlags_ = TheFlags; }
};

} // namespace chipstar

inline chipstar::Context *PrimaryContext = nullptr;
inline thread_local std::stack<chipstar::ExecItem *> ChipExecStack;
inline thread_local std::stack<chipstar::Context *> ChipCtxStack;

#endif
