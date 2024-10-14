/*
 * Copyright (c) 2021-22 chipStar developers
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
 * @file CHIPDriver.hh
 * @author Paulius Velesko (pvelesko@pglc.io)
 * @brief Header defining global CHIP classes and functions such as
 * Backend type pointer Backend which gets initialized at the start of
 * execution.
 * @version 0.1
 * @date 2021-08-19
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef CHIP_DRIVER_H
#define CHIP_DRIVER_H
#include <iostream>
#include <mutex>
#include <atomic>

/// Prevent more than one HIP API call from executing at the same time
extern std::mutex ApiMtx;

#include "Utils.hh"
#include "CHIPException.hh"
#include "chipStarConfig.hh"

// // Forward Declares
// class ExecItem;
// // class Device;
// // class Context;
// class Module;
// class Kernel;
// // class Backend;
// // class chipstar::Event;
// class Queue;
// // class Texture;

// /* HIP Graph API */
// class CHIPGraph;
// class CHIPGraphExec;
// class CHIPGraphNode;

namespace chipstar {
class Backend;
}

#include "CHIPBackend.hh"

/**
 * @brief
 * Global Backend pointer through which backend-specific operations are
 * performed
 */
extern chipstar::Backend *Backend;

/**
 * @brief
 * Singleton backend initialization flag
 */
extern std::once_flag Initialized;
extern std::once_flag Uninitialized;

/**
 * @brief
 * Singleton backend initialization function outer wrapper
 */
extern void CHIPInitialize();

/**
 * @brief
 * Singleton backend initialization function outer wrapper
 */
extern void CHIPUninitialize();

/**
 * @brief
 * Singleton backend initialization function called via std::call_once
 */
void CHIPInitializeCallOnce();

/**
 * @brief
 * Singleton backend uninitialization function called via std::call_once
 */
void CHIPUninitializeCallOnce();

extern hipError_t CHIPReinitialize(const uintptr_t *NativeHandles,
                                   int NumHandles);

/**
 * Number of fat binaries registerer through __hipRegisterFatBinary(). On
 * program exit this value (non-zero) will postpone chipStar runtime
 * uninitialization until the all the registered binaries have been
 * unregistered through __hipUnregisterFatBinary().
 */
extern std::atomic_ulong CHIPNumRegisteredFatBinaries;

/**
 * Keeps the track of the hipError_t from the last HIP API call.
 */
extern thread_local hipError_t CHIPTlsLastError;

class DeviceType {
public:
  enum Type { GPU, CPU, Accelerator, FPGA, Default };

private:
  Type Type_;

public:
  DeviceType(Type t) : Type_(t) {}
  DeviceType() : Type_(Default) {}
  DeviceType(const std::string &StrIn) {
    if (StrIn == "gpu")
      Type_ = DeviceType::GPU;
    else if (StrIn == "cpu")
      Type_ = DeviceType::CPU;
    else if (StrIn == "accel")
      Type_ = DeviceType::Accelerator;
    else if (StrIn == "fpga")
      Type_ = DeviceType::FPGA;
    else if (StrIn == "")
      Type_ = DeviceType::Default;
    else
      CHIPERR_LOG_AND_THROW("Invalid device type value: " + StrIn,
                            hipErrorInitializationError);
  }

  std::string_view str() const {
    switch (Type_) {
    case GPU:
      return "gpu";
    case CPU:
      return "cpu";
    case Accelerator:
      return "accel";
    case FPGA:
      return "fpga";
    case Default:
      return "default";
    default:
      assert(!"Unknown device type!");
      return "unknown";
    }
  }

  Type getType() const { return Type_; }
};

class BackendType {
public:
  enum Type { OpenCL, Level0, Default };

private:
  Type Type_;

public:
  BackendType() : Type_(Default) {}
  BackendType(Type t) : Type_(t) {}
  BackendType(const std::string &StrIn) {
    if (StrIn == "opencl") {
      Type_ = BackendType::OpenCL;
#ifndef HAVE_OPENCL
      assert(!"Invalid chipStar Backend Selected. This chipStar "
              "was not compiled with OpenCL backend");
#endif
    } else if (StrIn == "level0") {
      Type_ = BackendType::Level0;
#ifndef HAVE_LEVEL0
      assert(!"Invalid chipStar Backend Selected. This chipStar "
              "was not compiled with Level Zero backend");
#endif
    } else if (StrIn == "") {
#ifdef HAVE_LEVEL0
      Type_ = BackendType::Level0;
#elif HAVE_OPENCL
      Type_ = BackendType::OpenCL;
#else
      CHIPERR_LOG_AND_THROW("Invalid chipStar Backend Selected. This chipStar "
                            "was not compiled with OpenCL or Level0 backend",
                            hipErrorInitializationError);
#endif
    } else
      CHIPERR_LOG_AND_THROW("Invalid backend type value: " + StrIn,
                            hipErrorInitializationError);
  }

  const char *str() const {
    switch (Type_) {
    case OpenCL:
      return "opencl";
    case Level0:
      return "level0";
    case Default:
      return "default";
    default:
      assert(!"Unknown backend type!");
      return "unknown";
    }
  }

  Type getType() const { return Type_; }
};

class EnvVars {
private:
  int PlatformIdx_ = 0;
  DeviceType Device_{DeviceType::GPU};
  int DeviceIdx_ = 0;
  BackendType Backend_{BackendType::OpenCL};
  bool DumpSpirv_ = false;
  bool SkipUninit_ = false;
  bool LazyJit_ = true;
  std::string JitFlags_ = CHIP_DEFAULT_JIT_FLAGS;
  unsigned long L0EventTimeout_ = 0;
  int L0CollectEventsTimeout_ = 0;
  bool OCLDisableQueueProfiling_ = false;
  std::optional<std::string> OclUseAllocStrategy_;
  std::optional<std::string> ModuleCacheDir_;

public:
  EnvVars() {
    parseEnvironmentVariables();
    logDebugSettings();
  }

  int getPlatformIdx() const { return PlatformIdx_; }
  DeviceType getDevice() const { return Device_; }
  int getDeviceType() const { return Device_.getType(); }
  int getDeviceIdx() const { return DeviceIdx_; }
  BackendType getBackend() const { return Backend_; }
  bool getDumpSpirv() const { return DumpSpirv_; }
  bool getSkipUninit() const { return SkipUninit_; }
  const std::string &getJitFlags() const { return JitFlags_; }
  bool getLazyJit() const { return LazyJit_; }
  int getL0CollectEventsTimeout() const { return L0CollectEventsTimeout_; }
  unsigned long getL0EventTimeout() const {
    if (L0EventTimeout_ == 0)
      return UINT64_MAX;

    return L0EventTimeout_;
  }
  bool getOCLDisableQueueProfiling() const { return OCLDisableQueueProfiling_; }
  const std::optional<std::string> &getOclUseAllocStrategy() const noexcept {
    return OclUseAllocStrategy_;
  }
  const std::optional<std::string> &getModuleCacheDir() const {
    return ModuleCacheDir_;
  }

private:
  void parseEnvironmentVariables() {
    std::string value;

    PlatformIdx_ =
        readEnvVar("CHIP_PLATFORM", value) ? parseInt(value) : PlatformIdx_;
    Device_ =
        readEnvVar("CHIP_DEVICE_TYPE", value) ? DeviceType(value) : Device_;
    DeviceIdx_ =
        readEnvVar("CHIP_DEVICE", value) ? parseInt(value) : DeviceIdx_;
    Backend_ = readEnvVar("CHIP_BE", value) ? BackendType(value) : Backend_;
    DumpSpirv_ =
        readEnvVar("CHIP_DUMP_SPIRV", value) ? parseBoolean(value) : DumpSpirv_;
    SkipUninit_ = readEnvVar("CHIP_SKIP_UNINIT", value) ? parseBoolean(value)
                                                        : SkipUninit_;
    LazyJit_ =
        readEnvVar("CHIP_LAZY_JIT", value) ? parseBoolean(value) : LazyJit_;
    JitFlags_ =
        readEnvVar("CHIP_JIT_FLAGS_OVERRIDE", value, false) ? value : JitFlags_;
    L0CollectEventsTimeout_ =
        readEnvVar("CHIP_L0_COLLECT_EVENTS_TIMEOUT", value)
            ? parseInt(value)
            : L0CollectEventsTimeout_;
    L0EventTimeout_ = readEnvVar("CHIP_L0_EVENT_TIMEOUT", value)
                          ? parseInt(value)
                          : L0EventTimeout_;
    OCLDisableQueueProfiling_ =
        readEnvVar("CHIP_OCL_DISABLE_QUEUE_PROFILING", value)
            ? parseBoolean(value)
            : OCLDisableQueueProfiling_;
    OclUseAllocStrategy_ =
        readEnvVar("CHIP_OCL_USE_ALLOC_STRATEGY", value, true)
            ? value
            : OclUseAllocStrategy_;
    if (readEnvVar("CHIP_MODULE_CACHE_DIR", value, true)) {
      if (value.size())
        ModuleCacheDir_ = value;
    } else {
      ModuleCacheDir_ = "/tmp"; // If not set, default to "/tmp"
    }
  }

  int parseInt(const std::string &value) {
    if (value.empty())
      CHIPERR_LOG_AND_THROW("Empty value for integer environment variable",
                            hipErrorInitializationError);
    if (!isConvertibleToInt(value))
      CHIPERR_LOG_AND_THROW("Invalid integer value: " + value,
                            hipErrorInitializationError);
    int intValue = std::stoi(value);
    if (intValue < 0) {
      CHIPERR_LOG_AND_THROW("Negative value not allowed: " + value,
                            hipErrorInitializationError);
    }
    return intValue;
  }

  bool parseBoolean(const std::string &value) {
    if (value == "1" || value == "on")
      return true;
    if (value == "0" || value == "off")
      return false;
    CHIPERR_LOG_AND_THROW("Invalid boolean value: " + value,
                          hipErrorInitializationError);
    return false; // This return is never reached
  }

  void logDebugSettings() const {
    // Log the current settings
    logInfo("CHIP_PLATFORM={}", PlatformIdx_);
    logInfo("CHIP_DEVICE_TYPE={}", Device_.str());
    logInfo("CHIP_DEVICE={}", DeviceIdx_);
    logInfo("CHIP_BE={}", Backend_.str());
    logInfo("CHIP_DUMP_SPIRV={}", DumpSpirv_ ? "on" : "off");
    logInfo("CHIP_JIT_FLAGS_OVERRIDE={}", JitFlags_);
    logInfo("CHIP_L0_COLLECT_EVENTS_TIMEOUT={}", L0CollectEventsTimeout_);
    logInfo("CHIP_L0_EVENT_TIMEOUT={}", L0EventTimeout_);
    logInfo("CHIP_SKIP_UNINIT={}", SkipUninit_ ? "on" : "off");
    logInfo("CHIP_LAZY_JIT={}", LazyJit_ ? "on" : "off");
    logInfo("CHIP_OCL_DISABLE_QUEUE_PROFILING={}",
            OCLDisableQueueProfiling_ ? "on" : "off");
    logInfo("CHIP_OCL_USE_ALLOC_STRATEGY={}", OclUseAllocStrategy_.has_value()
                                                  ? OclUseAllocStrategy_.value()
                                                  : "off");
    logInfo("CHIP_MODULE_CACHE_DIR={}",
            ModuleCacheDir_.has_value() ? ModuleCacheDir_.value() : "off");
  }
};

extern EnvVars ChipEnvVars;

#endif