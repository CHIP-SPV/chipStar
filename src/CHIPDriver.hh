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
  DeviceType() {}
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
  BackendType(){};
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
  DeviceType Device_;
  int DeviceIdx_ = 0;
  BackendType Backend_;
  bool DumpSpirv_ = false;
  bool SkipUninit_ = false;
  bool LazyJit_ = true;
  std::string JitFlags_ = CHIP_DEFAULT_JIT_FLAGS;
  unsigned long L0EventTimeout_ = 0;
  int L0CollectEventsTimeout_ = 0;

public:
  EnvVars() {
    parseEnvironmentVariables();
    logDebugSettings();
  }

  int getPlatformIdx() const { return PlatformIdx_; }
  DeviceType getDevice() const { return Device_; }
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

    return L0EventTimeout_ * 1e9;
  }

private:
  void parseEnvironmentVariables() {
    // Parse all the environment variables and set the class members
    if (!readEnvVar("CHIP_PLATFORM").empty())
      PlatformIdx_ = parseInt("CHIP_PLATFORM");

    Device_ = DeviceType(readEnvVar("CHIP_DEVICE_TYPE"));

    if (!readEnvVar("CHIP_DEVICE").empty())
      DeviceIdx_ = parseInt("CHIP_DEVICE");

    Backend_ = BackendType(readEnvVar("CHIP_BE"));

    if (!readEnvVar("CHIP_DUMP_SPIRV").empty())
      DumpSpirv_ = parseBoolean("CHIP_DUMP_SPIRV");

    if (!readEnvVar("CHIP_SKIP_UNINIT").empty())
      SkipUninit_ = parseBoolean("CHIP_SKIP_UNINIT");

    if (!readEnvVar("CHIP_LAZY_JIT").empty())
      LazyJit_ = parseBoolean("CHIP_LAZY_JIT");

    JitFlags_ = parseJitFlags("CHIP_JIT_FLAGS_OVERRIDE");

    if (!readEnvVar("CHIP_L0_COLLECT_EVENTS_TIMEOUT").empty())
      L0CollectEventsTimeout_ = parseInt("CHIP_L0_COLLECT_EVENTS_TIMEOUT");

    if (!readEnvVar("CHIP_L0_EVENT_TIMEOUT").empty())
      L0EventTimeout_ = parseInt("CHIP_L0_EVENT_TIMEOUT");
  }

  std::string_view parseJitFlags(const std::string &StrIn) {
    if (readEnvVar(StrIn).empty())
      return CHIP_DEFAULT_JIT_FLAGS;

    return JitFlags_;
  }

  int parseInt(const std::string &StrIn) {
    const auto &Str = readEnvVar(StrIn);
    if (!isConvertibleToInt(Str))
      CHIPERR_LOG_AND_THROW("Invalid integer value: " + Str,
                            hipErrorInitializationError);
    return std::stoi(Str);
  }

  bool parseBoolean(const std::string &StrIn) {
    const auto &Str = readEnvVar(StrIn);
    if (Str == "1" || Str == "on")
      return true;
    if (Str == "0" || Str == "off")
      return false;
    CHIPERR_LOG_AND_THROW("Invalid boolean value: " + Str + "while parsing " +
                              StrIn,
                          hipErrorInitializationError);
    return false; // This return is never reached
  }

  void logDebugSettings() const {
    // Log the current settings
    logDebug("CHIP_PLATFORM={}", PlatformIdx_);
    logDebug("CHIP_DEVICE_TYPE={}", Device_.str());
    logDebug("CHIP_DEVICE={}", DeviceIdx_);
    logDebug("CHIP_BE={}", Backend_.str());
    logDebug("CHIP_DUMP_SPIRV={}", DumpSpirv_ ? "on" : "off");
    logDebug("CHIP_JIT_FLAGS_OVERRIDE={}", JitFlags_);
    logDebug("CHIP_L0_COLLECT_EVENTS_TIMEOUT={}", L0CollectEventsTimeout_);
    logDebug("CHIP_L0_EVENT_TIMEOUT={}", L0EventTimeout_);
    logDebug("CHIP_SKIP_UNINIT={}", SkipUninit_ ? "on" : "off");
  }
};

extern EnvVars ChipEnvVars;

#endif
