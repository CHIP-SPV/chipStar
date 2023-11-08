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

const char *CHIPGetBackendName();

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
  enum Type { GPU, CPU, ACCEL, FPGA, DEFAULT };

private:
  Type Type_;

public:
  DeviceType(Type TypeIn) : Type_(TypeIn) {}

  std::string str() const {
    switch (Type_) {
    case GPU:
      return "gpu";
    case CPU:
      return "cpu";
    case ACCEL:
      return "accel";
    case FPGA:
      return "fpga";  
    case DEFAULT:
      return "default";
    default:
      return "unknown";
    }
  }

  Type getType() const { return Type_; }
};

class BackendType {
public:
  enum Type { OPENCL, LEVEL0, DEFAULT };

private:
  Type Type_;

public:
  BackendType(Type TypeIn) : Type_(TypeIn) {}

  const char* str() const {
    switch (Type_) {
    case OPENCL:
      return "opencl";
    case LEVEL0:
      return "level0";
      case DEFAULT:
      return "default";
    default:
      return "unknown";
    }
  }

  Type getType() const { return Type_; }
};

class EnvVars {
public:
  int PlatformIdx;
  DeviceType Device;
  int DeviceIdx;
  BackendType Backend;
  bool DumpSpirv;
  std::string JitFlags;
  bool L0ImmCmdLists;
  int L0CollectEventsTimeout;

  EnvVars() : Device(DeviceType::GPU), Backend(BackendType::DEFAULT) {
    parseEnvironmentVariables();
    logDebugSettings();
  }

  std::string str() const {
    // String representation of ChipEnvVars
    return "PlatformIdx: " + std::to_string(PlatformIdx) +
           ", Device: " + Device.str() +
           ", DeviceIdx: " + std::to_string(DeviceIdx) +
           ", Backend: " + Backend.str() +
           ", DumpSpirv: " + (DumpSpirv ? "true" : "false") +
           ", JitFlags: " + JitFlags +
           ", L0ImmCmdLists: " + (L0ImmCmdLists ? "true" : "false") +
           ", L0CollectEventsTimeout: " +
           std::to_string(L0CollectEventsTimeout);
  }

private:
  void parseEnvironmentVariables() {
    // Parse all the environment variables and set the class members
    PlatformIdx = parseInt("CHIP_PLATFORM");
    Device = parseDeviceType("CHIP_DEVICE_TYPE");
    DeviceIdx = parseInt("CHIP_DEVICE");
    Backend = parseBackendType("CHIP_BE");
    DumpSpirv = parseBoolean("CHIP_DUMP_SPIRV");
    JitFlags = parseJitFlags("CHIP_JIT_FLAGS_OVERRIDE");
    L0ImmCmdLists = parseBoolean("CHIP_L0_IMM_CMD_LISTS");
    L0CollectEventsTimeout = parseInt("CHIP_L0_COLLECT_EVENTS_TIMEOUT");
  }

  std::string parseJitFlags(const std::string &StrIn) {
    auto str = readEnvVar(StrIn);
    if (str.empty()) {
      return CHIP_DEFAULT_JIT_FLAGS;
    }
    return str;
  }
  

  int parseInt(const std::string &StrIn) {
    auto str = readEnvVar(StrIn);
    if (!isConvertibleToInt(str)) {
      CHIPERR_LOG_AND_THROW("Invalid integer value: " + str,
                            hipErrorInitializationError);
    }
    return std::stoi(str);
  }

  bool parseBoolean(const std::string &StrIn) {
    auto str = readEnvVar(StrIn);
    if (str == "1" || str == "on") {
      return true;
    } else if (str == "0" || str == "off") {
      return false;
    }
    CHIPERR_LOG_AND_THROW("Invalid boolean value: " + str,
                          hipErrorInitializationError);
    return false; // This return is never reached
  }

  DeviceType parseDeviceType(const std::string &StrIn) {
    auto str = readEnvVar(StrIn);
    if (str == "gpu") {
      return DeviceType(DeviceType::GPU);
    } else if (str == "cpu") {
      return DeviceType(DeviceType::CPU);
    } else if (str == "accel") {
      return DeviceType(DeviceType::ACCEL);
    } else if (str == "fpga") {
      return DeviceType(DeviceType::FPGA);
    } else if (str == "default") {
      return DeviceType(DeviceType::DEFAULT);
    }
    CHIPERR_LOG_AND_THROW("Invalid device type value: " + str,
                          hipErrorInitializationError);
    return DeviceType(DeviceType::GPU); // This return is never reached
  }

  BackendType parseBackendType(const std::string &StrIn) {
    auto str = readEnvVar(StrIn);
    if (str == "opencl") {
      return BackendType(BackendType::OPENCL);
    } else if (str == "level0") {
      return BackendType(BackendType::LEVEL0);
    } else if (str == "default") {
      return BackendType(BackendType::DEFAULT);
    }
    CHIPERR_LOG_AND_THROW("Invalid backend type value: " + str,
                          hipErrorInitializationError);
  }

  void logDebugSettings() const {
    // Log the current settings
    logDebug("CHIP_PLATFORM=", std::to_string(PlatformIdx));
    logDebug("CHIP_DEVICE_TYPE=", Device.str());
    logDebug("CHIP_DEVICE=", std::to_string(DeviceIdx));
    logDebug("CHIP_BE=", Backend.str());
    logDebug("CHIP_DUMP_SPIRV=", DumpSpirv ? "on" : "off");
    logDebug("CHIP_JIT_FLAGS_OVERRIDE=", JitFlags);
    logDebug("CHIP_L0_IMM_CMD_LISTS=", L0ImmCmdLists ? "on" : "off");
    logDebug("CHIP_L0_COLLECT_EVENTS_TIMEOUT=",
             std::to_string(L0CollectEventsTimeout));
  }
};

extern EnvVars ChipEnvVars;

#endif
