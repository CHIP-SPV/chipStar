/**
 * @file HIPxxBackend.hh
 * @author Paulius Velesko (pvelesko@gmail.com)
 * @brief HIPxxBackend class definition. HIPxx backends are to inherit from this
 * base class and override desired virtual functions. Overrides for this class
 * are expected to be minimal with primary overrides being done on lower-level
 * classes such as HIPxxContext consturctors, etc.
 * @version 0.1
 * @date 2021-08-19
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef HIPXX_BACKEND_H
#define HIPXX_BACKEND_H

#include <algorithm>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

// Forward Declarations
class HIPxxDevice;
class HIPxxContext;
class HIPxxModule;
class HIPxxKernel;
class HIPxxBackend;

#include "HIPxxDriver.hh"
#include "logging.hh"
#include "temporary.hh"

class HIPxxModule {
 protected:
  std::vector<HIPxxKernel*> Kernels;

 public:
  HIPxxModule(){};
  ~HIPxxModule(){};

  HIPxxModule(std::string* module_str);
  void add_kernel(void* HostFunctionPtr, std::string HostFunctionName);
};

/**
 * @brief Contains information about the function on the host and device
 */
class HIPxxKernel {
 protected:
  /// Name of the function
  std::string HostFunctionName;
  /// Pointer to the host function
  void* HostFunctionPointer;
  /// Pointer to the device function
  void* DeviceFunctionPointer;

 public:
  HIPxxKernel(){};
  ~HIPxxKernel(){};
};

/**
 * @brief a HIPxxKernel and argument container to be submitted to HIPxxQueue
 */
class HIPxxExecItem {
 protected:
  /// Kernel to be ejjxecuted
  HIPxxKernel* Kernel;
  // TODO Args
 public:
  virtual void run(){};
};

/**
 * @brief Context class
 * Contexts contain execution queues and are created on top of a single or
 * multiple devices. Provides for creation of additional queues, events, and
 * interaction with devices.
 */
class HIPxxContext {
 protected:
  std::vector<HIPxxDevice*> Devices;

 public:
  HIPxxContext(){};
  ~HIPxxContext(){};

  /**
   * @brief Add a device to this context
   *
   * @param dev pointer to HIPxxDevice object
   * @return true if device was added successfully
   * @return false upon failure
   */
  bool add_device(HIPxxDevice* dev);
};

/**
 * @brief Compute device class
 */
class HIPxxDevice {
 protected:
  std::mutex DeviceMutex;

 public:
  /// Vector of contexts to which this device belongs to
  std::vector<HIPxxContext*> hipxx_contexts;
  /// hipxx_modules in binary representation
  std::vector<std::string*> ModulesStr;
  /// hipxx_modules in parsed representation
  std::vector<HIPxxModule*> hipxx_modules;

  /// Map host pointer to module in binary representation
  std::map<const void*, std::string*> HostPtrToModuleStrMap;
  /// Map host pointer to module in parsed representation
  std::map<const void*, HIPxxModule*> HostPtrToModuleMap;
  /// Map host pointer to a function name
  std::map<const void*, std::string> HostPtrToNameMap;

  /// default constructor
  HIPxxDevice(){};
  /// default desctructor
  ~HIPxxDevice(){};

  /**
   * @brief Add a context to the vector of HIPxxContexts* to which this device
   * belongs to
   * @param ctx pointer to HIPxxContext object
   * @return true if added successfully
   * @return false if failed to add
   */
  bool add_context(HIPxxContext* ctx);

  bool registerFunction(std::string* module_str, const void* HostFunction,
                        const char* FunctionName);

  /**
   * @brief Get the default context object
   *
   * @return HIPxxContext* pointer to the 0th element in the internal
   * context array
   */
  HIPxxContext* get_default_context();
  virtual std::string get_name() = 0;
};

/**
 * @brief Queue class for submitting kernels to for execution
 */
class HIPxxQueue {
 protected:
 public:
  /// Device on which this queue will execute
  HIPxxDevice* hipxx_device;
  /// Context to which device belongs to
  HIPxxContext* hipxx_context;
  HIPxxQueue(){};
  ~HIPxxQueue(){};

  /// Submit a kernel for execution
  virtual void submit(HIPxxExecItem* exec_item) {
    logDebug("HIPxxQueue.submit() Base Call");
  };

  virtual std::string get_info() {
    std::string info;
    info = hipxx_device->get_name();
    return info;
  };
};

/**
 * @brief Primary object to interact with the backend
 */
class HIPxxBackend {
 protected:
  /**
   * @brief hipxx_modules stored in binary representation.
   * During compilation each translation unit is parsed for functions that are
   * marked for execution on the device. These functions are then compiled to
   * device code and stored in binary representation.
   *  */
  std::vector<std::string*> ModulesStr;

 public:
  std::vector<HIPxxContext*> hipxx_contexts;
  std::vector<HIPxxQueue*> hipxx_queues;
  std::vector<HIPxxDevice*> hipxx_devices;
  HIPxxBackend() { logDebug("HIPxxBackend Base Constructor"); };
  ~HIPxxBackend(){};
  virtual void initialize(std::string HIPxxPlatformStr,
                          std::string HIPxxDeviceTypeStr,
                          std::string HIPxxDeviceStr) = 0;

  virtual void uninitialize() = 0;

  HIPxxQueue* get_default_queue() {
    if (hipxx_queues.size() == 0) {
      logCritical(
          "HIPxxBackend.get_default_queue() was called but no queues have been "
          "initialized;\n",
          "");
      std::abort();
    }

    return hipxx_queues[0];
  };
  std::vector<HIPxxDevice*> get_devices() { return hipxx_devices; }
  size_t get_num_devices() { return hipxx_devices.size(); }
  std::vector<std::string*> get_modules_str() { return ModulesStr; }
  void add_context(HIPxxContext* ctx_in) { hipxx_contexts.push_back(ctx_in); }
  void add_queue(HIPxxQueue* q_in) {
    logDebug("HIPxxBackend.add_queue()");
    hipxx_queues.push_back(q_in);
  }
  void add_device(HIPxxDevice* dev_in) { hipxx_devices.push_back(dev_in); }
  void submit(HIPxxExecItem* _e) {
    logDebug("HIPxxBackend.submit()");
    get_default_queue()->submit(_e);
  };

  void register_module(std::string* mod_str) {
    logTrace("HIPxxBackend->register_module()");
    get_modules_str().push_back(mod_str);
  };

  void unregister_module(std::string* mod_str) {
    logTrace("HIPxxBackend->unregister_module()");
    auto found_mod = std::find(ModulesStr.begin(), ModulesStr.end(), mod_str);
    if (found_mod != ModulesStr.end()) {
      get_modules_str().erase(found_mod);
    } else {
      logWarn(
          "Module {} not found in HIPxxBackend.ModulesStr while trying to "
          "unregister",
          (void*)mod_str);
    }
  };
};

#endif
