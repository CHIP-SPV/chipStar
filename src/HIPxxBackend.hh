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

#include "include/hip/hip.hh"

#include "HIPxxDriver.hh"
#include "logging.hh"
#include "temporary.hh"

class HIPxxEvent {
 protected:
  std::mutex mutex;
  HIPxxQueue* hipxx_queue;
  event_status_e status;
  unsigned flags;
  HIPxxContext* hipxx_context;

 public:
  HIPxxEvent(HIPxxContext* ctx_in, unsigned flags_in)
      : status(EVENT_STATUS_INIT), flags(flags_in), hipxx_context(ctx_in) {}

  HIPxxEvent() {
    // TODO:
  }

  virtual ~HIPxxEvent() {
    // TODO
    // if (Event) delete Event;
  }

  // virtual uint64_t getFinishTime();
  // virtual cl::Event getEvent() { return *Event; }
  // virtual bool isFromContext(cl::Context& Other) { return (Context == Other);
  // } virtual bool isFromStream(hipStream_t& Other) { return (Stream == Other);
  // } virtual bool isFinished() const { return (Status ==
  // EVENT_STATUS_RECORDED); } virtual bool isRecordingOrRecorded() const {
  //   return (Status >= EVENT_STATUS_RECORDING);
  // }
  // virtual bool recordStream(hipStream_t S, cl_event E);
  // virtual bool updateFinishStatus();
  // virtual bool wait();
};

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
  std::vector<HIPxxDevice*> hipxx_devices;

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
  virtual void* allocate(size_t size) = 0;
  std::vector<HIPxxDevice*>& get_devices() {
    if (hipxx_devices.size() == 0)
      logWarn(
          "HIPxxContext.get_devices() was called but hipxx_devices is empty");
    return hipxx_devices;
  };
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

  hipDevice_t pcie_idx;
  hipDeviceProp_t hip_device_props;

  /// default constructor
  HIPxxDevice() {
    logDebug("Device {} is {}: name \"{}\" \n", pcie_idx, (void*)this,
             hip_device_props.name);
  };
  /// default desctructor
  ~HIPxxDevice(){};

  /**
   * @brief Use a backend to populate device properties such as memory
   * available, frequencies, etc.
   */
  virtual void populate_device_properties() = 0;
  void copy_device_properties(hipDeviceProp_t* prop) {
    logTrace("HIPxxDevice->copy_device_properties()");
    if (prop)
      std::memcpy(prop, &this->hip_device_props, sizeof(hipDeviceProp_t));
  }

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
  std::vector<std::string*> modules_str;

 public:
  std::vector<HIPxxContext*> hipxx_contexts;
  std::vector<HIPxxQueue*> hipxx_queues;
  std::vector<HIPxxDevice*> hipxx_devices;
  HIPxxBackend() { logDebug("HIPxxBackend Base Constructor"); };
  ~HIPxxBackend(){};
  virtual void initialize(std::string HIPxxPlatformStr,
                          std::string HIPxxDeviceTypeStr,
                          std::string HIPxxDeviceStr){};

  virtual void initialize() = 0;

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

  HIPxxContext* get_default_context() {
    if (hipxx_contexts.size() == 0) {
      logCritical(
          "HIPxxBackend.get_default_context() was called but no contexts "
          "have been "
          "initialized;\n",
          "");
      std::abort();
    }
    return hipxx_contexts[0];
  };

  std::vector<HIPxxDevice*> get_devices() { return hipxx_devices; }
  size_t get_num_devices() { return hipxx_devices.size(); }
  std::vector<std::string*>& get_modules_str() { return modules_str; }
  void add_context(HIPxxContext* ctx_in) { hipxx_contexts.push_back(ctx_in); }
  void add_queue(HIPxxQueue* q_in) {
    logDebug("HIPxxBackend.add_queue()");
    hipxx_queues.push_back(q_in);
  }
  void add_device(HIPxxDevice* dev_in) {
    logTrace("HIPxxDeviceOpenCL.add_device() {}", dev_in->get_name());
    hipxx_devices.push_back(dev_in);
  }

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
    auto found_mod = std::find(modules_str.begin(), modules_str.end(), mod_str);
    if (found_mod != modules_str.end()) {
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
