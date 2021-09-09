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
#include <stack>

#include "include/hip/hip.hh"

#include "HIPxxDriver.hh"
#include "logging.hh"
#include "temporary.hh"

// fw declares
class HIPxxExecItem;
class HIPxxQueue;
class HIPxxContext;
class HIPxxDevice;

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
  std::mutex mtx;

 public:
  HIPxxModule(){};
  ~HIPxxModule(){};

  HIPxxModule(std::string* module_str);
  void add_kernel(void* HostFunctionPtr, std::string HostFunctionName);

  /**
   * @brief Take a binary representation of a module, compile it, extract
   * kernels and add them to this module;
   *
   * @param module_str binary representation of a module
   */
  virtual void compile(std::string* module_str) = 0;
};

/**
 * @brief Contains information about the function on the host and device
 */
class HIPxxKernel {
 protected:
  /// Name of the function
  std::string HostFunctionName;
  /// Pointer to the host function
  const void* HostFunctionPointer;
  /// Pointer to the device function
  const void* DeviceFunctionPointer;

 public:
  HIPxxKernel(){};
  ~HIPxxKernel(){};
  std::string get_name() { return HostFunctionName; }
  // TODO Error Handling?
  const void* get_host_ptr() { return HostFunctionPointer; }
  const void* get_device_ptr() { return DeviceFunctionPointer; }
};

/**
 * @brief a HIPxxKernel and argument container to be submitted to HIPxxQueue
 */
class HIPxxExecItem {
 protected:
  size_t SharedMem;
  hipStream_t Stream;
  std::vector<uint8_t> ArgData;
  std::vector<std::tuple<size_t, size_t>> OffsetsSizes;

 public:
  HIPxxKernel* Kernel;
  HIPxxQueue* q;
  dim3 GridDim;
  dim3 BlockDim;
  HIPxxExecItem(dim3 grid_in, dim3 block_in, size_t shared_in, hipStream_t q_in)
      : GridDim(grid_in), BlockDim(block_in), SharedMem(shared_in), q(q_in){};

  void set_arg(const void* arg, size_t size, size_t offset) {
    if ((offset + size) > ArgData.size()) ArgData.resize(offset + size + 1024);

    std::memcpy(ArgData.data() + offset, arg, size);
    logDebug("HIPxxExecItem.set_arg() on {} size {} offset {}\n", (void*)this,
             size, offset);
    OffsetsSizes.push_back(std::make_tuple(offset, size));
  }
  virtual hipError_t launch(HIPxxKernel* Kernel) {
    logWarn("Calling HIPxxExecItem.launch() base launch which does nothing");
    return hipSuccess;
  };

  virtual hipError_t launchByHostPtr(const void* hostPtr);
};
/**
 * @brief Compute device class
 */
class HIPxxDevice {
 protected:
  std::string device_name;
  std::mutex mtx;
  std::vector<HIPxxKernel*> hipxx_kernels;

 public:
  /// Vector of contexts to which this device belongs to
  std::vector<HIPxxContext*> hipxx_contexts;
  /// hipxx_modules in binary representation
  std::vector<std::string*> modules_str;
  /// hipxx_modules in parsed representation
  std::vector<HIPxxModule*> hipxx_modules;

  /// Map host pointer to module in binary representation
  std::map<const void*, std::string*> HostPtrToModuleStrMap;
  /// Map host pointer to module in parsed representation
  std::map<const void*, HIPxxModule*> HostPtrToModuleMap;
  /// Map host pointer to a function name
  std::map<const void*, std::string> HostPtrToNameMap;
  /// Map host pointer to HIPxxKernel
  std::map<const void*, HIPxxKernel*> HostPtrToKernelStrMap;

  // TODO
  std::vector<HIPxxKernel*>& get_kernels() { return hipxx_kernels; };
  void add_kernel(HIPxxKernel* kernel) {
    logTrace("Adding kernel {} to device # {} {}", kernel->get_name(), pcie_idx,
             device_name);
    hipxx_kernels.push_back(kernel);
  }

  hipDevice_t pcie_idx;
  hipDeviceProp_t hip_device_props;

  size_t TotalUsedMem, MaxUsedMem;

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

  bool register_function_as_kernel(std::string* module_str,
                                   const void* HostFunction,
                                   const char* FunctionName);

  HIPxxKernel* findKernelByHostPtr(const void* hostPtr);

  /**
   * @brief Get the default context object
   *
   * @return HIPxxContext* pointer to the 0th element in the internal
   * context array
   */
  HIPxxContext* get_default_context();
  virtual std::string get_name() = 0;

  bool reserve_mem(size_t bytes);

  bool release_mem(size_t bytes);

  bool getModuleAndFName(const void* HostFunction, std::string& FunctionName,
                         HIPxxModule* hipxx_module) {
    logTrace("HIPxxDevice.getModuleAndFName");
    std::lock_guard<std::mutex> Lock(mtx);

    auto it1 = HostPtrToModuleMap.find(HostFunction);
    auto it2 = HostPtrToNameMap.find(HostFunction);

    if ((it1 == HostPtrToModuleMap.end()) || (it2 == HostPtrToNameMap.end()))
      return false;

    FunctionName.assign(it2->second);
    hipxx_module = it1->second;
    return true;
  }
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
  std::vector<HIPxxQueue*> hipxx_queues;
  std::mutex mtx;

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
  hipError_t launchHostFunc(const void* HostFunction);

  std::vector<HIPxxDevice*>& get_devices() {
    if (hipxx_devices.size() == 0)
      logWarn(
          "HIPxxContext.get_devices() was called but hipxx_devices is empty");
    return hipxx_devices;
  }

  std::vector<HIPxxQueue*>& get_queues() {
    if (hipxx_queues.size() == 0) {
      logCritical(
          "HIPxxContext.get_queues() was called but no queues were added to "
          "this context");
      std::abort();
    }
    return hipxx_queues;
  }
  void add_queue(HIPxxQueue* q) { hipxx_queues.push_back(q); }

  virtual hipError_t memCopy(void* dst, const void* src, size_t size,
                             hipStream_t stream) = 0;

  hipStream_t findQueue(hipStream_t stream) {
    std::vector<HIPxxQueue*> Queues = get_queues();
    HIPxxQueue* DefaultQueue = Queues.at(0);
    if (stream == nullptr || stream == DefaultQueue) return DefaultQueue;

    auto I = std::find(Queues.begin(), Queues.end(), stream);
    if (I == Queues.end()) return nullptr;
    return *I;
  }
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
  std::mutex mtx;

 public:
  std::stack<HIPxxExecItem*> hipxx_execstack;
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

  std::vector<HIPxxQueue*>& get_queues() { return hipxx_queues; }
  HIPxxQueue* get_default_queue() {
    if (hipxx_queues.size() == 0) {
      logCritical(
          "HIPxxBackend.get_default_queue() was called but no queues have "
          "been initialized;\n");
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

  std::vector<HIPxxDevice*>& get_devices() { return hipxx_devices; }
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

  void register_module(std::string* mod_str) {
    logTrace("HIPxxBackend->register_module()");
    std::lock_guard<std::mutex> Lock(mtx);
    get_modules_str().push_back(mod_str);
  };

  void unregister_module(std::string* mod_str) {
    logTrace("HIPxxBackend->unregister_module()");
    auto found_mod = std::find(modules_str.begin(), modules_str.end(), mod_str);
    if (found_mod != modules_str.end()) {
      get_modules_str().erase(found_mod);
    } else {
      logWarn(
          "Module {} not found in HIPxxBackend.modules_str while trying to "
          "unregister",
          (void*)mod_str);
    }
  }

  hipError_t configure_call(dim3 grid, dim3 block, size_t shared,
                            hipStream_t q) {
    logTrace("HIPxxBackend->configure_call()");
    std::lock_guard<std::mutex> Lock(mtx);
    if (q == nullptr) q = get_default_queue();
    HIPxxExecItem* ex = new HIPxxExecItem(grid, block, shared, q);
    hipxx_execstack.push(ex);

    return hipSuccess;
  }

  hipError_t set_arg(const void* arg, size_t size, size_t offset) {
    logTrace("HIPxxBackend->set_arg()");
    std::lock_guard<std::mutex> Lock(mtx);
    HIPxxExecItem* ex = hipxx_execstack.top();
    ex->set_arg(arg, size, offset);

    return hipSuccess;
  }

  /**
   * @brief Register this function as a kernel for all devices initialized in
   * this backend
   *
   * @param module_str
   * @param HostFunctionPtr
   * @param FunctionName
   * @return true
   * @return false
   */
  virtual bool register_function_as_kernel(std::string* module_str,
                                           const void* HostFunctionPtr,
                                           const char* FunctionName) = 0;
};

/**
 * @brief Queue class for submitting kernels to for execution
 */
class HIPxxQueue {
 protected:
  std::mutex mtx;

 public:
  /// Device on which this queue will execute
  HIPxxDevice* hipxx_device;
  /// Context to which device belongs to
  HIPxxContext* hipxx_context;

  // TODO these should take device and context as arguments.
  HIPxxQueue(){};
  ~HIPxxQueue(){};

  virtual hipError_t memCopy(void* dst, const void* src, size_t size) = 0;

  /// Submit a kernel for execution
  virtual hipError_t launch(HIPxxExecItem* exec_item) = 0;

  virtual std::string get_info() {
    std::string info;
    info = hipxx_device->get_name();
    return info;
  }

  HIPxxDevice* get_device() {
    if (hipxx_device == nullptr) {
      logCritical(
          "HIPxxQueue.get_device() was called but device is a null pointer");
      std::abort();  // TODO Exception?
    }

    return hipxx_device;
  }
};

#endif
