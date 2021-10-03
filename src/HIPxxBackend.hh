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

#include "spirv.hh"
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
  HIPxxEvent(HIPxxContext* ctx_, unsigned flags_);
  HIPxxEvent();
  ~HIPxxEvent();

  virtual bool recordStream(HIPxxQueue* hipxx_queue_);
  virtual bool wait();
  virtual bool isFinished();
  virtual float getElapsedTime(HIPxxEvent* other);
};

class HIPxxModule {
 protected:
  std::vector<HIPxxKernel*> hipxx_kernels;
  std::mutex mtx;

 public:
  HIPxxModule();
  ~HIPxxModule();
  HIPxxModule(std::string* module_str);

  void addKernel(void* host_f_ptr, std::string host_f_name);

  /**
   * @brief Take a binary representation of a module, compile it, extract
   * kernels and add them to this module;
   *
   * @param module_str binary representation of a module
   */
  virtual void compile(std::string* module_str);
  virtual bool getsymboladdresssize(const char* name, void* dptr,
                                    size_t* bytes) = 0;
  virtual bool symbolsupported() = 0;
};

/**
 * @brief Contains information about the function on the host and device
 */
class HIPxxKernel {
 protected:
  /// Name of the function
  std::string host_f_name;
  /// Pointer to the host function
  const void* host_f_ptr;
  /// Pointer to the device function
  const void* dev_f_ptr;

 public:
  HIPxxKernel();
  ~HIPxxKernel();
  std::string getName();
  const void* getHostPtr();
  const void* getDevPtr();
};

/**
 * @brief a HIPxxKernel and argument container to be submitted to HIPxxQueue
 */
class HIPxxExecItem {
 protected:
  size_t shared_mem;
  hipStream_t stream;
  std::vector<uint8_t> arg_data;
  std::vector<std::tuple<size_t, size_t>> offset_sizes;

 public:
  HIPxxKernel* hipxx_kernel;
  HIPxxQueue* hipxx_queue;
  dim3 grid_dim;
  dim3 block_dim;

  HIPxxExecItem(dim3 grid_dim_, dim3 block_dim_, size_t shared_mem_,
                hipStream_t hipxx_queue_);
  ~HIPxxExecItem();

  void setArg(const void* arg, size_t size, size_t offset);
  virtual hipError_t launch(HIPxxKernel* Kernel);
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
  std::map<const void*, std::string*> host_ptr_to_module_str_map;
  /// Map host pointer to module in parsed representation
  std::map<const void*, HIPxxModule*> host_ptr_to_hipxxmodule_map;
  /// Map host pointer to a function name
  std::map<const void*, std::string> host_ptr_to_name_map;
  /// Map host pointer to HIPxxKernel
  std::map<const void*, HIPxxKernel*> host_ptr_to_hipxxkernel_map;

  hipDevice_t global_id;
  hipDeviceProp_t hip_device_props;
  size_t total_used_mem, max_used_mem;

  HIPxxDevice();
  ~HIPxxDevice();

  void addKernel(HIPxxKernel* kernel);
  std::vector<HIPxxKernel*>& getKernels();

  /**
   * @brief Use a backend to populate device properties such as memory
   * available, frequencies, etc.
   */
  virtual void populateDeviceProperties() = 0;
  void copyDeviceProperties(hipDeviceProp_t* prop);

  /**
   * @brief Add a context to the vector of HIPxxContexts* to which this device
   * belongs to
   * @param ctx pointer to HIPxxContext object
   * @return true if added successfully
   * @return false if failed to add
   */
  bool addContext(HIPxxContext* ctx);

  HIPxxKernel* findKernelByHostPtr(const void* hostPtr);

  /**
   * @brief Get the default context object
   *
   * @return HIPxxContext* pointer to the 0th element in the internal
   * context array
   */
  HIPxxContext* getDefaultContext();
  virtual std::string getName() = 0;

  bool getModuleAndFName(const void* HostFunction, std::string& FunctionName,
                         HIPxxModule* hipxx_module);
  bool allocate(size_t bytes);
  bool free(size_t bytes);
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
  HIPxxContext();
  ~HIPxxContext();

  /**
   * @brief Add a device to this context
   *
   * @param dev pointer to HIPxxDevice object
   * @return true if device was added successfully
   * @return false upon failure
   */
  bool addDevice(HIPxxDevice* dev);
  void addQueue(HIPxxQueue* q);
  HIPxxQueue* getDefaultQueue();
  hipStream_t findQueue(hipStream_t stream);
  std::vector<HIPxxDevice*>& getDevices();
  std::vector<HIPxxQueue*>& getQueues();

  virtual void* allocate(size_t size) = 0;
  virtual hipError_t memCopy(void* dst, const void* src, size_t size,
                             hipStream_t stream) = 0;

  virtual bool registerFunctionAsKernel(std::string* module_str,
                                        const void* HostFunctionPtr,
                                        const char* FunctionName) = 0;
  hipError_t launchHostFunc(const void* HostFunction);
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
    logTrace("HIPxxDevice.add_device() {}", dev_in->getName());
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
    ex->setArg(arg, size, offset);

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
                                           const char* FunctionName) {
    logTrace("HIPxxBackend.register_function_as_kernel()");
    for (auto& ctx : hipxx_contexts) {
      ctx->registerFunctionAsKernel(module_str, HostFunctionPtr, FunctionName);
    }
    return true;
  }
};

/**
 * @brief Queue class for submitting kernels to for execution
 */
class HIPxxQueue {
 protected:
  std::mutex mtx;
  int priority;

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

  virtual std::string get_info();

  HIPxxDevice* get_device();
};

#endif
