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

#include "HIPxxDriver.hh"
#include "temporary.hh"

// Forward Declarations
class HIPxxDevice;
class HIPxxContext;
class HIPxxModule;
class HIPxxKernel;

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
  /// Kernel to be executed
  HIPxxKernel* Kernel;
  // TODO Args
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

  /// Vector of contexts to which this device belongs to
  std::vector<HIPxxContext*> xxContexts;
  /// Modules in binary representation
  std::vector<std::string*> ModulesStr;
  /// Modules in parsed representation
  std::vector<HIPxxModule*> Modules;

  /// Map host pointer to module in binary representation
  std::map<const void*, std::string*> HostPtrToModuleStrMap;
  /// Map host pointer to module in parsed representation
  std::map<const void*, HIPxxModule*> HostPtrToModuleMap;
  /// Map host pointer to a function name
  std::map<const void*, std::string> HostPtrToNameMap;

 public:
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
};

/**
 * @brief Queue class for submitting kernels to for execution
 */
class HIPxxQueue {
 protected:
  /// Device on which this queue will execute
  HIPxxDevice* xxDevice;
  /// Context to which device belongs to
  HIPxxContext* xxContext;

 public:
  HIPxxQueue(){};
  ~HIPxxQueue(){};

  /// Initialize this queue for a given device
  virtual void initialize(HIPxxDevice* dev) = 0;
  /// Submit a kernel for execution
  virtual void submit(HIPxxExecItem* exec_item) = 0;
};

/**
 * @brief Primary object to interact with the backend
 */
class HIPxxBackend {
 protected:
  std::vector<HIPxxContext*> xxContexts;
  std::vector<HIPxxQueue*> xxQueues;
  std::vector<HIPxxDevice*> xxDevices;
  /**
   * @brief Modules stored in binary representation.
   * During compilation each translation unit is parsed for functions that are
   * marked for execution on the device. These functions are then compiled to
   * device code and stored in binary representation.
   *  */
  std::vector<std::string*> ModulesStr;

 public:
  HIPxxBackend() { std::cout << "HIPxxBackend Base Constructor\n"; };
  ~HIPxxBackend(){};
  virtual void initialize() = 0;

  std::vector<HIPxxDevice*> get_devices() { return xxDevices; }
  size_t get_num_devices() { return xxDevices.size(); }
  std::vector<std::string*> get_modules_bin() { return ModulesStr; }
};

#endif
