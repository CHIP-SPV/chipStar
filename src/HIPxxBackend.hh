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

#include <iostream>
#include <string>
#include <vector>

// Forward Declarations
class HIPxxDevice;

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
  bool add_device(HIPxxDevice* dev) {
    Devices.push_back(dev);
    // TODO check for success
    return true;
  }
};

/**
 * @brief Compute device class
 */
class HIPxxDevice {
 protected:
  /// Vector of contexts to which this device belongs to
  std::vector<HIPxxContext*> xxContexts;

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
  bool add_context(HIPxxContext* ctx) {
    xxContexts.push_back(ctx);
    // TODO check for success
    return true;
  }

  /**
   * @brief Get the default context object
   *
   * @return HIPxxContext* pointer to the 0th element in the internal context
   * array
   */
  HIPxxContext* get_default_context() {
    // TODO Check for initialization
    // if (xxContexts.size() == 0)
    return xxContexts.at(0);
  }
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

 public:
  HIPxxBackend() { std::cout << "HIPxxBackend Base Constructor\n"; };
  ~HIPxxBackend(){};
  virtual void initialize() = 0;
};

#endif
