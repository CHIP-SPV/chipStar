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
#include <vector>

class HIPxxContext {
 protected:
 public:
  HIPxxContext(){};
  ~HIPxxContext(){};
};

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

class HIPxxQueue {
 protected:
  HIPxxDevice* xxDevice;
  HIPxxContext* xxContext;

 public:
  HIPxxQueue(){};
  ~HIPxxQueue(){};
  virtual void initialize(HIPxxDevice* dev) = 0;
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
