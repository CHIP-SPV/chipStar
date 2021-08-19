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
  std::vector<HIPxxContext*> xxContexts;

 public:
  HIPxxDevice(){};
  ~HIPxxDevice(){};
  virtual void initialize() = 0;
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

class HIPxxBackend {
 protected:
  HIPxxContext* xxContext;

 public:
  HIPxxBackend() { std::cout << "HIPxxBackend Base Constructor\n"; };
  ~HIPxxBackend(){};
  virtual void initialize() = 0;
};

#endif
