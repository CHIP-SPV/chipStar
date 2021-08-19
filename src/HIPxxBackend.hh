
#ifndef HIPXX_BACKEND_H
#define HIPXX_BACKEND_H

#include <iostream>

class HIPxxBackend {
 public:
  HIPxxBackend() { std::cout << "HIPxxBackend Base Constructor\n"; }
  virtual void initialize() = 0;
};

#endif
