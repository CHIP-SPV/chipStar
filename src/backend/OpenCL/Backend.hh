#ifndef HIPXX_BACKEND_OPENCL_H
#define HIPXX_BACKEND_OPENCL_H

#include "../../HIPxxBackend.hh"
class HIPxxBackendOpenCL : public HIPxxBackend {
 public:
  void initialize() override {
    std::cout << "HIPxxBackendOpenCL Initialize\n";
  };
};

#endif