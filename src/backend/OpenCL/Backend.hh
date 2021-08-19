/**
 * @file Backend.hh
 * @author Paulius Velesko (pvelesko@gmail.com)
 * @brief OpenCL backend for HIPxx. HIPxxBackendOpenCL class definition with
 * inheritance from HIPxxBackend. Subsequent virtual function overrides.
 * @version 0.1
 * @date 2021-08-19
 *
 * @copyright Copyright (c) 2021
 *
 */
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