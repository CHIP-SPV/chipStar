#include "HIPxxDriver.hh"

std::once_flag initialized;
HIPxxBackend* Backend;

void _initialize() {
  std::cout << "HIPxxDriver Initialize\n";
  // Get the current Backend Env Var
  Backend = new HIPxxBackendOpenCL();
};

void initialize() { std::call_once(initialized, &_initialize); }