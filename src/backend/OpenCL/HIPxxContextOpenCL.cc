#include "HIPxxBackendOpenCL.hh"

HIPxxContextOpenCL::HIPxxContextOpenCL(cl::Context *ctx_in) {
  logDebug("HIPxxContextOpenCL Initialized via OpenCL Context pointer.");
  cl_ctx = ctx_in;
}

void *HIPxxContextOpenCL::allocate(size_t size) {
  logWarn("HIPxxContextOpenCL->allocate() not yet implemented");
  return (void *)0xDEADBEEF;
}