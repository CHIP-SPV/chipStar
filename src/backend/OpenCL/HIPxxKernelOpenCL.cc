#include "HIPxxBackendOpenCL.hh"

OCLFuncInfo* HIPxxKernelOpenCL::get_func_info() const {
  logWarn("HIPxxKernelOpenCL->getFuncInfo() not yet implemented");
  return FuncInfo;
  // return new OCLFuncInfo();
}
