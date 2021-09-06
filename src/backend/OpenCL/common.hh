#ifndef OCL_BE_COMMON_H
#define OCL_BE_COMMON_H

#include <vector>
#include <map>
#include "../../HIPxxBackend.hh"

enum class OCLType : unsigned { POD = 0, Pointer = 1, Image = 2, Sampler = 3 };

enum class OCLSpace : unsigned {
  Private = 0,
  Global = 1,
  Constant = 2,
  Local = 3,
  Unknown = 1000
};

struct OCLArgTypeInfo {
  OCLType type;
  OCLSpace space;
  size_t size;
};

struct OCLFuncInfo {
  std::vector<OCLArgTypeInfo> ArgTypeInfo;
  OCLArgTypeInfo retTypeInfo;
};

typedef std::map<int32_t, OCLFuncInfo *> OCLFuncInfoMap;

typedef std::map<std::string, OCLFuncInfo *> OpenCLFunctionInfoMap;

static int setLocalSize(size_t shared, OCLFuncInfo *FuncInfo,
                        cl_kernel kernel) {
  logWarn("setLocalSize not yet implemented");
  int err = CL_SUCCESS;

  if (shared > 0) {
    logDebug("setLocalMemSize to {}\n", shared);
    size_t LastArgIdx = FuncInfo->ArgTypeInfo.size() - 1;
    if (FuncInfo->ArgTypeInfo[LastArgIdx].space != OCLSpace::Local) {
      // this can happen if for example the llvm optimizes away
      // the dynamic local variable
      logWarn(
          "Can't set the dynamic local size, "
          "because the kernel doesn't use any local memory.\n");
    } else {
      err = ::clSetKernelArg(kernel, LastArgIdx, shared, nullptr);
      if (err != CL_SUCCESS) {
        logError("clSetKernelArg() failed to set dynamic local size!\n");
      }
    }
  }

  return err;
}

#endif