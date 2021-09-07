#include "HIPxxBackendOpenCL.hh"

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

hipError_t HIPxxExecItemOpenCL::launch(HIPxxKernel *hipxx_kernel) {
  logTrace("HIPxxExecItemOpenCL->launch()");
  HIPxxQueueOpenCL *ocl_q = (HIPxxQueueOpenCL *)q;
  return (hipError_t)(ocl_q->launch(Kernel, this) == hipSuccess);
}

int HIPxxExecItemOpenCL::setup_all_args(HIPxxKernelOpenCL *kernel) {
  OCLFuncInfo *FuncInfo = kernel->get_func_info();
  size_t NumLocals = 0;
  for (size_t i = 0; i < FuncInfo->ArgTypeInfo.size(); ++i) {
    if (FuncInfo->ArgTypeInfo[i].space == OCLSpace::Local) ++NumLocals;
  }
  // there can only be one dynamic shared mem variable, per cuda spec
  assert(NumLocals <= 1);

  if ((OffsetsSizes.size() + NumLocals) != FuncInfo->ArgTypeInfo.size()) {
    logError("Some arguments are still unset\n");
    return CL_INVALID_VALUE;
  }

  if (OffsetsSizes.size() == 0) return CL_SUCCESS;

  std::sort(OffsetsSizes.begin(), OffsetsSizes.end());
  if ((std::get<0>(OffsetsSizes[0]) != 0) ||
      (std::get<1>(OffsetsSizes[0]) == 0)) {
    logError("Invalid offset/size\n");
    return CL_INVALID_VALUE;
  }

  // check args are set
  if (OffsetsSizes.size() > 1) {
    for (size_t i = 1; i < OffsetsSizes.size(); ++i) {
      if ((std::get<0>(OffsetsSizes[i]) == 0) ||
          (std::get<1>(OffsetsSizes[i]) == 0) ||
          ((std::get<0>(OffsetsSizes[i - 1]) +
            std::get<1>(OffsetsSizes[i - 1])) > std::get<0>(OffsetsSizes[i]))) {
        logError("Invalid offset/size\n");
        return CL_INVALID_VALUE;
      }
    }
  }

  const unsigned char *start = ArgData.data();
  void *p;
  int err;
  for (cl_uint i = 0; i < OffsetsSizes.size(); ++i) {
    OCLArgTypeInfo &ai = FuncInfo->ArgTypeInfo[i];
    logDebug("ARG {}: OS[0]: {} OS[1]: {} \n      TYPE {} SPAC {} SIZE {}\n", i,
             std::get<0>(OffsetsSizes[i]), std::get<1>(OffsetsSizes[i]),
             (unsigned)ai.type, (unsigned)ai.space, ai.size);

    if (ai.type == OCLType::Pointer) {
      // TODO other than global AS ?
      assert(ai.size == sizeof(void *));
      assert(std::get<1>(OffsetsSizes[i]) == ai.size);
      p = *(void **)(start + std::get<0>(OffsetsSizes[i]));
      logDebug("setArg SVM {} to {}\n", i, p);
      err = ::clSetKernelArgSVMPointer(kernel->get().get(), i, p);
      if (err != CL_SUCCESS) {
        logDebug("clSetKernelArgSVMPointer failed with error {}\n", err);
        return err;
      }
    } else {
      size_t size = std::get<1>(OffsetsSizes[i]);
      size_t offs = std::get<0>(OffsetsSizes[i]);
      void *value = (void *)(start + offs);
      logDebug("setArg {} size {} offs {}\n", i, size, offs);
      err = ::clSetKernelArg(kernel->get().get(), i, size, value);
      if (err != CL_SUCCESS) {
        logDebug("clSetKernelArg failed with error {}\n", err);
        return err;
      }
    }
  }

  return setLocalSize(SharedMem, FuncInfo, kernel->get().get());
}
