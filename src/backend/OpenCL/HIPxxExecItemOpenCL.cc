#include "HIPxxBackendOpenCL.hh"

static int setLocalSize(size_t shared, OCLFuncInfo *FuncInfo,
                        cl_kernel kernel) {
  logWarn("setLocalSize");
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
  return hipSuccess;
  // return (hipError_t)(ocl_q->launch(Kernel, this) == hipSuccess);
}

int HIPxxExecItemOpenCL::setup_all_args(HIPxxKernelOpenCL *kernel) {
  OCLFuncInfo *FuncInfo = kernel->get_func_info();
  size_t NumLocals = 0;
  for (size_t i = 0; i < FuncInfo->ArgTypeInfo.size(); ++i) {
    if (FuncInfo->ArgTypeInfo[i].space == OCLSpace::Local) ++NumLocals;
  }
  // there can only be one dynamic shared mem variable, per cuda spec
  assert(NumLocals <= 1);

  if ((offset_sizes.size() + NumLocals) != FuncInfo->ArgTypeInfo.size()) {
    logError("Some arguments are still unset\n");
    return CL_INVALID_VALUE;
  }

  if (offset_sizes.size() == 0) return CL_SUCCESS;

  std::sort(offset_sizes.begin(), offset_sizes.end());
  if ((std::get<0>(offset_sizes[0]) != 0) ||
      (std::get<1>(offset_sizes[0]) == 0)) {
    logError("Invalid offset/size\n");
    return CL_INVALID_VALUE;
  }

  // check args are set
  if (offset_sizes.size() > 1) {
    for (size_t i = 1; i < offset_sizes.size(); ++i) {
      if ((std::get<0>(offset_sizes[i]) == 0) ||
          (std::get<1>(offset_sizes[i]) == 0) ||
          ((std::get<0>(offset_sizes[i - 1]) +
            std::get<1>(offset_sizes[i - 1])) > std::get<0>(offset_sizes[i]))) {
        logError("Invalid offset/size\n");
        return CL_INVALID_VALUE;
      }
    }
  }

  const unsigned char *start = arg_data.data();
  void *p;
  int err;
  for (cl_uint i = 0; i < offset_sizes.size(); ++i) {
    OCLArgTypeInfo &ai = FuncInfo->ArgTypeInfo[i];
    logDebug("ARG {}: OS[0]: {} OS[1]: {} \n      TYPE {} SPAC {} SIZE {}\n", i,
             std::get<0>(offset_sizes[i]), std::get<1>(offset_sizes[i]),
             (unsigned)ai.type, (unsigned)ai.space, ai.size);

    if (ai.type == OCLType::Pointer) {
      // TODO other than global AS ?
      assert(ai.size == sizeof(void *));
      assert(std::get<1>(offset_sizes[i]) == ai.size);
      p = *(void **)(start + std::get<0>(offset_sizes[i]));
      logDebug("setArg SVM {} to {}\n", i, p);
      err = ::clSetKernelArgSVMPointer(kernel->get().get(), i, p);
      if (err != CL_SUCCESS) {
        logDebug("clSetKernelArgSVMPointer failed with error {}\n", err);
        return err;
      }
    } else {
      size_t size = std::get<1>(offset_sizes[i]);
      size_t offs = std::get<0>(offset_sizes[i]);
      void *value = (void *)(start + offs);
      logDebug("setArg {} size {} offs {}\n", i, size, offs);
      err = ::clSetKernelArg(kernel->get().get(), i, size, value);
      if (err != CL_SUCCESS) {
        logDebug("clSetKernelArg failed with error {}\n", err);
        return err;
      }
    }
  }

  return setLocalSize(shared_mem, FuncInfo, kernel->get().get());
}
