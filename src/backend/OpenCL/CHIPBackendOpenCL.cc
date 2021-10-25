#include "CHIPBackendOpenCL.hh"

void CHIPModuleOpenCL::compile(CHIPDevice *chip_dev_) {
  CHIPDeviceOpenCL *chip_dev_ocl = (CHIPDeviceOpenCL *)chip_dev_;
  CHIPContextOpenCL *chip_ctx_ocl =
      (CHIPContextOpenCL *)(chip_dev_ocl->getContext());

  int err;
  std::vector<char> binary_vec(src.begin(), src.end());
  auto Program = cl::Program(*(chip_ctx_ocl->get()), binary_vec, false, &err);
  if (err != CL_SUCCESS) {
    logError("CreateProgramWithIL Failed: {}\n", err);
    std::abort();
  }

  //   for (CHIPDevice *chip_dev : chip_devices) {
  std::string name = chip_dev_ocl->getName();
  int build_failed = Program.build("-x spir -cl-kernel-arg-info");

  std::string log =
      Program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*chip_dev_ocl->cl_dev, &err);
  if (err != CL_SUCCESS) {
    logError("clGetProgramBuildInfo() Failed:{}\n", err);
    std::abort();
  }
  logDebug("Program BUILD LOG for device #{}:{}:\n{}\n",
           chip_dev_ocl->getDeviceId(), name, log);
  if (build_failed != CL_SUCCESS) {
    logError("clBuildProgram() Failed: {}\n", build_failed);
    std::abort();
  }

  std::vector<cl::Kernel> kernels;
  err = Program.createKernels(&kernels);
  if (err != CL_SUCCESS) {
    logError("clCreateKernels() Failed: {}\n", err);
    std::abort();
  }
  logDebug("Kernels in CHIPModuleOpenCL: {} \n", kernels.size());
  for (int kernel_idx = 0; kernel_idx < kernels.size(); kernel_idx++) {
    CHIPKernelOpenCL *chip_kernel =
        new CHIPKernelOpenCL(std::move(kernels[kernel_idx]), func_infos);
    chip_kernels.push_back(chip_kernel);
  }
}

CHIPContextOpenCL::CHIPContextOpenCL(cl::Context *ctx_in) {
  logDebug("CHIPContextOpenCL Initialized via OpenCL Context pointer.");
  cl_ctx = ctx_in;
}

void *CHIPContextOpenCL::allocate_(size_t size, size_t alignment,
                                   CHIPMemoryType mem_type) {
  void *retval;

  retval = svm_memory.allocate(*cl_ctx, size);
  return retval;
}

hipError_t CHIPContextOpenCL::memCopy(void *dst, const void *src, size_t size,
                                      hipStream_t stream) {
  logWarn("CHIPContextOpenCL::memCopy not implemented");
  // FIND_QUEUE_LOCKED(stream);
  std::lock_guard<std::mutex> Lock(mtx);
  CHIPQueue *Queue = findQueue(stream);
  if (Queue == nullptr) return hipErrorInvalidResourceHandle;

  if (svm_memory.hasPointer(dst) || svm_memory.hasPointer(src))
    return Queue->memCopy(dst, src, size);
  else
    return hipErrorInvalidDevicePointer;
}

#include "CHIPBackendOpenCL.hh"

CHIPDeviceOpenCL::CHIPDeviceOpenCL(CHIPContextOpenCL *chip_ctx,
                                   cl::Device *dev_in, int idx) {
  logDebug(
      "CHIPDeviceOpenCL initialized via OpenCL device pointer and context "
      "pointer");
  cl_dev = dev_in;
  cl_ctx = chip_ctx->cl_ctx;
  idx = idx;

  chip_ctx->addDevice(this);
  ctx = chip_ctx;
}

void CHIPDeviceOpenCL::populateDeviceProperties_() {
  logTrace("CHIPDeviceOpenCL->populate_device_properties()");
  cl_int err;
  std::string Temp;

  assert(cl_dev != nullptr);
  Temp = cl_dev->getInfo<CL_DEVICE_NAME>();
  strncpy(hip_device_props.name, Temp.c_str(), 255);
  hip_device_props.name[255] = 0;

  hip_device_props.totalGlobalMem =
      cl_dev->getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>(&err);

  hip_device_props.sharedMemPerBlock =
      cl_dev->getInfo<CL_DEVICE_LOCAL_MEM_SIZE>(&err);

  hip_device_props.maxThreadsPerBlock =
      cl_dev->getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&err);

  std::vector<size_t> wi = cl_dev->getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

  hip_device_props.maxThreadsDim[0] = wi[0];
  hip_device_props.maxThreadsDim[1] = wi[1];
  hip_device_props.maxThreadsDim[2] = wi[2];

  // Maximum configured clock frequency of the device in MHz.
  hip_device_props.clockRate =
      1000 * cl_dev->getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();

  hip_device_props.multiProcessorCount =
      cl_dev->getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  hip_device_props.l2CacheSize =
      cl_dev->getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();

  // not actually correct
  hip_device_props.totalConstMem =
      cl_dev->getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();

  // totally made up
  hip_device_props.regsPerBlock = 64;

  // The minimum subgroup size on an intel GPU
  if (cl_dev->getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) {
    std::vector<uint> sg = cl_dev->getInfo<CL_DEVICE_SUB_GROUP_SIZES_INTEL>();
    if (sg.begin() != sg.end())
      hip_device_props.warpSize = *std::min_element(sg.begin(), sg.end());
  }
  hip_device_props.maxGridSize[0] = hip_device_props.maxGridSize[1] =
      hip_device_props.maxGridSize[2] = 65536;
  hip_device_props.memoryClockRate = 1000;
  hip_device_props.memoryBusWidth = 256;
  hip_device_props.major = 2;
  hip_device_props.minor = 0;

  hip_device_props.maxThreadsPerMultiProcessor = 10;

  hip_device_props.computeMode = 0;
  hip_device_props.arch = {};

  Temp = cl_dev->getInfo<CL_DEVICE_EXTENSIONS>();
  if (Temp.find("cl_khr_global_int32_base_atomics") != std::string::npos)
    hip_device_props.arch.hasGlobalInt32Atomics = 1;
  else
    hip_device_props.arch.hasGlobalInt32Atomics = 0;

  if (Temp.find("cl_khr_local_int32_base_atomics") != std::string::npos)
    hip_device_props.arch.hasSharedInt32Atomics = 1;
  else
    hip_device_props.arch.hasSharedInt32Atomics = 0;

  if (Temp.find("cl_khr_int64_base_atomics") != std::string::npos) {
    hip_device_props.arch.hasGlobalInt64Atomics = 1;
    hip_device_props.arch.hasSharedInt64Atomics = 1;
  } else {
    hip_device_props.arch.hasGlobalInt64Atomics = 1;
    hip_device_props.arch.hasSharedInt64Atomics = 1;
  }

  if (Temp.find("cl_khr_fp64") != std::string::npos)
    hip_device_props.arch.hasDoubles = 1;
  else
    hip_device_props.arch.hasDoubles = 0;

  hip_device_props.clockInstructionRate = 2465;
  hip_device_props.concurrentKernels = 1;
  hip_device_props.pciDomainID = 0;
  hip_device_props.pciBusID = 0x10;
  hip_device_props.pciDeviceID = 0x40 + idx;
  hip_device_props.isMultiGpuBoard = 0;
  hip_device_props.canMapHostMemory = 1;
  hip_device_props.gcnArch = 0;
  hip_device_props.integrated = 0;
  hip_device_props.maxSharedMemoryPerMultiProcessor = 0;
}

std::string CHIPDeviceOpenCL::getName() {
  if (cl_dev == nullptr) {
    logCritical("CHIPDeviceOpenCL.get_name() called on uninitialized ptr");
    std::abort();
  }
  return std::string(cl_dev->getInfo<CL_DEVICE_NAME>());
}

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

int CHIPExecItemOpenCL::setup_all_args(CHIPKernelOpenCL *kernel) {
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

hipError_t CHIPQueueOpenCL::launch(CHIPExecItem *exec_item) {
  // std::lock_guard<std::mutex> Lock(mtx);
  logTrace("CHIPQueueOpenCL->launch()");
  CHIPExecItemOpenCL *chip_ocl_exec_item = (CHIPExecItemOpenCL *)exec_item;
  CHIPKernelOpenCL *kernel =
      (CHIPKernelOpenCL *)chip_ocl_exec_item->getKernel();
  assert(kernel != nullptr);
  logTrace("Launching Kernel {}", kernel->get_name());

  if (chip_ocl_exec_item->setup_all_args(kernel) != CL_SUCCESS) {
    logError("Failed to set kernel arguments for launch! \n");
    return hipErrorLaunchFailure;
  }

  dim3 GridDim = chip_ocl_exec_item->getGrid();
  dim3 BlockDim = chip_ocl_exec_item->getBlock();

  const cl::NDRange global(GridDim.x * BlockDim.x, GridDim.y * BlockDim.y,
                           GridDim.z * BlockDim.z);
  const cl::NDRange local(BlockDim.x, BlockDim.y, BlockDim.z);

  cl::Event ev;
  int err = cl_q->enqueueNDRangeKernel(kernel->get(), cl::NullRange, global,
                                       local, nullptr, &ev);

  if (err != CL_SUCCESS)
    logError("clEnqueueNDRangeKernel() failed with: {}\n", err);
  hipError_t retval = (err == CL_SUCCESS) ? hipSuccess : hipErrorLaunchFailure;

  // TODO
  // cl_event LastEvent;
  // if (retval == hipSuccess) {
  //   if (LastEvent != nullptr) {
  //     logDebug("Launch: LastEvent == {}, will be: {}", (void *)LastEvent,
  //              (void *)ev.get());
  //     clReleaseEvent(LastEvent);
  //   } else
  //     logDebug("launch: LastEvent == NULL, will be: {}\n", (void *)ev.get());
  //   LastEvent = ev.get();
  //   clRetainEvent(LastEvent);
  // }

  // TODO remove this
  // delete chip_ocl_exec_item;
  return retval;
}

CHIPQueueOpenCL::CHIPQueueOpenCL(CHIPDevice *chip_device_)
    : CHIPQueue(chip_device_) {
  cl_ctx = ((CHIPContextOpenCL *)chip_context)->get();
  cl_dev = ((CHIPDeviceOpenCL *)chip_device)->get();

  cl_q = new cl::CommandQueue(*cl_ctx, *cl_dev);
}

CHIPQueueOpenCL::~CHIPQueueOpenCL() {
  delete cl_ctx;
  delete cl_dev;
}

hipError_t CHIPQueueOpenCL::memCopy(void *dst, const void *src, size_t size) {
  std::lock_guard<std::mutex> Lock(mtx);
  logDebug("clSVMmemcpy {} -> {} / {} B\n", src, dst, size);
  cl_event ev = nullptr;
  auto LastEvent = ev;
  int retval = ::clEnqueueSVMMemcpy(cl_q->get(), CL_FALSE, dst, src, size, 0,
                                    nullptr, &ev);
  if (retval == CL_SUCCESS) {
    // TODO
    if (LastEvent != nullptr) {
      logDebug("memCopy: LastEvent == {}, will be: {}", (void *)LastEvent,
               (void *)ev);
      clReleaseEvent(LastEvent);
    } else
      logDebug("memCopy: LastEvent == NULL, will be: {}\n", (void *)ev);
    LastEvent = ev;
  } else {
    logError("clEnqueueSVMMemCopy() failed with error {}\n", retval);
  }
  return (retval == CL_SUCCESS) ? hipSuccess : hipErrorLaunchFailure;
}

hipError_t CHIPQueueOpenCL::memCopyAsync(void *dst, const void *src,
                                         size_t size) {
  logCritical("CHIPQueueOpenCL::memCopyAsync");
  std::abort();
}

void CHIPQueueOpenCL::finish() {
  logCritical("CHIPQueueOpenCL::finish() not yet implemented");
  std::abort();
}

void CHIPDeviceOpenCL::reset() {
  logCritical("CHIPDeviceOpenCL::reset() not yet implemented");
  std::abort();
}

// CHIPQueueOpenCL(CHIPContextOpenCL *_ctx, CHIPDeviceOpenCL *_dev) {
//   std::cout << "CHIPQueueOpenCL Initialized via context, device
//   pointers\n"; cl_ctx = _ctx->cl_ctx; cl_dev = _dev->cl_dev; cl_q = new
//   cl::CommandQueue(*cl_ctx, *cl_dev);
// };
