#include "HIPxxBackendOpenCL.hh"

hipError_t HIPxxQueueOpenCL::launch(HIPxxExecItem *exec_item) {
  // std::lock_guard<std::mutex> Lock(mtx);
  logTrace("HIPxxQueueOpenCL->launch()");
  HIPxxExecItemOpenCL *hipxx_ocl_exec_item = (HIPxxExecItemOpenCL *)exec_item;
  HIPxxKernelOpenCL *kernel =
      (HIPxxKernelOpenCL *)hipxx_ocl_exec_item->hipxx_kernel;
  assert(kernel != nullptr);
  logTrace("Launching Kernel {}", kernel->get_name());

  if (hipxx_ocl_exec_item->setup_all_args(kernel) != CL_SUCCESS) {
    logError("Failed to set kernel arguments for launch! \n");
    return hipErrorLaunchFailure;
  }

  dim3 GridDim = hipxx_ocl_exec_item->grid_dim;
  dim3 BlockDim = hipxx_ocl_exec_item->block_dim;

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
  // delete hipxx_ocl_exec_item;
  return retval;
}

HIPxxQueueOpenCL::HIPxxQueueOpenCL(HIPxxContextOpenCL *_ctx,
                                   HIPxxDeviceOpenCL *_dev) {
  logDebug("HIPxxQueueOpenCL Initialized via context, device pointers");
  cl_ctx = _ctx->cl_ctx;
  cl_dev = _dev->cl_dev;
  cl_q = new cl::CommandQueue(*cl_ctx, *cl_dev);
  hipxx_device = _dev;
  hipxx_context = _ctx;
  hipxx_context->addQueue(this);
}

HIPxxQueueOpenCL::~HIPxxQueueOpenCL() {
  delete cl_ctx;
  delete cl_dev;
}

hipError_t HIPxxQueueOpenCL::memCopy(void *dst, const void *src, size_t size) {
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

// HIPxxQueueOpenCL(HIPxxContextOpenCL *_ctx, HIPxxDeviceOpenCL *_dev) {
//   std::cout << "HIPxxQueueOpenCL Initialized via context, device
//   pointers\n"; cl_ctx = _ctx->cl_ctx; cl_dev = _dev->cl_dev; cl_q = new
//   cl::CommandQueue(*cl_ctx, *cl_dev);
// };
