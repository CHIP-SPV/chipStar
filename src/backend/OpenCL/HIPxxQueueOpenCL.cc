#include "HIPxxBackendOpenCL.hh"

hipError_t HIPxxQueueOpenCL::launch(HIPxxKernel *kernel,
                                    HIPxxExecItem *exec_item) {
  std::lock_guard<std::mutex> Lock(mtx);
  logTrace("HIPxxQueueOpenCL->launch()");
  HIPxxExecItemOpenCL *hipxx_ocl_exec_item = (HIPxxExecItemOpenCL *)exec_item;
  HIPxxKernelOpenCL *hipxx_opencl_kernel = (HIPxxKernelOpenCL *)kernel;
  //_e->run();

  if (hipxx_ocl_exec_item->setup_all_args(hipxx_opencl_kernel) != CL_SUCCESS) {
    logError("Failed to set kernel arguments for launch! \n");
    return hipErrorLaunchFailure;
  }
  return hipSuccess;
}

HIPxxQueueOpenCL::HIPxxQueueOpenCL(HIPxxContextOpenCL *_ctx,
                                   HIPxxDeviceOpenCL *_dev) {
  logDebug("HIPxxQueueOpenCL Initialized via context, device pointers");
  cl_ctx = _ctx->cl_ctx;
  cl_dev = _dev->cl_dev;
  cl_q = new cl::CommandQueue(*cl_ctx, *cl_dev);
  hipxx_device = _dev;
  hipxx_context = _ctx;
  hipxx_context->add_queue(this);
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
