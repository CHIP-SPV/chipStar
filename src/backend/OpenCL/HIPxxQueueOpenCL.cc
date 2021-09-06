#include "OpenCLBackend.hh"

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
}

HIPxxQueueOpenCL::~HIPxxQueueOpenCL() {
  delete cl_ctx;
  delete cl_dev;
}
