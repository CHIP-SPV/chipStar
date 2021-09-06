#include "HIPxxBackendOpenCL.hh"

HIPxxContextOpenCL::HIPxxContextOpenCL(cl::Context *ctx_in) {
  logDebug("HIPxxContextOpenCL Initialized via OpenCL Context pointer.");
  cl_ctx = ctx_in;
}

void *HIPxxContextOpenCL::allocate(size_t size) {
  std::lock_guard<std::mutex> Lock(mtx);
  void *retval;

  for (auto dev : hipxx_devices) {
    if (!dev->reserve_mem(size)) return nullptr;
    retval = svm_memory.allocate(*cl_ctx, size);
    if (retval == nullptr) dev->release_mem(size);
  }

  return retval;
}

hipError_t HIPxxContextOpenCL::memCopy(void *dst, const void *src, size_t size,
                                       hipStream_t stream) {
  logWarn("HIPxxContextOpenCL::memCopy not implemented");
  // FIND_QUEUE_LOCKED(stream);
  std::lock_guard<std::mutex> Lock(mtx);
  HIPxxQueue *Queue = findQueue(stream);
  if (Queue == nullptr) return hipErrorInvalidResourceHandle;

  if (svm_memory.hasPointer(dst) || svm_memory.hasPointer(src))
    return Queue->memCopy(dst, src, size);
  else
    return hipErrorInvalidDevicePointer;
}