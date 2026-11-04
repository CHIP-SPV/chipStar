// Linked into TestFix2362CoarseMapHostPointer: reports coarse-only SVM and
// aborts on an SVM map/unmap of the watched range.
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <unistd.h>

extern "C" {
char *CoarseWatchBegin, *CoarseWatchEnd;
int CoarseCapsRewritten, CoarseMapCalls;
}

extern "C" int clGetDeviceInfo(void *Dev, unsigned Param, size_t Size,
                               void *Val, size_t *SizeRet) {
  static auto Real = (int (*)(void *, unsigned, size_t, void *, size_t *))dlsym(
      RTLD_NEXT, "clGetDeviceInfo");
  int Err = Real(Dev, Param, Size, Val, SizeRet);
  const unsigned long CL_DEVICE_SVM_CAPABILITIES = 0x1053;
  const unsigned long CoarseBuffer = 1; // CL_DEVICE_SVM_COARSE_GRAIN_BUFFER
  if (Err == 0 && Param == CL_DEVICE_SVM_CAPABILITIES && Val &&
      Size >= sizeof(unsigned long) && (*(unsigned long *)Val & CoarseBuffer)) {
    *(unsigned long *)Val = CoarseBuffer;
    CoarseCapsRewritten = 1;
  }
  return Err;
}

static void check(const char *Call, void *Ptr) {
  if ((char *)Ptr >= CoarseWatchBegin && (char *)Ptr < CoarseWatchEnd) {
    std::printf("FAIL: %s on the registered host pointer %p\n", Call, Ptr);
    std::fflush(stdout);
    _exit(1);
  }
  ++CoarseMapCalls;
}

extern "C" int clEnqueueSVMMap(void *Q, unsigned Blocking, unsigned long Flags,
                               void *Ptr, size_t Size, unsigned N,
                               const void *Wait, void *Ev) {
  static auto Real = (int (*)(void *, unsigned, unsigned long, void *, size_t,
                              unsigned, const void *, void *))dlsym(
      RTLD_NEXT, "clEnqueueSVMMap");
  check("clEnqueueSVMMap", Ptr);
  return Real(Q, Blocking, Flags, Ptr, Size, N, Wait, Ev);
}

extern "C" int clEnqueueSVMUnmap(void *Q, void *Ptr, unsigned N,
                                 const void *Wait, void *Ev) {
  static auto Real = (int (*)(void *, void *, unsigned, const void *, void *))
      dlsym(RTLD_NEXT, "clEnqueueSVMUnmap");
  check("clEnqueueSVMUnmap", Ptr);
  return Real(Q, Ptr, N, Wait, Ev);
}
