// LD_PRELOAD: fail the first clFinish call with CL_OUT_OF_RESOURCES.

#include <dlfcn.h>

#include <cstdio>

extern "C" int clFinish(void *CommandQueue) {
  static auto RealClFinish =
      reinterpret_cast<int (*)(void *)>(dlsym(RTLD_NEXT, "clFinish"));
  static int Calls = 0;
  int Status = RealClFinish ? RealClFinish(CommandQueue) : -5;
  if (__atomic_add_fetch(&Calls, 1, __ATOMIC_SEQ_CST) == 1) {
    std::fprintf(stderr, "clfinish-interposer: failing call 1\n");
    return -5; // CL_OUT_OF_RESOURCES
  }
  return Status;
}
