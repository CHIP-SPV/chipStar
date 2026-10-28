// Linked into TestFix2008ClSetEventCallbackErrorMap to override libOpenCL's
// clSetEventCallback: every call returns CL_OUT_OF_RESOURCES.
#include <cstdio>
#include <cstdlib>

extern "C" int clSetEventCallback(void *, int, void (*)(void *, int, void *),
                                  void *) {
  std::fprintf(stderr, "interposer: failing clSetEventCallback\n");
  setenv("CLSETEVENTCALLBACK_FAILED", "1", 1);
  return -5; // CL_OUT_OF_RESOURCES
}
