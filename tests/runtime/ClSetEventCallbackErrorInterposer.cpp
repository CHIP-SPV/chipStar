// LD_PRELOAD interposer for TestFix2008ClSetEventCallbackErrorMap: the first
// clSetEventCallback call returns CL_OUT_OF_RESOURCES, the rest are chained.
#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>

extern "C" int clSetEventCallback(void *Event, int Type,
                                  void (*Fn)(void *, int, void *),
                                  void *UserData) {
  using Fn_t = int (*)(void *, int, void (*)(void *, int, void *), void *);
  static auto Real = reinterpret_cast<Fn_t>(dlsym(RTLD_NEXT, "clSetEventCallback"));
  static bool Failed = false;
  if (!Failed || !Real) {
    Failed = true;
    std::fprintf(stderr, "interposer: failing clSetEventCallback\n");
    setenv("CLSETEVENTCALLBACK_FAILED", "1", 1);
    return -5; // CL_OUT_OF_RESOURCES
  }
  return Real(Event, Type, Fn, UserData);
}
