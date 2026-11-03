// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2283:
// hipInitFromNativeHandles deleted the backend while graphs, graph execs and
// modules still pointed into it.
#include <hip/hip_interop.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Weak so this also links on a build without the OpenCL backend.
extern "C" __attribute__((weak)) int clRetainContext(void *Context);

#define CHECK(Cond, Msg)                                                       \
  do {                                                                         \
    if (!(Cond)) {                                                             \
      printf("FAIL: %s\n", Msg);                                               \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  int N = 0;
  hipGetBackendNativeHandles(0, nullptr, &N);
  std::vector<uintptr_t> H(N);
  hipGetBackendNativeHandles(0, H.data(), nullptr);

  hipGraph_t Graph;
  CHECK(hipGraphCreate(&Graph, 0) == hipSuccess, "hipGraphCreate");
  hipGraphNode_t Node;
  CHECK(hipGraphAddEmptyNode(&Node, Graph, nullptr, 0) == hipSuccess,
        "hipGraphAddEmptyNode");
  CHECK(hipInitFromNativeHandles(&H[1], 4) == hipErrorIllegalState,
        "reinit with a live graph");

  hipGraphExec_t Exec;
  CHECK(hipGraphInstantiate(&Exec, Graph, nullptr, nullptr, 0) == hipSuccess,
        "hipGraphInstantiate");
  CHECK(hipGraphDestroy(Graph) == hipSuccess, "hipGraphDestroy");
  CHECK(hipInitFromNativeHandles(&H[1], 4) == hipErrorIllegalState,
        "reinit with a live graph exec");
  CHECK(hipGraphExecDestroy(Exec) == hipSuccess, "hipGraphExecDestroy");

  std::string Path = __FILE__;
  Path = Path.substr(0, Path.rfind('/') + 1) + "inputs/Module.bin";
  hipModule_t Module;
  CHECK(hipModuleLoad(&Module, Path.c_str()) == hipSuccess, "hipModuleLoad");
  CHECK(hipInitFromNativeHandles(&H[1], 4) == hipErrorIllegalState,
        "reinit with a live module");
  hipFunction_t Fn;
  CHECK(hipModuleGetFunction(&Fn, Module, "add1") == hipSuccess,
        "module unusable after rejected reinit");
  CHECK(hipModuleUnload(Module) == hipSuccess, "hipModuleUnload");

  // Level Zero's old backend destroys its own context and queue, so a
  // backend's own handles cannot be reinitialized from there.
  if (strcmp(reinterpret_cast<const char *>(H[0]), "opencl")) {
    printf("PASS\n");
    return 0;
  }
  // The new backend adopts the context without retaining it.
  clRetainContext(reinterpret_cast<void *>(H[3]));
  CHECK(hipInitFromNativeHandles(&H[1], 4) == hipSuccess,
        "reinit after everything was destroyed");
  printf("PASS\n");
  return 0;
}
