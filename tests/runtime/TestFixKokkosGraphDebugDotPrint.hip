// Reproduces: hipGraphDebugDotPrint returns hipErrorNotSupported and writes
// nothing. Kokkos hip.graph_instantiate_and_debug_dot_print expects a DOT
// file that names the kernel of each kernel node.
#include <hip/hip_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

__global__ void incrementKernel(int *Data) { ++*Data; }

#define CHECK(Call)                                                            \
  do {                                                                         \
    hipError_t Err = (Call);                                                   \
    if (Err != hipSuccess) {                                                   \
      std::cout << "FAIL: " #Call " -> " << hipGetErrorString(Err) << "\n";    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  int *Dev;
  CHECK(hipMalloc(&Dev, sizeof(int)));

  void *Args[] = {&Dev};
  hipKernelNodeParams Params = {};
  Params.func = (void *)incrementKernel;
  Params.gridDim = dim3(1);
  Params.blockDim = dim3(1);
  Params.sharedMemBytes = 0;
  Params.kernelParams = Args;
  Params.extra = nullptr;

  // root (empty) -> kernel
  hipGraph_t Graph;
  CHECK(hipGraphCreate(&Graph, 0));
  hipGraphNode_t Root, Kernel;
  CHECK(hipGraphAddEmptyNode(&Root, Graph, nullptr, 0));
  CHECK(hipGraphAddKernelNode(&Kernel, Graph, &Root, 1, &Params));

  const auto DotPath = std::filesystem::temp_directory_path() /
                       "TestFixKokkosGraphDebugDotPrint.dot";
  std::filesystem::remove(DotPath);

  CHECK(hipGraphDebugDotPrint(Graph, DotPath.c_str(),
                              hipGraphDebugDotFlagsVerbose));

  if (!std::filesystem::exists(DotPath)) {
    std::cout << "FAIL: " << DotPath << " was not written\n";
    return 2;
  }

  std::stringstream Buffer;
  Buffer << std::ifstream(DotPath).rdbuf();
  const std::string Dot = Buffer.str();
  std::filesystem::remove(DotPath);

  if (Dot.find("incrementKernel") == std::string::npos) {
    std::cout << "FAIL: kernel name missing from DOT output:\n" << Dot;
    return 3;
  }

  CHECK(hipGraphDestroy(Graph));
  CHECK(hipFree(Dev));
  std::cout << "PASSED\n";
  return 0;
}
