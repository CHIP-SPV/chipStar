// Reproduces #1852: hipGraphDestroy and hipGraphExecDestroy leak the graph's
// nodes and each kernel node's exec item.
//
// Replacing the global operator new/delete applies to the whole process, so
// counting live allocations sees what the runtime creates and frees.

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>

static std::atomic<long> LiveAllocs{0};

void *operator new(std::size_t Size) {
  void *Ptr = std::malloc(Size ? Size : 1);
  if (!Ptr)
    throw std::bad_alloc();
  ++LiveAllocs;
  return Ptr;
}
void operator delete(void *Ptr) noexcept {
  if (!Ptr)
    return;
  --LiveAllocs;
  std::free(Ptr);
}
void operator delete(void *Ptr, std::size_t) noexcept { operator delete(Ptr); }

__global__ void k(int *Out, int V) { *Out = V; }

#define CHECK(X)                                                               \
  if ((X) != hipSuccess) {                                                     \
    std::printf("FAIL: %s\n", #X);                                             \
    return false;                                                              \
  }

// One graph lifetime: build, clone, instantiate, launch, destroy. With
// GraphFirst the graph is destroyed before its executable graph is launched.
static bool round(int *Out, int V, bool GraphFirst) {
  void *Args[] = {&Out, &V};
  hipKernelNodeParams P = {};
  P.func = reinterpret_cast<void *>(k);
  P.gridDim = dim3(1);
  P.blockDim = dim3(1);
  P.kernelParams = Args;

  hipGraph_t Child, G, Clone;
  hipGraphNode_t ChildEmpty, KNode, Empty, Doomed, ChildNode;
  CHECK(hipGraphCreate(&Child, 0));
  CHECK(hipGraphAddEmptyNode(&ChildEmpty, Child, nullptr, 0));
  CHECK(hipGraphCreate(&G, 0));
  CHECK(hipGraphAddKernelNode(&KNode, G, nullptr, 0, &P));
  CHECK(hipGraphAddEmptyNode(&Empty, G, &KNode, 1));
  CHECK(hipGraphAddChildGraphNode(&ChildNode, G, &Empty, 1, Child));
  CHECK(hipGraphDestroy(Child));
  CHECK(hipGraphClone(&Clone, G));
  CHECK(hipGraphDestroy(Clone));

  hipGraphExec_t Exec;
  CHECK(hipGraphInstantiate(&Exec, G, nullptr, nullptr, 0));
  if (GraphFirst)
    CHECK(hipGraphDestroy(G));
  CHECK(hipGraphLaunch(Exec, 0));
  CHECK(hipStreamSynchronize(0));
  CHECK(hipGraphExecDestroy(Exec));
  if (!GraphFirst) {
    CHECK(hipGraphAddEmptyNode(&Doomed, G, &KNode, 1));
    CHECK(hipGraphDestroyNode(Doomed));
    CHECK(hipGraphDestroy(G));
  }

  int Got = 0;
  CHECK(hipMemcpy(&Got, Out, sizeof(int), hipMemcpyDeviceToHost));
  if (Got != V) {
    std::printf("FAIL: kernel wrote %d, expected %d\n", Got, V);
    return false;
  }
  return true;
}

int main() {
  int *Out = nullptr;
  if (hipMalloc(&Out, sizeof(int)) != hipSuccess) {
    std::printf("FAIL: hipMalloc\n");
    return 1;
  }
  for (int I = 0; I < 4; I++) // Warm up: module build, caches.
    if (!round(Out, I, false))
      return 1;

  constexpr int N = 64;
  long Before = LiveAllocs;
  for (int I = 0; I < N; I++)
    if (!round(Out, I + 100, false))
      return 1;
  long Leaked = LiveAllocs - Before;
  std::printf("%d graph rounds left %ld allocations\n", N, Leaked);
  std::fflush(stdout);
  if (Leaked >= N) {
    std::printf("FAIL: graphs leak\n");
    return 1;
  }
  // HIP allows destroying a graph while its executable graph lives on.
  if (!round(Out, 7, true))
    return 1;
  std::printf("PASS\n");
  return 0;
}
