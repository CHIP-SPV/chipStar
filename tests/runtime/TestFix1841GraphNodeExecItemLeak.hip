// Reproduces #1841: CHIPGraphNodeKernel::setParams leaks its exec item when
// node construction throws after creating it, and deletes it twice when
// replacing it throws in createExecItem.
//
// No public input fails deterministically after createExecItem, so this
// replaces the global operator new and fails one chosen allocation made by
// libCHIP. Exec items are recognised and counted by being allocated from
// createExecItem.

#include <hip/hip_runtime.h>

#include <dlfcn.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

enum Inject { None, AtExecItem, AfterExecItem };

static thread_local bool Track = false; // Classify this thread's allocations.
static thread_local Inject Pending = None;
static std::atomic<long> LiveExecItems{0};

static bool calledFrom(void *Ret, const char *Lib, const char *Sym) {
  Dl_info Info;
  return dladdr(Ret, &Info) && Info.dli_fname &&
         std::strstr(Info.dli_fname, Lib) &&
         (!Sym || (Info.dli_sname && std::strstr(Info.dli_sname, Sym)));
}

void *operator new(std::size_t Size) {
  bool IsExecItem = false;
  if (Track) {
    void *Ret = __builtin_return_address(0);
    Track = false; // No recursion through dladdr.
    IsExecItem = calledFrom(Ret, "libCHIP", "createExecItem");
    bool FromChip = IsExecItem || calledFrom(Ret, "libCHIP", nullptr);
    Track = true;
    if ((Pending == AtExecItem && IsExecItem) ||
        (Pending == AfterExecItem && FromChip && !IsExecItem &&
         LiveExecItems > 0)) {
      Pending = None;
      throw std::bad_alloc();
    }
  }
  // A 16-byte header keeps the result aligned and marks exec items.
  auto *Hdr = static_cast<std::size_t *>(std::malloc(Size + 16));
  if (!Hdr)
    throw std::bad_alloc();
  Hdr[0] = IsExecItem;
  LiveExecItems += IsExecItem;
  return Hdr + 2;
}
void operator delete(void *Ptr) noexcept {
  if (!Ptr)
    return;
  auto *Hdr = static_cast<std::size_t *>(Ptr) - 2;
  LiveExecItems -= Hdr[0];
  std::free(Hdr);
}
void operator delete(void *Ptr, std::size_t) noexcept { operator delete(Ptr); }

__global__ void k(int *Out) { *Out = 1; }
__global__ void warmUp(int *Out) { *Out = 2; }

// Runs F with one allocation failing; true if exactly that failure escaped.
template <typename Fn> static bool failOnce(Inject What, Fn F) {
  Pending = What;
  bool Threw = false;
  try {
    F();
  } catch (std::bad_alloc &) {
    Threw = true;
  }
  bool Fired = Pending == None;
  Pending = None;
  return Threw && Fired;
}

int main() {
  int *Out = nullptr;
  void *Args[] = {&Out};
  hipKernelNodeParams P = {};
  P.func = reinterpret_cast<void *>(warmUp);
  P.gridDim = dim3(1);
  P.blockDim = dim3(1);
  P.kernelParams = Args;

  // Warm up untracked on another kernel of the module, so that k's first exec
  // item still allocates its kernel handle after being created.
  hipGraph_t G;
  hipGraphNode_t N;
  if (hipGraphCreate(&G, 0) != hipSuccess ||
      hipGraphAddKernelNode(&N, G, nullptr, 0, &P) != hipSuccess ||
      hipGraphDestroy(G) != hipSuccess) {
    std::printf("FAIL: warm-up\n");
    return 1;
  }
  P.func = reinterpret_cast<void *>(k);
  Track = true;
  bool Ok = true;

  // Node construction throws after its exec item was created.
  (void)hipGraphCreate(&G, 0);
  if (!failOnce(AfterExecItem,
                [&] { (void)hipGraphAddKernelNode(&N, G, nullptr, 0, &P); })) {
    std::printf("FAIL: construction: injected failure did not escape\n");
    Ok = false;
  }
  (void)hipGraphDestroy(G);
  std::printf("construction: %ld exec items leaked\n", (long)LiveExecItems);
  Ok &= LiveExecItems == 0;

  // Replacing the exec item throws in createExecItem; the node keeps its old
  // one, or destroying the node would delete a destroyed item.
  (void)hipGraphCreate(&G, 0);
  (void)hipGraphAddKernelNode(&N, G, nullptr, 0, &P);
  long Before = LiveExecItems;
  if (!failOnce(AtExecItem,
                [&] { (void)hipGraphKernelNodeSetParams(N, &P); })) {
    std::printf("FAIL: setParams: injected failure did not escape\n");
    Ok = false;
  }
  std::printf("setParams: %ld exec items live, expected %ld\n",
              (long)LiveExecItems, Before);
  if (LiveExecItems != Before) {
    std::printf("FAIL\n");
    return 1;
  }
  (void)hipGraphDestroyNode(N); // hipGraphDestroy does not delete nodes.
  (void)hipGraphDestroy(G);
  Ok &= LiveExecItems == 0;
  Track = false;

  if (!Ok) {
    std::printf("FAIL\n");
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
