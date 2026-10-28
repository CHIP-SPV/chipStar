// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2164: a module
// whose compile throws must not leak the VkPipelineCache it created. Counts
// vkCreatePipelineCache/vkDestroyPipelineCache through interposers defined in
// this executable and checks they balance across a failed module compile.
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdio>
#include <dlfcn.h>

static std::atomic<int> Created{0}, Destroyed{0};

extern "C" int vkCreatePipelineCache(void *Dev, const void *Info,
                                     const void *Alloc, void **Out) {
  static auto Real = (int (*)(void *, const void *, const void *, void **))
      dlsym(RTLD_NEXT, "vkCreatePipelineCache");
  int R = Real(Dev, Info, Alloc, Out);
  if (R == 0)
    Created++;
  return R;
}

extern "C" void vkDestroyPipelineCache(void *Dev, void *Cache,
                                       const void *Alloc) {
  static auto Real = (void (*)(void *, void *, const void *))dlsym(
      RTLD_NEXT, "vkDestroyPipelineCache");
  if (Cache)
    Destroyed++;
  Real(Dev, Cache, Alloc);
}

extern __device__ void undefinedFn(int);
__global__ void kernel(int X) { undefinedFn(X); }

int main() {
  (void)hipFree(nullptr); // Device init creates its own long-lived cache.
  int C0 = Created, D0 = Destroyed;
  kernel<<<1, 1>>>(0);
  if (hipDeviceSynchronize() == hipSuccess && hipGetLastError() == hipSuccess) {
    printf("FAIL: launch of a kernel calling an undefined function succeeded\n");
    return 1;
  }
  if (Created == C0) {
    printf("HIP_SKIP_THIS_TEST: no Vulkan pipeline cache was created\n");
    return CHIP_SKIP_TEST;
  }
  if (Created - C0 != Destroyed - D0) {
    printf("FAIL: created %d pipeline caches, destroyed %d\n", Created - C0,
           Destroyed - D0);
    return 1;
  }
  printf("PASS\n");
  return 0;
}
