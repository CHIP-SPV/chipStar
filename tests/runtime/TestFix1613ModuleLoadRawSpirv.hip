// hipModuleLoadData must reject a raw SPIR-V image, not scan past its end (#1613).
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <sys/mman.h>
#include <unistd.h>

int main() {
  int N = 0; // a failed load must come from the image, not a missing device
  if (hipGetDeviceCount(&N) != hipSuccess || N < 1)
    return CHIP_SKIP_TEST;

  long Page = sysconf(_SC_PAGESIZE);
  char *Map = (char *)mmap(nullptr, 2 * Page, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (Map == MAP_FAILED || mprotect(Map + Page, Page, PROT_NONE) != 0)
    return 1;

  // SPIR-V header only, ending exactly at the guard page.
  uint32_t *Img = (uint32_t *)(Map + Page - 5 * sizeof(uint32_t));
  Img[0] = 0x07230203;
  Img[1] = 0x00010000;
  Img[2] = Img[3] = Img[4] = 0;

  hipModule_t Mod = nullptr;
  hipError_t Err = hipModuleLoadData(&Mod, Img);
  if (Err == hipSuccess) {
    printf("FAIL: raw SPIR-V image was accepted\n");
    return 1;
  }
  printf("PASSED (%s)\n", hipGetErrorString(Err));
  return 0;
}
