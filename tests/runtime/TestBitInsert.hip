#include <hip/hip_runtime.h>

__global__ void bitInsert(unsigned *Dst, unsigned Src0, unsigned Src1,
                          unsigned Offset, unsigned Width) {
  *Dst = __bitinsert_u32(Src0, Src1, Offset, Width);
}

__global__ void bitInsert(uint64_t *Dst, uint64_t Src0, uint64_t Src1,
                          uint64_t Offset, uint64_t Width) {
  *Dst = __bitinsert_u64(Src0, Src1, Offset, Width);
}

int main() {
  unsigned *Dst1D, Dst1H = 0;
  uint64_t *Dst2D, Dst2H = 0;

  (void)hipMalloc(&Dst1D, sizeof(unsigned));
  (void)hipMalloc(&Dst2D, sizeof(uint64_t));

  (void)bitInsert<<<1, 1>>>(Dst1D, 0xffff4321u, 0xffff8765ul,
                            // Arguments + a garbage bit that should be ignored.
                            16u + 32u, 16u + 32u);
  hipMemcpy(&Dst1H, Dst1D, sizeof(unsigned), hipMemcpyDeviceToHost);
  if (Dst1H != 0x87654321)
    return 1;

  (void)bitInsert<<<1, 1>>>(Dst2D, 0xffffffff44332211ul, 0xffffffff88776655ul,
                            // Arguments + a garbage bit that should be ignored.
                            32u + 64u, 32u + 64u);
  (void)hipMemcpy(&Dst2H, Dst2D, sizeof(uint64_t), hipMemcpyDeviceToHost);
  if (Dst2H != 0x8877665544332211ul)
    return 2;

  (void)hipFree(Dst1D);
  (void)hipFree(Dst2D);
  return 0;
}
