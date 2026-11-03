// Reproduces #2303: hipMemset3D with pitch == width and height < ysize filled
// Width*Height*Depth contiguous bytes, ignoring the slice stride.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

#define CHECK(x)                                                               \
  if ((x) != hipSuccess) {                                                     \
    printf("FAIL: %s\n", #x);                                                  \
    return 1;                                                                  \
  }

int main() {
  hipPitchedPtr P;
  CHECK(hipMalloc3D(&P, make_hipExtent(128, 2, 4)));
  if (P.pitch != 128 || P.ysize != 2) {
    printf("FAIL: pitch %zu ysize %zu, test needs 128 and 2\n", P.pitch,
           P.ysize);
    return 1;
  }
  size_t Size = P.pitch * P.ysize * 4;
  CHECK(hipMemset(P.ptr, 0, Size));
  CHECK(hipMemset3D(P, 0x5A, make_hipExtent(128, 1, 4)));
  std::vector<unsigned char> H(Size);
  CHECK(hipMemcpy(H.data(), P.ptr, Size, hipMemcpyDeviceToHost));
  for (size_t O = 0; O < Size; ++O)
    if (H[O] != ((O % 256) < 128 ? 0x5A : 0)) {
      printf("FAIL: offset %zu is 0x%02x\n", O, H[O]);
      return 1;
    }
  CHECK(hipFree(P.ptr));
  printf("PASS\n");
  return 0;
}
