// Reproduces #2340: hipMemcpy3D steps destination slices by the extent height.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

#define CHECK(X)                                                               \
  do {                                                                         \
    hipError_t Err_ = (X);                                                     \
    if (Err_ != hipSuccess) {                                                  \
      printf("FAIL: %s returned %d\n", #X, (int)Err_);                         \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// Copies Ext of S into D and checks every byte of D against Ext.
static int pitchedCase(const char *Name, hipPitchedPtr S, hipPitchedPtr D,
                       hipExtent Ext) {
  size_t DBytes = D.pitch * D.ysize * 2;
  CHECK(hipMemset(D.ptr, 0, DBytes));
  hipMemcpy3DParms P = {};
  P.srcPtr = S;
  P.dstPtr = D;
  P.extent = Ext;
  P.kind = hipMemcpyDeviceToDevice;
  CHECK(hipMemcpy3D(&P));
  std::vector<unsigned char> H(DBytes);
  CHECK(hipMemcpy(H.data(), D.ptr, DBytes, hipMemcpyDeviceToHost));
  for (size_t Z = 0; Z < 2; Z++)
    for (size_t Y = 0; Y < D.ysize; Y++)
      for (size_t X = 0; X < D.pitch; X++) {
        size_t Off = Z * D.ysize * D.pitch + Y * D.pitch + X;
        unsigned char Want =
            (X < Ext.width && Y < Ext.height && Z < Ext.depth) ? 0x3C : 0;
        if (H[Off] != Want) {
          printf("FAIL: %s: D[%zu] (x=%zu y=%zu z=%zu) = 0x%02x, expected "
                 "0x%02x\n",
                 Name, Off, X, Y, Z, H[Off], Want);
          return 1;
        }
      }
  return 0;
}

// Host source into a 3D array with a partial height extent, read back whole.
static int arrayCase(const char *Name, size_t CopyWidth) {
  const size_t W = 16, Hgt = 3, Dp = 2;
  hipChannelFormatDesc Desc = hipCreateChannelDesc<unsigned int>();
  hipArray_t A;
  hipError_t Err = hipMalloc3DArray(&A, &Desc, make_hipExtent(W, Hgt, Dp), 0);
  if (Err == hipErrorNotSupported)
    return CHIP_SKIP_TEST;
  CHECK(Err);
  size_t Pitch = W * sizeof(unsigned int), Bytes = Pitch * Hgt * Dp;
  std::vector<unsigned char> Src(Bytes, 0x3C), Zero(Bytes, 0), H(Bytes, 0xAA);
  hipMemcpy3DParms P = {};
  P.srcPtr = make_hipPitchedPtr(Zero.data(), Pitch, W, Hgt);
  P.dstArray = A;
  P.extent = make_hipExtent(W, Hgt, Dp);
  P.kind = hipMemcpyHostToDevice;
  CHECK(hipMemcpy3D(&P));
  P = {};
  P.srcPtr = make_hipPitchedPtr(Src.data(), Pitch, W, Hgt);
  P.dstArray = A;
  P.extent = make_hipExtent(CopyWidth, 1, Dp);
  P.kind = hipMemcpyHostToDevice;
  CHECK(hipMemcpy3D(&P));
  P = {};
  P.srcArray = A;
  P.dstPtr = make_hipPitchedPtr(H.data(), Pitch, W, Hgt);
  P.extent = make_hipExtent(W, Hgt, Dp);
  P.kind = hipMemcpyDeviceToHost;
  CHECK(hipMemcpy3D(&P));
  for (size_t Z = 0; Z < Dp; Z++)
    for (size_t Y = 0; Y < Hgt; Y++)
      for (size_t X = 0; X < Pitch; X++) {
        size_t Off = Z * Hgt * Pitch + Y * Pitch + X;
        unsigned char Want =
            (X < CopyWidth * sizeof(unsigned int) && Y < 1) ? 0x3C : 0;
        if (H[Off] != Want) {
          printf("FAIL: %s: array byte %zu (x=%zu y=%zu z=%zu) = 0x%02x, "
                 "expected 0x%02x\n",
                 Name, Off, X, Y, Z, H[Off], Want);
          return 1;
        }
      }
  CHECK(hipFreeArray(A));
  return 0;
}

int main() {
  hipPitchedPtr S, D;
  CHECK(hipMalloc3D(&S, make_hipExtent(128, 2, 2)));
  CHECK(hipMalloc3D(&D, make_hipExtent(128, 3, 2)));
  CHECK(hipMemset(S.ptr, 0x3C, S.pitch * S.ysize * 2));

  int Fail = pitchedCase("narrow", S, D, make_hipExtent(127, 1, 2));
  if (S.pitch == D.pitch)
    Fail |= pitchedCase("full width", S, D, make_hipExtent(D.pitch, 1, 2));

  // Full width could take one flat copy, narrower needs a copy per row.
  int R = arrayCase("array full width", 16);
  if (R != CHIP_SKIP_TEST) {
    Fail |= R;
    Fail |= arrayCase("array narrow", 15);
  }
  if (Fail)
    return 1;
  printf("PASS\n");
  return 0;
}
