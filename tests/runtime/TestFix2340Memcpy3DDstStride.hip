// Reproduces #2340: hipMemcpy3D steps destination slices by the extent height.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
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

// A layered array may have height 0, which allocates one row per layer.
static int layeredCase() {
  const size_t W = 16, Dp = 2, Pitch = W * sizeof(unsigned int);
  hipChannelFormatDesc Desc = hipCreateChannelDesc<unsigned int>();
  hipArray_t A;
  CHECK(hipMalloc3DArray(&A, &Desc, make_hipExtent(W, 0, Dp), hipArrayLayered));
  CHECK(hipMemset(A->data, 0, Pitch * Dp));
  std::vector<unsigned char> Src(Pitch * Dp, 0x3C), H(Pitch * Dp, 0xAA);
  hipMemcpy3DParms P = {};
  P.srcPtr = make_hipPitchedPtr(Src.data(), Pitch, W, 1);
  P.dstArray = A;
  P.extent = make_hipExtent(W - 1, 1, Dp);
  P.kind = hipMemcpyHostToDevice;
  CHECK(hipMemcpy3D(&P));
  CHECK(hipMemcpy(H.data(), A->data, H.size(), hipMemcpyDeviceToHost));
  for (size_t Off = 0; Off < H.size(); Off++) {
    unsigned char Want =
        Off % Pitch < (W - 1) * sizeof(unsigned int) ? 0x3C : 0;
    if (H[Off] != Want) {
      printf("FAIL: layered: byte %zu (row %zu) = 0x%02x, expected 0x%02x\n",
             Off, Off / Pitch, H[Off], Want);
      return 1;
    }
  }
  CHECK(hipFreeArray(A));
  return 0;
}

// A host to host copy waits for earlier stream work and is complete on return.
static void fillSrc(hipStream_t, hipError_t, void *Src) {
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  memset(Src, 0x3C, 16 * 2 * 2);
}

static int hostOrderCase() {
  hipStream_t Stream;
  CHECK(hipStreamCreate(&Stream));
  unsigned char Src[16 * 2 * 2] = {}, Dst[16 * 1 * 2] = {};
  CHECK(hipStreamAddCallback(Stream, fillSrc, Src, 0));
  hipMemcpy3DParms P = {};
  P.srcPtr = make_hipPitchedPtr(Src, 16, 16, 2);
  P.dstPtr = make_hipPitchedPtr(Dst, 16, 16, 1);
  P.extent = make_hipExtent(16, 1, 2);
  P.kind = hipMemcpyHostToHost;
  CHECK(hipMemcpy3DAsync(&P, Stream));
  int R = 0;
  for (size_t I = 0; I < sizeof Dst; I++)
    if (Dst[I] != 0x3C) {
      printf("FAIL: host order: Dst[%zu] = 0x%02x, expected 0x3c\n", I, Dst[I]);
      R = 1;
      break;
    }
  CHECK(hipStreamSynchronize(Stream));
  CHECK(hipStreamDestroy(Stream));
  return R;
}

int main() {
  hipPitchedPtr S, D;
  CHECK(hipMalloc3D(&S, make_hipExtent(128, 2, 2)));
  CHECK(hipMalloc3D(&D, make_hipExtent(128, 3, 2)));
  CHECK(hipMemset(S.ptr, 0x3C, S.pitch * S.ysize * 2));

  int Fail = pitchedCase("narrow", S, D, make_hipExtent(127, 1, 2));
  if (S.pitch == D.pitch)
    Fail |= pitchedCase("full width", S, D, make_hipExtent(D.pitch, 1, 2));

  int R = arrayCase("array full width", 16);
  if (R != CHIP_SKIP_TEST) {
    Fail |= R;
    Fail |= arrayCase("array narrow", 15);
    Fail |= layeredCase();
  }
  Fail |= hostOrderCase();
  if (Fail)
    return 1;
  printf("PASS\n");
  return 0;
}
