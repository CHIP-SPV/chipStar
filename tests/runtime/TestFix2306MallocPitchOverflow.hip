// Reproduces #2306: hipMallocPitch overflowed in pitch/size, hipMalloc3D leaked.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

static int Failed = 0;

static void check(const char *Name, bool Ok) {
  if (!Ok) {
    printf("FAIL: %s\n", Name);
    Failed = 1;
  }
}

int main() {
  void *P = nullptr;
  size_t Pitch = 0;
  hipError_t E;

  E = hipMallocPitch(&P, &Pitch, SIZE_MAX, 1);
  printf("a: err=%d ptr=%p pitch=%zu\n", E, P, Pitch);
  check("a: SIZE_MAX width -> InvalidValue, null", E == hipErrorInvalidValue && !P);
  if (P) (void)hipFree(P);

  P = nullptr;
  Pitch = 0;
  E = hipMallocPitch(&P, &Pitch, (1ull << 63) + 128, 2);
  printf("b: err=%d ptr=%p pitch=%zu\n", E, P, Pitch);
  check("b: pitch*height overflow -> OutOfMemory, null", E == hipErrorOutOfMemory && !P);
  if (P) (void)hipFree(P);

  P = nullptr;
  Pitch = 0;
  E = hipMallocPitch(&P, &Pitch, 1ull << 32, 1ull << 32);
  printf("c: err=%d ptr=%p pitch=%zu\n", E, P, Pitch);
  check("c: width*height overflow -> OutOfMemory, null", E == hipErrorOutOfMemory && !P);
  if (P) (void)hipFree(P);

  hipPitchedPtr PP = {};
  E = hipMalloc3D(&PP, make_hipExtent(SIZE_MAX, 1, 1));
  printf("d: err=%d ptr=%p\n", E, PP.ptr);
  check("d: hipMalloc3D SIZE_MAX width -> InvalidValue, null", E == hipErrorInvalidValue && !PP.ptr);
  if (PP.ptr) (void)hipFree(PP.ptr);

  PP = {};
  E = hipMalloc3D(&PP, make_hipExtent(256, (1ull << 55) + 1, 2));
  printf("e: err=%d ptr=%p\n", E, PP.ptr);
  check("e: hipMalloc3D size overflow -> OutOfMemory, null", E == hipErrorOutOfMemory && !PP.ptr);
  if (PP.ptr) (void)hipFree(PP.ptr);

  P = nullptr;
  Pitch = 0;
  E = hipMallocPitch(&P, &Pitch, 1, 1);
  printf("f: err=%d ptr=%p pitch=%zu\n", E, P, Pitch);
  check("f: control 1x1 -> Success, pitch 128", E == hipSuccess && P && Pitch == 128);
  if (P) (void)hipFree(P);

  PP = {};
  PP.pitch = 123;
  E = hipMalloc3D(&PP, make_hipExtent(1, 0, 0));
  printf("g: err=%d ptr=%p pitch=%zu\n", E, PP.ptr, PP.pitch);
  check("g: control zero height -> Success, null, pitch 0", E == hipSuccess && !PP.ptr && PP.pitch == 0);

  if (Failed)
    return 1;
  printf("PASS\n");
  return 0;
}
