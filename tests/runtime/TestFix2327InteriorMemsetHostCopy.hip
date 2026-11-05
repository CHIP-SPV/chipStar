// Reproduces https://github.com/CHIP-SPV/chipStar/issues/2327: an interior hipMemset must write the host copy at Dst's offset.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

constexpr size_t Size = 4096, Off = 1024, Len = 16;
constexpr unsigned char Old = 0xAA, New = 0x55;

__global__ void readByte(const unsigned char *In, unsigned char *Out) {
  *Out = *In;
}

static int Failures = 0;

#define CHECK(Expr)                                                            \
  do {                                                                         \
    hipError_t E = (Expr);                                                     \
    if (E != hipSuccess) {                                                     \
      printf("FAIL: %s -> %s at line %d\n", #Expr, hipGetErrorString(E),       \
             __LINE__);                                                        \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define EXPECT(Name, Got, Want)                                                \
  if ((Got) != (Want)) {                                                       \
    printf("%s: %s is 0x%02x, expected 0x%02x\n", Name, #Got, (Got), (Want));  \
    Failures++;                                                                \
  }

static int check(const char *Name, unsigned char *Host, unsigned char *Dev,
                 bool UseD8, unsigned char *DevOut) {
  auto Fill = [&](unsigned char *P, unsigned char V, size_t N) {
    return UseD8 ? hipMemsetD8((hipDeviceptr_t)P, V, N) : hipMemset(P, V, N);
  };
  // Seed through the runtime: https://github.com/CHIP-SPV/chipStar/issues/2365 loses host stores.
  CHECK(Fill(Dev, Old, Size));
  CHECK(Fill(Dev + Off, New, Len));
  EXPECT(Name, Host[0], Old);
  EXPECT(Name, Host[Off], New);
  EXPECT(Name, Host[Off + Len - 1], New);
  EXPECT(Name, Host[Off + Len], Old);
  readByte<<<1, 1>>>(Dev + Off, DevOut);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  unsigned char Got = 0;
  CHECK(hipMemcpy(&Got, DevOut, 1, hipMemcpyDeviceToHost));
  EXPECT(Name, Got, New);
  return 0;
}

int main() {
  hipDeviceProp_t Prop;
  CHECK(hipGetDeviceProperties(&Prop, 0));
  if (!Prop.canMapHostMemory) {
    printf("HIP_SKIP_THIS_TEST: requires canMapHostMemory\n");
    return CHIP_SKIP_TEST;
  }
  unsigned char *DevOut;
  CHECK(hipMalloc(&DevOut, 1));

  unsigned char *P;
  CHECK(hipHostMalloc((void **)&P, Size, hipHostMallocMapped));
  if (check("hipHostMalloc+hipMemset", P, P, false, DevOut))
    return 1;
  CHECK(hipHostFree(P));

  void *Base;
  if (posix_memalign(&Base, 4096, Size))
    return 1;
  unsigned char *Dev;
  CHECK(hipHostRegister(Base, Size, hipHostRegisterMapped));
  CHECK(hipHostGetDevicePointer((void **)&Dev, Base, 0));
  if (check("hipHostRegister+hipMemsetD8", (unsigned char *)Base, Dev, true,
            DevOut))
    return 1;
  CHECK(hipHostUnregister(Base));
  free(Base);
  CHECK(hipFree(DevOut));

  printf(Failures ? "FAIL\n" : "PASS\n");
  return Failures != 0;
}
