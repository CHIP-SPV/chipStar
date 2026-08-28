// Reproduces: hipHostGetDevicePointer() called on a pointer inside a
// hipHostMalloc(hipHostMallocMapped) or hipHostRegister(hipHostRegisterMapped)
// buffer returns the device address of the allocation base, dropping the
// offset of the queried pointer within the allocation. Trilinos STK does one
// hipHostMalloc per bucket and asks for the device pointer of every field's
// interior pointer (stk_mesh FieldDataManager get_host_pointer_for_device), so
// every field but the first gets a device pointer to the wrong bytes.
//
// For several interior pointers this checks two things: the returned device
// pointer has the same offset from the base's device pointer as the host
// pointer has from the host base, and a kernel reading one word through the
// returned device pointer sees the value the host wrote at that interior
// pointer. The hipHostRegister half queries an interior pointer before the
// base, which is the order STK uses, so the lazily created device backing is
// exercised the same way.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(Expr)                                                            \
  do {                                                                         \
    hipError_t E = (Expr);                                                     \
    if (E != hipSuccess) {                                                     \
      printf("FAIL: %s -> %s (%d) at line %d\n", #Expr, hipGetErrorString(E),  \
             (int)E, __LINE__);                                                \
      return 1;                                                                \
    }                                                                          \
  } while (0)

constexpr size_t Size = 1u << 20; // 1 MiB
constexpr size_t Offsets[] = {4096, 65536 + 8, Size - 8};
constexpr size_t NumOffsets = sizeof(Offsets) / sizeof(Offsets[0]);

__global__ void readWord(const uint32_t *In, uint32_t *Out) { *Out = *In; }

static int Failures = 0;

static void fillPattern(char *Base) {
  for (size_t I = 0; I < Size / sizeof(uint32_t); I++) {
    uint32_t W = 0x9e3779b9u * (uint32_t)I + 0x1234567u;
    std::memcpy(Base + I * sizeof(uint32_t), &W, sizeof(W));
  }
}

// Queries the device pointer of Base + Offset and verifies both the pointer
// arithmetic against DevBase and the bytes a kernel reads through it.
static int checkInterior(const char *What, char *Base, void *DevBase,
                         size_t Offset, uint32_t *DevOut) {
  char *H = Base + Offset;
  void *Dev = nullptr;
  CHECK(hipHostGetDevicePointer(&Dev, H, 0));
  ptrdiff_t DevOff = (char *)Dev - (char *)DevBase;
  printf("%s: host base+%zu -> device base%+td\n", What, Offset, DevOff);
  if (DevOff != (ptrdiff_t)Offset) {
    printf("  mismatch: device offset %td, expected %zu\n", DevOff, Offset);
    Failures++;
  }

  const uint32_t Sentinel = 0xdeadbeefu;
  CHECK(hipMemcpy(DevOut, &Sentinel, sizeof(Sentinel), hipMemcpyHostToDevice));
  readWord<<<1, 1>>>((const uint32_t *)Dev, DevOut);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  uint32_t Got = 0;
  CHECK(hipMemcpy(&Got, DevOut, sizeof(Got), hipMemcpyDeviceToHost));
  uint32_t Want;
  std::memcpy(&Want, H, sizeof(Want));
  if (Got != Want) {
    printf("  mismatch: kernel read 0x%08x through the device pointer, host "
           "holds 0x%08x at that offset\n",
           Got, Want);
    Failures++;
  }
  return 0;
}

static int testHostMalloc(uint32_t *DevOut) {
  char *Base = nullptr;
  CHECK(hipHostMalloc((void **)&Base, Size, hipHostMallocMapped));
  fillPattern(Base);

  void *DevBase = nullptr;
  CHECK(hipHostGetDevicePointer(&DevBase, Base, 0));
  printf("hipHostMalloc: host %p -> device %p\n", (void *)Base, DevBase);

  for (size_t I = 0; I < NumOffsets; I++)
    if (checkInterior("hipHostMalloc", Base, DevBase, Offsets[I], DevOut))
      return 1;

  CHECK(hipHostFree(Base));
  return 0;
}

static int testHostRegister(uint32_t *DevOut) {
  char *Base = nullptr;
  if (posix_memalign((void **)&Base, 4096, Size) != 0 || !Base) {
    printf("FAIL: posix_memalign\n");
    return 1;
  }
  fillPattern(Base);
  CHECK(hipHostRegister(Base, Size, hipHostRegisterMapped));

  // Interior pointer first: this is the query that creates the device backing.
  void *DevFirst = nullptr;
  CHECK(hipHostGetDevicePointer(&DevFirst, Base + Offsets[0], 0));

  void *DevBase = nullptr;
  CHECK(hipHostGetDevicePointer(&DevBase, Base, 0));
  printf("hipHostRegister: host %p -> device %p\n", (void *)Base, DevBase);

  ptrdiff_t FirstOff = (char *)DevFirst - (char *)DevBase;
  if (FirstOff != (ptrdiff_t)Offsets[0]) {
    printf("  mismatch: first query (base+%zu) sits at device base%+td\n",
           Offsets[0], FirstOff);
    Failures++;
  }

  for (size_t I = 0; I < NumOffsets; I++)
    if (checkInterior("hipHostRegister", Base, DevBase, Offsets[I], DevOut))
      return 1;

  CHECK(hipHostUnregister(Base));
  free(Base);
  return 0;
}

int main() {
  uint32_t *DevOut = nullptr;
  CHECK(hipMalloc(&DevOut, sizeof(uint32_t)));

  if (testHostMalloc(DevOut))
    return 1;
  if (testHostRegister(DevOut))
    return 1;

  CHECK(hipFree(DevOut));

  if (Failures) {
    printf("FAIL: %d mismatches\n", Failures);
    return 1;
  }
  printf("PASS\n");
  return 0;
}
