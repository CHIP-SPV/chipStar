// Reproduces #1866: a std::exception thrown inside a HIP API escaped it as a
// C++ exception instead of being returned as an error code.
#include <hip/hip_runtime_api.h>
#include "CHIPDriver.hh"
#include "CHIPException.hh"
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>

static thread_local bool FailNextNew = false;

void *operator new(std::size_t N) {
  if (FailNextNew) {
    FailNextNew = false;
    throw std::bad_alloc();
  }
  if (void *P = std::malloc(N ? N : 1))
    return P;
  throw std::bad_alloc();
}
void operator delete(void *P) noexcept { std::free(P); }
void operator delete(void *P, std::size_t) noexcept { std::free(P); }

static hipError_t throwRuntimeError() {
  CHIP_TRY
  throw std::runtime_error("injected");
  CHIP_CATCH
}

static bool check(const char *What, hipError_t (*Call)(), hipError_t Expected) {
  hipError_t Err;
  try {
    Err = Call();
  } catch (const std::exception &E) {
    printf("FAIL: %s: exception escaped: %s\n", What, E.what());
    return false;
  }
  if (Err != Expected) {
    printf("FAIL: %s: got %s, expected %s\n", What, hipGetErrorName(Err),
           hipGetErrorName(Expected));
    return false;
  }
  return true;
}

static hipError_t createWithFailingNew() {
  hipGraph_t G;
  FailNextNew = true;
  hipError_t Err = hipGraphCreate(&G, 0);
  FailNextNew = false;
  return Err;
}

int main() {
  hipGraph_t G;
  if (hipGraphCreate(&G, 0) != hipSuccess) {
    printf("FAIL: warm-up call failed\n");
    return 1;
  }
  bool Ok = check("bad_alloc", createWithFailingNew, hipErrorOutOfMemory);
  Ok &= check("runtime_error", throwRuntimeError, hipErrorUnknown);
  if (!Ok)
    return 1;
  printf("PASS\n");
  return 0;
}
