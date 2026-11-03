// A failed hipInitFromNativeHandles must not leave the per thread default
// queue to be destroyed at thread exit against a deleted Backend.
// https://github.com/CHIP-SPV/chipStar/issues/2282
#include <hip/hip_interop.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

// TLS destructors run before atexit handlers, so PASS prints only if they survive.
static void printPass() {
  printf("PASS\n");
  fflush(stdout);
}

int main() {
  (void)hipStreamSynchronize(hipStreamPerThread);
  if (hipInitFromNativeHandles(nullptr, 0) != hipErrorInitializationError) {
    printf("FAIL: expected hipErrorInitializationError\n");
    return 1;
  }
  atexit(printPass);
  return 0;
}
