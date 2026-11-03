// https://github.com/CHIP-SPV/chipStar/issues/2243: a clFinish error in uninitialize must not terminate at exit.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <dlfcn.h>
#include <unistd.h>

static bool Armed = false;
static bool Injected = false;
static int Hits = 0;
// Plain array: exe statics with destructors are gone before libCHIP's destructor runs.
static void *Seen[16];
static int NumSeen = 0;

// CHIPUninitializeCallOnce finishes each queue once, uninitialize repeats it: fail the repeat.
extern "C" int clFinish(void *Q) {
  static auto Real = (int (*)(void *))dlsym(RTLD_NEXT, "clFinish");
  Hits++;
  if (Armed) {
    for (int I = 0; I < NumSeen; I++)
      if (Seen[I] == Q) {
        Injected = true;
        return -5; // CL_OUT_OF_RESOURCES
      }
    if (NumSeen < 16)
      Seen[NumSeen++] = Q;
  }
  return Real(Q);
}

// Runs after the atexit module destructor reaches uninitialize, before libCHIP's destructor.
__attribute__((destructor)) static void CheckInjected() {
  if (Armed && !Injected) {
    printf("no clFinish error was injected\n");
    fflush(stdout);
    _exit(1);
  }
}

int main() {
  (void)hipStreamSynchronize(0);
  if (!Hits) {
    printf("HIP_SKIP_THIS_TEST: no clFinish call was intercepted\n");
    return CHIP_SKIP_TEST;
  }
  Armed = true;
  return 0;
}
