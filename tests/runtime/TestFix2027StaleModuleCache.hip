// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2027: a library
// loaded after another was dlclosed must not get the old library's kernel.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <dlfcn.h>

__global__ void mainKernel(int *O) { *O = 0; }

static int runLib(const char *Path) {
  void *H = dlopen(Path, RTLD_NOW | RTLD_LOCAL);
  if (!H)
    return -3;
  int R = reinterpret_cast<int (*)()>(dlsym(H, "run"))();
  dlclose(H);
  return R;
}

int main() {
  int *D;
  (void)hipMalloc(&D, sizeof(int));
  mainKernel<<<1, 1>>>(D); // keeps the main fat binary and the backend alive
  (void)hipDeviceSynchronize();
  int A = runLib(LIB_A), B = runLib(LIB_B);
  bool Ok = A == 1 && B == 2;
  std::printf("libA=%d (want 1) libB=%d (want 2) %s\n", A, B,
              Ok ? "PASS" : "FAIL");
  return !Ok;
}
