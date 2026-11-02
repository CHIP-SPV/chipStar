// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2027: a library
// loaded after another was dlclosed must not get the old library's kernel.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <dlfcn.h>
#include <string>

__global__ void mainKernel(int *O) { *O = 0; }

static int runLib(const std::string &Path) {
  void *H = dlopen(Path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!H)
    return -3;
  int R = reinterpret_cast<int (*)()>(dlsym(H, "run"))();
  dlclose(H);
  return R;
}

// Libraries sit next to the executable, which is run by its relocated path.
int main(int, char **argv) {
  std::string Dir = argv[0];
  Dir.erase(Dir.rfind('/') + 1);
  int *D;
  (void)hipMalloc(&D, sizeof(int));
  mainKernel<<<1, 1>>>(D); // keeps the main fat binary and the backend alive
  (void)hipDeviceSynchronize();
  int A = runLib(Dir + LIB_A), B = runLib(Dir + LIB_B);
  bool Ok = A == 1 && B == 2;
  std::printf("libA=%d (want 1) libB=%d (want 2) %s\n", A, B,
              Ok ? "PASS" : "FAIL");
  return !Ok;
}
