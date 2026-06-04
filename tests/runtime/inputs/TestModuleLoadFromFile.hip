// Regression test: hipModuleLoad (the file-based module loader) used to be a
// no-op stub whose body was #if 0'd out. It returned hipSuccess without ever
// initializing *Module, so callers received a garbage/null module handle and a
// subsequent hipModuleGetFunction reported "kernel not found" or segfaulted.
// This test loads a valid module file and resolves a known kernel from it.
#include <hip/hip_runtime_api.h>
#include <cstdio>

int main(int ArgC, char *ArgV[]) {
  if (ArgC != 2) {
    printf("usage: %s <module-file>\n", ArgV[0]);
    return 2;
  }
  hipModule_t Module = nullptr;
  if (hipModuleLoad(&Module, ArgV[1]) != hipSuccess) {
    printf("FAIL: hipModuleLoad returned an error\n");
    return 3;
  }
  if (Module == nullptr) {
    printf("FAIL: hipModuleLoad returned success but left Module unset\n");
    return 4;
  }
  hipFunction_t Fn = nullptr;
  if (hipModuleGetFunction(&Fn, Module, "add1") != hipSuccess || Fn == nullptr) {
    printf("FAIL: hipModuleGetFunction(add1) failed after hipModuleLoad\n");
    return 5;
  }
  printf("PASS: hipModuleLoad + hipModuleGetFunction resolved 'add1'\n");
  return 0;
}
