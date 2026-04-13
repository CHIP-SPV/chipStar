#include <hip/hip_runtime.h>
#include <iostream>

// Declared in helper.c (pure C, uses 'class' as a C identifier).
// If -xhip/-x hip is applied to helper.c the compiler rejects it.
extern "C" int get_value(void);

int main() {
  // Verify chipStar runtime is reachable.
  int deviceCount = 0;
  if (hipGetDeviceCount(&deviceCount) != hipSuccess)
    return 1;

  // Call the pure-C helper to confirm it compiled correctly.
  int v = get_value();
  std::cout << "get_value() = " << v << "\n";
  return v != 42;
}
