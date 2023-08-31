// Simulate a header shared among HIP and C++ sources.
#include <hip/hip_runtime.h>
#if defined(__HIPCC__) || defined(__HIP__)
#  define DEVICE __device__
#else
#  define DEVICE
#endif
DEVICE int foo() { return 123; }

// Simulate some C++ source file.
int main() { return !(foo() == 123); }
