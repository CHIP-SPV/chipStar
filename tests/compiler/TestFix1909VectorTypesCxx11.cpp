// The HIP headers must compile as C++11 (#1909).
#include <hip/hip_runtime.h>

bool f(int2 a, int2 b) { return a == b; }
