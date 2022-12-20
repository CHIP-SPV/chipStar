#include "hip/hip_runtime.h"

#define ERR_CHECK(x)                                                           \
  do {                                                                         \
    if (x != hipSuccess) {                                                     \
      std::cerr << "FAILED: HIP API error\n";                                  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)