#include "hip/hip_runtime.h"

#define ERR_CHECK                                                              \
  do {                                                                         \
    if (err != hipSuccess) {                                                   \
      std::cerr << "HIP API error\n";                                          \
      return -1;                                                               \
    }                                                                          \
  } while (0)