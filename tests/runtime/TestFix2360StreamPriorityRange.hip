// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2360: the stream
// priority range must be defined and every stream priority must lie inside it.
#include <hip/hip_runtime.h>

#include <cstdio>

#define CHECK(X)                                                               \
  do {                                                                         \
    if (!(X)) {                                                                \
      printf("FAIL: %s (line %d)\n", #X, __LINE__);                            \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  int Least = -1000, Greatest = -1000, P = -1000;
  CHECK(hipDeviceGetStreamPriorityRange(&Least, &Greatest) == hipSuccess);
  CHECK(Least >= Greatest);

  CHECK(hipStreamGetPriority(nullptr, &P) == hipSuccess);
  CHECK(P >= Greatest && P <= Least);

  hipStream_t S;
  CHECK(hipStreamCreate(&S) == hipSuccess);
  CHECK(hipStreamGetPriority(S, &P) == hipSuccess);
  CHECK(P >= Greatest && P <= Least);
  CHECK(hipStreamDestroy(S) == hipSuccess);

  for (int Want : {Greatest, Least}) {
    CHECK(hipStreamCreateWithPriority(&S, hipStreamDefault, Want) ==
          hipSuccess);
    CHECK(hipStreamGetPriority(S, &P) == hipSuccess);
    CHECK(P == Want);
    CHECK(hipStreamDestroy(S) == hipSuccess);
  }
  printf("PASS\n");
  return 0;
}
