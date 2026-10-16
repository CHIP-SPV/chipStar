// Reproduces #1984: re-recording an event must not grow its dependency list.
#include "CHIPBackend.hh"
#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
  hipStream_t S;
  hipEvent_t E;
  if (hipStreamCreate(&S) != hipSuccess || hipEventCreate(&E) != hipSuccess)
    return 1;
  auto *ChipE = static_cast<chipstar::Event *>(E);

  size_t First = 0, Last = 0;
  for (int I = 0; I < 100; ++I) {
    if (hipEventRecord(E, S) != hipSuccess)
      return 1;
    LOCK(ChipE->DependsOnListMtx);
    (I == 0 ? First : Last) = ChipE->DependsOnList.size();
  }
  if (hipStreamSynchronize(S) != hipSuccess)
    return 1;
  if (First == 0 || Last != First) {
    printf("FAIL: %zu dependencies after 1 record, %zu after 100\n",
           First, Last);
    return 1;
  }
  printf("PASS\n");
  return 0;
}
