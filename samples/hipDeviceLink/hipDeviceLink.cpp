#include "hipDeviceLinkConsts.h"
#include <hip/hip_runtime.h>
#include <assert.h>
#include <stdio.h>
 
int main() {
  int *hostIn, *hostOut;
  hostIn = new int[NUM];
  hostOut = new int[NUM];
  for (int i = 0; i < NUM; i++) {
    hostIn[i] = -1 * i;
    hostOut[i] = 0;
  }
  writeGlobal(hostIn);
  readGlobal(hostOut);
  for (int i = 0; i < NUM; i++) {
    assert(hostIn[i] == hostOut[i]);
  }
  delete[] hostIn;
  delete[] hostOut;
  printf("PASSED!\n");
}
