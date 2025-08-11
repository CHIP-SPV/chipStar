#include <hip/hip_runtime.h>

__global__ void testNew(int* num) {
  volatile int* i = (int*)malloc(sizeof(int));
  *num = *i;
}

int main() {}
