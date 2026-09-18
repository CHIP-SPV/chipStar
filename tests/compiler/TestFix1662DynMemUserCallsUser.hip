// A kernel that writes dynamic shared memory and calls another writer of it.
extern __shared__ int smem[];
__device__ __attribute__((noinline)) void h(int v) { smem[0] = v; }
__global__ void k(int v) { smem[0] = v; h(v); }
int main() { return 0; }
