// A kernel calling a shared memory user whose result and argument sign extend.
extern __shared__ char smem[];
__device__ __attribute__((noinline)) char h(char v) { smem[0] = v; return v; }
__global__ void a(char v, char *o) { o[0] = h(v); }
int main() { return 0; }
