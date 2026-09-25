// A kernel calling a shared memory user whose result and argument sign extend;
// signed char, since plain char is unsigned on aarch64.
extern __shared__ char smem[];
__device__ __attribute__((noinline)) signed char h(signed char v) { smem[0] = v; return v; }
__global__ void a(signed char v, signed char *o) { o[0] = h(v); }
int main() { return 0; }
