// Kernels that call a shared memory user both directly and through another user.
extern __shared__ int smem[];
__device__ __attribute__((noinline)) void h(int v) { smem[0] = v; }
__device__ __attribute__((noinline)) void b(int v) { h(v); }
__global__ void a(int v) { b(v); h(v); }
// k and c mirror a and b in definition order, so either use list order has a
// callee that is cloned before its caller.
__device__ void c(int v);
__global__ void k(int v) { c(v); h(v); }
__device__ __attribute__((noinline)) void c(int v) { h(v); }
int main() { return 0; }
