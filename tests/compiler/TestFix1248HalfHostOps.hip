// Reproduces issue #1248: __half arithmetic/comparison operators must be
// __host__ __device__ (as in ROCm 6.4+), not __device__-only.
#include <hip/hip_fp16.h>

bool host_half_ops() {
    __half a = __float2half(1.5f);
    __half b = __float2half(2.5f);

    __half c = a + b;
    __half d = a - b;
    __half e = a * b;
    __half f = a / b;
    bool lt = a < b;
    bool gt = a > b;
    bool eq = a == b;
    bool ne = a != b;
    bool le = a <= b;
    bool ge = a >= b;
    a += b;
    a -= b;
    a *= b;
    a /= b;
    ++a;
    --a;
    __half g = +a;
    __half h = -a;

    (void)c; (void)d; (void)e; (void)f;
    (void)g; (void)h;
    return lt && !gt && !eq && ne && le && !ge;
}

int main() {
    return 0;
}
