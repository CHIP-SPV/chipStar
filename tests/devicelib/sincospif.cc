#include <hip/hip_runtime.h>
__global__ void sincospif_kernel() {
    float k;
    sincospif(1.0f,&k, &k);
}

int main() {
    // run the sincospif kernel
    sincospif_kernel<<<1,1>>>();
}