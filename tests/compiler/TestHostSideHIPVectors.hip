// Test HIP vectors on the host-side code.
// A regression test for https://github.com/CHIP-SPV/chipStar/issues/527.
#include <hip/hip_runtime.h>

float4 checkMul(float4 x, float4 y) { return x * y; }
float4 checkMul(float4 x, float y) { return x * y; }
float4 checkMul(float x, float4 y) { return x * y; }
