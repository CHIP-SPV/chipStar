// Reproduces #1896: IGC miscompiles a 32-lane __shfl of __float_as_int(f).
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>

__global__ void k(const float4 *pos, const float4 *posB, float *out) {
  unsigned lid = threadIdx.x;
  float4 p1 = pos[lid], p2 = posB[lid];
  float e = 0;
  int src = (lid + 1) & 31;
#pragma unroll 1
  for (int j = 0; j < 32; j++) {
    float dx = p2.x - p1.x, dy = p2.y - p1.y, dz = p2.z - p1.z;
    if (p2.w != 0.0f)
      e += (dx * dx + dy * dy + dz * dz) * (float)(j + 1) * p2.w;
    p2.x = __int_as_float(__shfl(__float_as_int(p2.x), src));
    p2.y = __int_as_float(__shfl(__float_as_int(p2.y), src));
    p2.z = __int_as_float(__shfl(__float_as_int(p2.z), src));
    p2.w = __int_as_float(__shfl(__float_as_int(p2.w), src));
  }
  out[lid] = e;
}

int main() {
  constexpr int N = 64;
  float4 pos[N], posB[N];
  float out[N];
  for (int i = 0; i < N; ++i) {
    pos[i] = make_float4(i, 0, 0, 1);
    posB[i] = make_float4(0.5f * i, i % 7, 1, 1 + i % 3);
  }
  float4 *dPos, *dPosB;
  float *dOut;
  if (hipMalloc(&dPos, sizeof(pos)) != hipSuccess ||
      hipMalloc(&dPosB, sizeof(posB)) != hipSuccess ||
      hipMalloc(&dOut, sizeof(out)) != hipSuccess ||
      hipMemcpy(dPos, pos, sizeof(pos), hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(dPosB, posB, sizeof(posB), hipMemcpyHostToDevice) != hipSuccess)
    return 1;
  hipLaunchKernelGGL(k, dim3(1), dim3(N), 0, 0, dPos, dPosB, dOut);
  if (hipMemcpy(out, dOut, sizeof(out), hipMemcpyDeviceToHost) != hipSuccess)
    return 1;
  int wrong = 0;
  for (int l = 0; l < N; ++l) {
    float e = 0;
    for (int j = 0; j < 32; ++j) {
      float4 p2 = posB[(l & ~31) | ((l + j) & 31)];
      float dx = p2.x - pos[l].x, dy = p2.y - pos[l].y, dz = p2.z - pos[l].z;
      e += (dx * dx + dy * dy + dz * dz) * (float)(j + 1) * p2.w;
    }
    wrong += std::fabs(out[l] - e) > 1e-4f * std::fabs(e);
  }
  printf("%d of %d lanes wrong\n%s\n", wrong, N, wrong ? "FAIL" : "PASS");
  return wrong != 0;
}
