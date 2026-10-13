// Canary for https://github.com/CHIP-SPV/chipStar/issues/1896. MEANT TO FAIL
// EVENTUALLY. Runs a plain OpenCL C kernel, no chipStar involved, and asserts
// IGC still miscompiles a SIMD32 sub_group_shuffle of as_int(float). When it
// fails, delete HipFloatShufflePass and its registration in
// llvm_passes/HipPasses.cpp and this test together, then close #1896 naming
// the IGC release that fixed it.
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static const char *Src = R"CLC(
__attribute__((intel_reqd_sub_group_size(32)))
__kernel void k(__global const float4* pos, __global const float4* posB, __global float* out) {
  uint lid = get_local_id(0);
  float4 p1 = pos[lid];
  float4 p2 = posB[lid];
  float e = 0;
  uint src = (get_sub_group_local_id() & ~31u) | ((lid + 1) & 31u);
  #pragma unroll 1
  for (int j = 0; j < 32; j++) {
    float dx = p2.x - p1.x, dy = p2.y - p1.y, dz = p2.z - p1.z;
    if (p2.w != 0.0f)
      e += (dx * dx + dy * dy + dz * dz) * (float)(j + 1) * p2.w;
    p2 = (float4)(as_float(sub_group_shuffle(as_int(p2.x), src)), as_float(sub_group_shuffle(as_int(p2.y), src)),
                  as_float(sub_group_shuffle(as_int(p2.z), src)), as_float(sub_group_shuffle(as_int(p2.w), src)));
  }
  out[lid] = e;
}
)CLC";

int main() {
  constexpr int N = 64;
  cl_uint NP = 0;
  clGetPlatformIDs(0, nullptr, &NP);
  std::vector<cl_platform_id> Ps(NP);
  clGetPlatformIDs(NP, Ps.data(), nullptr);
  // Only IGC targets without 8-wide sub-groups (PVC, Xe2, Xe3) take the
  // in-place shuffle path; DG2, ADL, MTL compute this correctly.
  cl_device_id Dev = nullptr;
  for (cl_platform_id P : Ps) {
    cl_uint ND = 0;
    clGetDeviceIDs(P, CL_DEVICE_TYPE_GPU, 0, nullptr, &ND);
    std::vector<cl_device_id> Ds(ND);
    clGetDeviceIDs(P, CL_DEVICE_TYPE_GPU, ND, Ds.data(), nullptr);
    for (cl_device_id D : Ds) {
      cl_uint Vendor = 0;
      size_t Sizes[8] = {}, Bytes = 0;
      clGetDeviceInfo(D, CL_DEVICE_VENDOR_ID, sizeof(Vendor), &Vendor, nullptr);
      if (Vendor == 0x8086 && !Dev &&
          clGetDeviceInfo(D, CL_DEVICE_SUB_GROUP_SIZES_INTEL, sizeof(Sizes),
                          Sizes, &Bytes) == CL_SUCCESS &&
          std::count(Sizes, Sizes + Bytes / sizeof(size_t), 8) == 0 &&
          std::count(Sizes, Sizes + Bytes / sizeof(size_t), 32) == 1)
        Dev = D;
    }
  }
  if (!Dev) {
    printf("HIP_SKIP_THIS_TEST: no affected Intel GPU\n");
    return 0;
  }
  cl_int Err;
  cl_context Ctx = clCreateContext(nullptr, 1, &Dev, nullptr, nullptr, &Err);
  cl_command_queue Q = clCreateCommandQueue(Ctx, Dev, 0, &Err);
  cl_program Prog = clCreateProgramWithSource(Ctx, 1, &Src, nullptr, &Err);
  if (clBuildProgram(Prog, 1, &Dev, "", nullptr, nullptr) != CL_SUCCESS) {
    printf("FAIL: kernel did not build\n");
    return 1;
  }
  cl_kernel K = clCreateKernel(Prog, "k", &Err);

  std::vector<cl_float4> Pos(N), PosB(N);
  for (int I = 0; I < N; ++I) {
    Pos[I] = {{(float)I, 0.0f, 0.0f, 1.0f}};
    PosB[I] = {{0.5f * I, (float)(I % 7), 1.0f, 1.0f + I % 3}};
  }
  cl_mem BPos = clCreateBuffer(Ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               N * sizeof(cl_float4), Pos.data(), &Err);
  cl_mem BPosB = clCreateBuffer(Ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                N * sizeof(cl_float4), PosB.data(), &Err);
  cl_mem BOut =
      clCreateBuffer(Ctx, CL_MEM_WRITE_ONLY, N * sizeof(float), nullptr, &Err);
  clSetKernelArg(K, 0, sizeof(cl_mem), &BPos);
  clSetKernelArg(K, 1, sizeof(cl_mem), &BPosB);
  clSetKernelArg(K, 2, sizeof(cl_mem), &BOut);
  size_t Global = N, Local = N;
  std::vector<float> Out(N);
  if (clEnqueueNDRangeKernel(Q, K, 1, nullptr, &Global, &Local, 0, nullptr,
                             nullptr) != CL_SUCCESS ||
      clEnqueueReadBuffer(Q, BOut, CL_TRUE, 0, N * sizeof(float), Out.data(), 0,
                          nullptr, nullptr) != CL_SUCCESS) {
    printf("FAIL: kernel did not run\n");
    return 1;
  }

  int Wrong = 0;
  for (int L = 0; L < N; ++L) {
    float E = 0;
    for (int J = 0; J < 32; ++J) {
      const cl_float4 &P2 = PosB[(L & ~31) | ((L + J) & 31)];
      float Dx = P2.s[0] - Pos[L].s[0], Dy = P2.s[1] - Pos[L].s[1],
            Dz = P2.s[2] - Pos[L].s[2];
      E += (Dx * Dx + Dy * Dy + Dz * Dz) * (float)(J + 1) * P2.s[3];
    }
    Wrong += std::fabs(Out[L] - E) > 1e-4f * std::fabs(E);
  }
  printf("%d of %d lanes wrong\n", Wrong, N);
  if (Wrong == 0) {
    printf("FAIL: CANARY FIRED, IGC shuffles as_int(float) correctly now; see "
           "the header\n");
    return 1;
  }
  printf("PASS: IGC still miscompiles the shuffle\n");
  return 0;
}
