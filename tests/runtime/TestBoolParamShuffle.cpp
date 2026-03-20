/**
 * Regression test for IGC bug: subgroup shuffle returns wrong results
 * when any kernel function parameter has OpTypeBool in SPIR-V.
 *
 * Two kernels perform an inclusive prefix sum (warp scan) via __shfl_up_sync:
 *   - scan_int_param: 4th parameter is int    (should PASS)
 *   - scan_bool_param: 4th parameter is bool  (FAIL without workaround)
 *
 * The bool parameter is unused by the scan logic; its mere presence in the
 * function signature triggers the IGC miscompilation.
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static constexpr int WARP_SIZE = 32;
static constexpr int N = 64; // two warps

/// Inclusive prefix sum using __shfl_up_sync -- 4th param is int.
__global__ void scan_int_param(const int *in, int *out, int n,
                               int /*unused_flag*/) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n)
    return;

  int val = in[tid];
  for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
    int up = __shfl_up(val, offset, WARP_SIZE);
    if ((tid % WARP_SIZE) >= offset)
      val += up;
  }
  out[tid] = val;
}

/// Inclusive prefix sum using __shfl_up_sync -- 4th param is bool.
__global__ void scan_bool_param(const int *in, int *out, int n,
                                bool /*unused_flag*/) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n)
    return;

  int val = in[tid];
  for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
    int up = __shfl_up(val, offset, WARP_SIZE);
    if ((tid % WARP_SIZE) >= offset)
      val += up;
  }
  out[tid] = val;
}

static bool verify(const char *label, const int *out, int n) {
  bool pass = true;
  int first_fail = -1;
  for (int i = 0; i < n; i++) {
    // Expected: inclusive prefix sum within each warp.
    // Input is all-ones so expected[i] = (i % WARP_SIZE) + 1.
    int expected = (i % WARP_SIZE) + 1;
    if (out[i] != expected) {
      if (first_fail < 0)
        first_fail = i;
      pass = false;
    }
  }
  if (pass) {
    printf("%s: PASS\n", label);
  } else {
    printf("%s: FAIL (first mismatch at [%d]: got %d, expected %d)\n", label,
           first_fail, out[first_fail], (first_fail % WARP_SIZE) + 1);
  }
  return pass;
}

int main() {
  // Skip on devices that don't support the Intel SPIR-V subgroup ops chipStar
  // emits for __shfl_up. e.g. Mali-G52 has cl_khr_subgroup_shuffle but not
  // cl_intel_subgroups, so the kernel SPIR-V is rejected at clBuildProgram.
  // Detect by device name since chipStar reports warpSize=32 unconditionally.
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  if (std::strstr(prop.name, "Mali") != nullptr) {
    printf("SKIP: device '%s' lacks SPV_INTEL_subgroups support\n", prop.name);
    return CHIP_SKIP_TEST;
  }

  int h_in[N], h_out[N];
  for (int i = 0; i < N; i++)
    h_in[i] = 1;

  int *d_in, *d_out;
  hipMalloc(&d_in, N * sizeof(int));
  hipMalloc(&d_out, N * sizeof(int));
  hipMemcpy(d_in, h_in, N * sizeof(int), hipMemcpyHostToDevice);

  int blocks = (N + 63) / 64;

  // Test 1: int parameter (baseline, should always pass)
  hipMemset(d_out, 0, N * sizeof(int));
  scan_int_param<<<blocks, 64>>>(d_in, d_out, N, 0);
  hipDeviceSynchronize();
  hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost);
  bool pass1 = verify("scan_int_param", h_out, N);

  // Test 2: bool parameter (triggers IGC bug without workaround)
  hipMemset(d_out, 0, N * sizeof(int));
  scan_bool_param<<<blocks, 64>>>(d_in, d_out, N, false);
  hipDeviceSynchronize();
  hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost);
  bool pass2 = verify("scan_bool_param", h_out, N);

  hipFree(d_in);
  hipFree(d_out);

  return (pass1 && pass2) ? 0 : 1;
}
