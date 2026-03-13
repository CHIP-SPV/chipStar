/*
 * Copyright (c) 2024 chipStar developers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

// Standalone reproducer for chipStar issue #1200:
// Silent segfault (exit 139) when running projection basis kernels via HIPRTC.
//
// This test reproduces the scenario from libCEED t319-basis (projection interp
// and grad in multiple dimensions) which segfaults on Intel Arc A770 with the
// OpenCL backend. The kernel mirrors the hip-ref-basis-tensor.h Interp kernel
// compiled via HIPRTC in libCEED's HIP reference backend.

#include "TestCommon.hh"
#include <vector>

// The projection basis kernel source from libCEED's hip-ref-basis-tensor.h,
// inlined with CeedInt/CeedScalar substituted with concrete types.
// This is the exact kernel shape that libCEED compiles for dim=3 projection.
static constexpr auto ProjectionBasisKernelSource = R"---(
typedef int    CeedInt;
typedef double CeedScalar;

extern "C" __global__ void Interp(const CeedInt num_elem,
                                  const CeedInt transpose,
                                  const CeedScalar * __restrict__ interp_1d,
                                  const CeedScalar * __restrict__ u,
                                  CeedScalar       * __restrict__ v) {
  const CeedInt i = threadIdx.x;

  __shared__ CeedScalar s_mem[BASIS_Q_1D * BASIS_P_1D + 2 * BASIS_BUF_LEN];
  CeedScalar *s_interp_1d = s_mem;
  CeedScalar *s_buffer_1  = s_mem + BASIS_Q_1D * BASIS_P_1D;
  CeedScalar *s_buffer_2  = s_buffer_1 + BASIS_BUF_LEN;

  for (CeedInt k = i; k < BASIS_Q_1D * BASIS_P_1D; k += blockDim.x) {
    s_interp_1d[k] = interp_1d[k];
  }

  const CeedInt P             = transpose ? BASIS_Q_1D : BASIS_P_1D;
  const CeedInt Q             = transpose ? BASIS_P_1D : BASIS_Q_1D;
  const CeedInt stride_0      = transpose ? 1 : BASIS_P_1D;
  const CeedInt stride_1      = transpose ? BASIS_P_1D : 1;
  const CeedInt u_stride      = transpose ? BASIS_NUM_QPTS : BASIS_NUM_NODES;
  const CeedInt v_stride      = transpose ? BASIS_NUM_NODES : BASIS_NUM_QPTS;
  const CeedInt u_comp_stride = num_elem * (transpose ? BASIS_NUM_QPTS : BASIS_NUM_NODES);
  const CeedInt v_comp_stride = num_elem * (transpose ? BASIS_NUM_NODES : BASIS_NUM_QPTS);
  const CeedInt u_size        = transpose ? BASIS_NUM_QPTS : BASIS_NUM_NODES;

  for (CeedInt elem = blockIdx.x; elem < num_elem; elem += gridDim.x) {
    for (CeedInt comp = 0; comp < BASIS_NUM_COMP; comp++) {
      const CeedScalar *cur_u = u + elem * u_stride + comp * u_comp_stride;
      CeedScalar       *cur_v = v + elem * v_stride + comp * v_comp_stride;
      CeedInt           pre   = u_size;
      CeedInt           post  = 1;

      for (CeedInt k = i; k < u_size; k += blockDim.x) {
        s_buffer_1[k] = cur_u[k];
      }
      for (CeedInt d = 0; d < BASIS_DIM; d++) {
        __syncthreads();
        pre /= P;
        const CeedScalar *in       = d % 2 ? s_buffer_2 : s_buffer_1;
        CeedScalar       *out      = d == BASIS_DIM - 1 ? cur_v : (d % 2 ? s_buffer_1 : s_buffer_2);
        const CeedInt     writeLen = pre * post * Q;

        for (CeedInt k = i; k < writeLen; k += blockDim.x) {
          const CeedInt c  = k % post;
          const CeedInt j  = (k / post) % Q;
          const CeedInt a  = k / (post * Q);
          CeedScalar    vk = 0;

          for (CeedInt b = 0; b < P; b++)
            vk += s_interp_1d[j * stride_0 + b * stride_1] * in[(a * P + b) * post + c];
          out[k] = vk;
        }
        post *= Q;
      }
    }
  }
}
)---";

// Compile and run the projection basis kernel with the given parameters.
// dim=3, P_1d=5, Q_1d=6 matches the libCEED t319-basis projection case.
static void testProjectionKernel(int dim, int P_1d, int Q_1d) {
  std::cerr << "Testing projection kernel: dim=" << dim
            << " P_1d=" << P_1d << " Q_1d=" << Q_1d << "\n";

  int num_comp = 1;
  int num_nodes_1d = P_1d;
  int num_qpts_1d  = Q_1d;
  int num_nodes = 1, num_qpts = 1;
  for (int d = 0; d < dim; d++) {
    num_nodes *= num_nodes_1d;
    num_qpts  *= num_qpts_1d;
  }
  int buf_len = num_comp;
  {
    int max_1d = P_1d > Q_1d ? P_1d : Q_1d;
    int val = max_1d;
    for (int d = 1; d < dim; d++) val *= max_1d;
    buf_len *= val;
  }

  // Build a simple identity-like interp_1d matrix (Q_1d x P_1d).
  // For test purposes we just need something non-trivial.
  std::vector<double> interp_1d(Q_1d * P_1d, 0.0);
  for (int q = 0; q < Q_1d; q++)
    for (int p = 0; p < P_1d; p++)
      interp_1d[q * P_1d + p] = (p == q % P_1d) ? 1.0 : 0.1 * (q * P_1d + p);

  // Input: num_nodes doubles, output: num_qpts doubles
  int num_elem = 1;
  std::vector<double> u_h(num_nodes, 1.0);
  for (int i = 0; i < num_nodes; i++)
    u_h[i] = (double)(i + 1) * 0.01;

  // Allocate device memory
  double *d_interp = nullptr, *d_u = nullptr, *d_v = nullptr;
  HIP_CHECK(hipMalloc(&d_interp, Q_1d * P_1d * sizeof(double)));
  HIP_CHECK(hipMalloc(&d_u, num_nodes * sizeof(double)));
  HIP_CHECK(hipMalloc(&d_v, num_qpts  * sizeof(double)));
  HIP_CHECK(hipMemcpy(d_interp, interp_1d.data(), Q_1d * P_1d * sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_u, u_h.data(), num_nodes * sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_v, 0, num_qpts * sizeof(double)));

  // Compile the kernel via HIPRTC
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, ProjectionBasisKernelSource,
                                   "projection_basis", 0, nullptr, nullptr));

  // Build the #define options that libCEED passes
  auto makeDefine = [](const char *name, int val) -> std::string {
    return std::string("-D") + name + "=" + std::to_string(val);
  };
  std::vector<std::string> opt_strs = {
    makeDefine("BASIS_Q_1D",      Q_1d),
    makeDefine("BASIS_P_1D",      P_1d),
    makeDefine("BASIS_BUF_LEN",   buf_len),
    makeDefine("BASIS_DIM",       dim),
    makeDefine("BASIS_NUM_COMP",  num_comp),
    makeDefine("BASIS_NUM_NODES", num_nodes),
    makeDefine("BASIS_NUM_QPTS",  num_qpts),
    // These three options are what libCEED passes (and chipStar ignores):
    "-default-device",
    "--gpu-architecture=unavailable",
    "-munsafe-fp-atomics",
  };
  std::vector<const char *> opts;
  for (auto &s : opt_strs) opts.push_back(s.c_str());

  auto code = HiprtcAssertCompileProgram(prog, opts);

  // Load the compiled module
  hipModule_t   module;
  hipFunction_t kernel;
  HIP_CHECK(hipModuleLoadData(&module, code.data()));
  HIP_CHECK(hipModuleGetFunction(&kernel, module, "Interp"));

  // Launch kernel: block_size = min(num_qpts, 64)
  int block_size = num_qpts < 64 ? num_qpts : 64;
  int is_transpose = 0;
  void *args[] = { &num_elem, &is_transpose, &d_interp, &d_u, &d_v };
  HIP_CHECK(hipModuleLaunchKernel(kernel, /*grid*/ 1, 1, 1,
                                  /*block*/ block_size, 1, 1,
                                  0, nullptr, args, nullptr));
  // Intentionally NO hipDeviceSynchronize() here — this mirrors libCEED's
  // CeedBasisApply_Hip which launches without syncing.  The subsequent
  // hipModuleUnload + hipFree sequence (matching CeedBasisDestroy_Hip) is
  // what triggers the silent segfault on Intel Arc A770 / OpenCL backend
  // (chipStar issue #1200).  hipFreeInternal calls hipDeviceSynchronizeInternal
  // which calls clFinish, crashing inside libigdrcl.so.

  // Mirror CeedBasisDestroy_Hip order: hipModuleUnload first, then hipFree.
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  HIP_CHECK(hipModuleUnload(module));

  // hipFree triggers implicit sync (hipDeviceSynchronizeInternal → clFinish).
  // This is the crash site on affected hardware.
  HIP_CHECK(hipFree(d_interp));
  HIP_CHECK(hipFree(d_u));
  HIP_CHECK(hipFree(d_v));

  std::cerr << "  PASSED (dim=" << dim << " P=" << P_1d << " Q=" << Q_1d << ")\n";
}

int main() {
  // Test the projection basis kernel for each dimension as libCEED t319 does.
  // dim=1: P_from=5, P_to=6 => P_1d=5, Q_1d=6
  // dim=2: same P/Q, larger arrays
  // dim=3: same P/Q but 3D, largest arrays — this is the crashing case
  for (int dim = 1; dim <= 3; dim++) {
    testProjectionKernel(dim, /*P_1d=*/5, /*Q_1d=*/6);
  }
  std::cerr << "All projection basis HIPRTC tests PASSED\n";
  return 0;
}
