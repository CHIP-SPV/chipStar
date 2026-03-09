/// \file TestHeCBenchFluidSim.hip
/// Minimal reproducer for chipStar issue #1180:
///   HeCBench fluidSim benchmark fails due to IGC miscompilation
///   of __align__(64) struct fields when used in complex expressions
///   with make_double4 and arithmetic involving other byval parameters.
///
/// The bug: when an __align__(64) struct passed by value to a kernel has
/// its fields loaded into double4 vectors AND those vectors are used in
/// arithmetic with other byval double4 parameters, the IGC produces
/// incorrect code that reads the wrong struct field values (s4-s7 get
/// the values of s0-s3).

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>

typedef struct __align__(64) {
  double s0, s1, s2, s3, s4, s5, s6, s7;
} double8;

// Reproduces the fluidSim LBM kernel pattern:
// - Two double8 structs passed by value
// - A pointer arg and scalar args
// - double4 temporaries built from struct fields via make_double4
// - Arithmetic combining the double4 temps with other byval double4 args
__global__ void lbmKernel(
    double *out,
    const double8 dirX,
    const double8 dirY,
    const double *__restrict__ weight,
    double omega,
    double f0, double4 f1234, double4 f5678) {
  // Build double4 vectors from struct fields (triggers the IGC bug)
  double4 temp = make_double4(f1234.x + f5678.x, f1234.y + f5678.y,
                               f1234.z + f5678.z, f1234.w + f5678.w);
  double rho = f0 + temp.x + temp.y + temp.z + temp.w;

  double4 x1234 = make_double4(dirX.s0, dirX.s1, dirX.s2, dirX.s3);
  double4 x5678 = make_double4(dirX.s4, dirX.s5, dirX.s6, dirX.s7);
  double4 y1234 = make_double4(dirY.s0, dirY.s1, dirY.s2, dirY.s3);
  double4 y5678 = make_double4(dirY.s4, dirY.s5, dirY.s6, dirY.s7);

  // Use the double4 vectors in arithmetic (required to trigger the bug)
  double ux = ((f1234.x * x1234.x + f1234.y * x1234.y) +
               (f1234.z * x1234.z + f1234.w * x1234.w) +
               (f5678.x * x5678.x + f5678.y * x5678.y) +
               (f5678.z * x5678.z + f5678.w * x5678.w)) / rho;
  double uy = ((f1234.x * y1234.x + f1234.y * y1234.y) +
               (f1234.z * y1234.z + f1234.w * y1234.w) +
               (f5678.x * y5678.x + f5678.y * y5678.y) +
               (f5678.z * y5678.z + f5678.w * y5678.w)) / rho;

  out[0] = rho;
  out[1] = ux;
  out[2] = uy;
  // Output the struct field values the kernel actually sees
  out[3] = dirX.s0; out[4] = dirX.s1; out[5] = dirX.s2; out[6] = dirX.s3;
  out[7] = dirX.s4; out[8] = dirX.s5; out[9] = dirX.s6; out[10] = dirX.s7;
  out[11] = dirY.s0; out[12] = dirY.s1; out[13] = dirY.s2; out[14] = dirY.s3;
  out[15] = dirY.s4; out[16] = dirY.s5; out[17] = dirY.s6; out[18] = dirY.s7;
}

int main() {
  // Direction vectors from the fluidSim benchmark (LBM D2Q9)
  double8 dirX, dirY;
  dirX.s0 = 1.0;  dirX.s1 = 0.0;  dirX.s2 = -1.0; dirX.s3 = 0.0;
  dirX.s4 = 1.0;  dirX.s5 = -1.0; dirX.s6 = -1.0; dirX.s7 = 1.0;
  dirY.s0 = 0.0;  dirY.s1 = 1.0;  dirY.s2 = 0.0;  dirY.s3 = -1.0;
  dirY.s4 = 1.0;  dirY.s5 = 1.0;  dirY.s6 = -1.0; dirY.s7 = -1.0;

  double *d_out, h_out[20];
  double *d_weight;
  hipMalloc(&d_out, 20 * sizeof(double));
  hipMalloc(&d_weight, 9 * sizeof(double));

  double w[9] = {4.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9,
                 1.0/36, 1.0/36, 1.0/36, 1.0/36};
  hipMemcpy(d_weight, w, 9 * sizeof(double), hipMemcpyHostToDevice);

  double4 f1234 = make_double4(1.0, 1.0, 1.0, 1.0);
  double4 f5678 = make_double4(0.5, 0.5, 0.5, 0.5);

  lbmKernel<<<1, 1>>>(d_out, dirX, dirY, d_weight, 1.2, 5.0, f1234, f5678);
  hipMemcpy(h_out, d_out, 20 * sizeof(double), hipMemcpyDeviceToHost);

  double expX[] = {1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0};
  double expY[] = {0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0};

  bool pass = true;
  for (int i = 0; i < 8; i++) {
    if (h_out[3 + i] != expX[i]) {
      printf("dirX.s%d = %g, expected %g\n", i, h_out[3 + i], expX[i]);
      pass = false;
    }
    if (h_out[11 + i] != expY[i]) {
      printf("dirY.s%d = %g, expected %g\n", i, h_out[11 + i], expY[i]);
      pass = false;
    }
  }

  printf("%s\n", pass ? "PASS" : "FAIL");

  hipFree(d_out);
  hipFree(d_weight);
  return pass ? 0 : 1;
}
