#include <hip/hip_runtime.h>
#include "common.h"

extern "C" __launch_bounds__(256) void __global__ testFields(Fields_Hip fields, int Q) {
  printf("ENTERING KERNEL:\n");
  printf("fields.outputs[0][0]: %f\n", fields.outputs[0][0]);

  fields.outputs[0][0] = 33.33f;

  printf("EXITING KERNEL:\n");
  printf("fields.outputs[0][0]: %f\n", fields.outputs[0][0]);

  return;
}

int main() {
  const int Q = 1;
  Fields_Hip fields;
  for (int i = 0; i < CEED_HIP_NUMBER_FIELDS; i++) {
    fields.inputs[i] = NULL;
    fields.outputs[i] = NULL;
  }
#ifdef USE_HIPHOSTMALLOC
  hipHostMalloc(&fields.inputs[0], Q * sizeof(CeedScalar), 0);
  hipHostMalloc(&fields.inputs[1], Q * sizeof(CeedScalar), 0);
  hipHostMalloc(&fields.outputs[0], Q * sizeof(CeedScalar), 0);
#else
  hipMalloc(&fields.inputs[0], Q * sizeof(CeedScalar));
  hipMalloc(&fields.inputs[1], Q * sizeof(CeedScalar));
  hipMalloc(&fields.outputs[0], Q * sizeof(CeedScalar));
#endif

  hipMemset(fields.inputs[0], 0, Q * sizeof(CeedScalar));
  hipMemset(fields.inputs[1], 0, Q * sizeof(CeedScalar));
  // output vector does not get initialized after hipMalloc
  // hipMemset(fields.outputs[0], zeroes_f, Q * sizeof(CeedScalar));

  hipLaunchKernelGGL(testFields, dim3(1), dim3(1), 0, 0, fields, Q);
  float out[Q];
  hipMemcpy(out, fields.outputs[0], Q * sizeof(CeedScalar),
            hipMemcpyDeviceToHost);
  hipDeviceSynchronize();
  printf("out[0]: %f\n", out[0]);

  if(abs(out[0] - 33.33f) > 1e-6) {
    printf("FAILED\n");
  } else {
    printf("PASSED\n");
  }

}
