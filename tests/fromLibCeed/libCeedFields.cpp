#include <hip/hip_runtime.h>
#include "common.h"

void __global__ testFields(Fields_Hip fields, int Q) {
  CeedScalar *input_0 = fields.inputs[0];
  CeedScalar *output_0 = fields.outputs[0];

  for (int i = 0; i < Q; i++) {
    output_0[i] = input_0[i];
  }

  return;
}

void testFields_cpu(Fields_Hip fields, int Q) {
  CeedScalar *input_0 = fields.inputs[0];
  CeedScalar *output_0 = fields.outputs[0];

  for (int i = 0; i < Q; i++) {
    output_0[i] = input_0[i];
  }

  return;
}

int main() {
  const int Q = 10;
  Fields_Hip fields;
  hipHostMalloc(&fields.inputs[0], Q * sizeof(CeedScalar));
  hipHostMalloc(&fields.outputs[0], Q * sizeof(CeedScalar));

  initFields(fields, Q);
  printFields(fields, Q);
  float sumCpu = 0;
  testFields_cpu(fields, Q);
  printFields(fields, Q);
  for (int i = 0; i < Q; i++) {
    sumCpu += fields.outputs[0][i];
  }

  initFields(fields, Q);
  printFields(fields, Q);
  float sumGpu = 0;
  hipLaunchKernelGGL(testFields, dim3(1), dim3(1), 0, 0, fields, Q);
  hipDeviceSynchronize();
  printFields(fields, Q);
  for (int i = 0; i < Q; i++) {
    sumGpu += fields.outputs[0][i];
  }

  if (sumCpu != sumGpu) {
    printf("FAILED\n");
  } else {
    printf("PASSED\n");
  }
}
