#pragma once
#define CEED_HIP_NUMBER_FIELDS 16
typedef float CeedScalar;
typedef int32_t CeedInt;

typedef struct {
  CeedScalar *inputs[CEED_HIP_NUMBER_FIELDS];
  CeedScalar *outputs[CEED_HIP_NUMBER_FIELDS];
} Fields_Hip;

void printFields(Fields_Hip fields, int Q) {
  printf("field.inputs[0]:\n");
  for (int i = 0; i < Q; i++) {
    printf("  %d: %f\n", i, fields.inputs[0][i]);
  }

  printf("field.inputs[1]:\n");
  for (int i = 0; i < Q; i++) {
    printf("  %d: %f\n", i, fields.inputs[1][i]);
  }

  printf("field.outputs:\n");
  for (int i = 0; i < Q; i++) {
    printf("  %d: %f\n", i, fields.outputs[0][i]);
  }

  printf("\n");
}

void initFields(Fields_Hip fields, int Q) {
  for (int i = 0; i < Q; i++) {
    fields.inputs[0][i] = i;
    fields.inputs[1][i] = i;
  }
  for (int i = 0; i < Q; i++) {
    fields.outputs[0][i] = 0;
  }
}