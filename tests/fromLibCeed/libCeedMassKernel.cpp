#include "hip/hip_runtime.h"
#define CEED_HIP_NUMBER_FIELDS 16

typedef float CeedScalar;
typedef int32_t CeedInt;
typedef struct {
  CeedScalar *inputs[CEED_HIP_NUMBER_FIELDS];
  CeedScalar *outputs[CEED_HIP_NUMBER_FIELDS];
} Fields_Hip;

inline __device__ __host__ int mass(void *ctx, const CeedInt Q,
                           const CeedScalar *const *in,
                           CeedScalar *const *out) {
  const CeedScalar *rho = in[0], *u = in[1];
  CeedScalar *v = out[0];
  for (CeedInt i = 0; i < Q; i++) {
    v[i] = rho[i] * u[i];
  }
  return 0;
}

//------------------------------------------------------------------------------
// Read from quadrature points
//------------------------------------------------------------------------------
template <int SIZE>
inline __device__ __host__ void readQuads(const CeedInt quad, const CeedInt num_qpts,
                                 const CeedScalar *d_u, CeedScalar *r_u) {
  for (CeedInt comp = 0; comp < SIZE; comp++) {
    r_u[comp] = d_u[quad + num_qpts * comp];
  }
}

//------------------------------------------------------------------------------
// Write at quadrature points
//------------------------------------------------------------------------------
template <int SIZE>
inline __device__ __host__ void writeQuads(const CeedInt quad, const CeedInt num_qpts,
                                  const CeedScalar *r_v, CeedScalar *d_v) {
  for (CeedInt comp = 0; comp < SIZE; comp++) {
    d_v[quad + num_qpts * comp] = r_v[comp];
  }
}

extern "C" __launch_bounds__(BLOCK_SIZE) __global__ 
    void CeedKernelHipRefQFunction_mass(void *ctx, CeedInt Q,
                                        Fields_Hip fields) {
  // Input fields
  const CeedInt size_input_0 = 1;
  CeedScalar input_0[size_input_0];
  const CeedInt size_input_1 = 1;
  CeedScalar input_1[size_input_1];
  const CeedScalar *inputs[2];
  inputs[0] = input_0;
  inputs[1] = input_1;

  // Output fields
  const CeedInt size_output_0 = 1;
  CeedScalar output_0[size_output_0];
  CeedScalar *outputs[1];
  outputs[0] = output_0;

  // Loop over quadrature points
  // for (CeedInt q = blockIdx.x * blockDim.x + threadIdx.x; q < Q;
  //      q += blockDim.x * gridDim.x)
       for (CeedInt q = 0; q < Q; q++)
        {
    // -- Load inputs
    readQuads<size_input_0>(q, Q, fields.inputs[0], input_0);
    readQuads<size_input_1>(q, Q, fields.inputs[1], input_1);

    // -- Call QFunction
    mass(ctx, 1, inputs, outputs);

    // -- Write outputs
    writeQuads<size_output_0>(q, Q, output_0, fields.outputs[0]);
  }
}

void CeedKernelHipRefQFunction_mass_cpu(void *ctx, CeedInt Q,
                                               Fields_Hip fields) {
  // Input fields
  const CeedInt size_input_0 = 1;
  CeedScalar input_0[size_input_0];
  const CeedInt size_input_1 = 1;
  CeedScalar input_1[size_input_1];
  const CeedScalar *inputs[2];
  inputs[0] = input_0;
  inputs[1] = input_1;

  // Output fields
  const CeedInt size_output_0 = 1;
  CeedScalar output_0[size_output_0];
  CeedScalar *outputs[1];
  outputs[0] = output_0;

  int blockIdx_x = 1;
  int blockDim_x = 256;
  // int threadIdx_x = 0;
  int gridDim_x = 1;

  // Loop over quadrature points
  for (CeedInt q = 0; q < Q; q++) {
    // -- Load inputs
    readQuads<size_input_0>(q, Q, fields.inputs[0], input_0); // r_u[comp] = d_u[quad + num_qpts * comp];
    readQuads<size_input_1>(q, Q, fields.inputs[1], input_1);

    // -- Call QFunction
    mass(ctx, 1, inputs, outputs);

    // -- Write outputs
    writeQuads<size_output_0>(q, Q, output_0, fields.outputs[0]);
  }
}

void printFields(Fields_Hip fields, int Q) {
  printf("field.inputs[0]:\n");
  for(int i = 0; i < Q; i++) {
    printf("  %d: %f\n", i, fields.inputs[0][i]);
  }

  printf("field.inputs[0]:\n");
  for(int i = 0; i < Q; i++) {
    printf("  %d: %f\n", i, fields.inputs[1][i]);
  }

  printf("field.outputs:\n");
  for(int i = 0; i < Q; i++) {
    printf("  %d: %f\n", i, fields.outputs[0][i]);
  }
}

void initFields(Fields_Hip fields) {
  printf("field.inputs:\n");
  for(int i = 0; i < CEED_HIP_NUMBER_FIELDS; i++) {
    fields.inputs[i][0] = i;
  }
  printf("field.outputs:\n");
  for(int i = 0; i < CEED_HIP_NUMBER_FIELDS; i++) {
    fields.outputs[i][0] = 1;
  }
}

int main() {
  /*
  Launching kernel CeedKernelHipRefQFunction_mass
  GridDim: <1, 1, 1> BlockDim: <256, 1, 1>
  NumArgs: 3
  Arg 0: Pointer 8 0x5555569d7250
  Arg 1: POD 4 0x7fffffffaf9c
  Arg 2: POD 256 0x5555569d7150
  */

  void *ctx = nullptr;
  CeedInt Q = 120;
  Fields_Hip fields;
  hipHostMalloc(fields.inputs, CEED_HIP_NUMBER_FIELDS * sizeof(CeedScalar), 0);
  hipHostMalloc(fields.outputs, CEED_HIP_NUMBER_FIELDS * sizeof(CeedScalar), 0);

  hipHostMalloc(&fields.inputs[0], Q * sizeof(CeedScalar), 0);
  hipHostMalloc(&fields.inputs[1], Q * sizeof(CeedScalar), 0);

  hipHostMalloc(&fields.outputs[0], Q * sizeof(CeedScalar), 0);

  for(int i = 0; i < Q; i++){
    fields.inputs[0][i] = i;
    fields.inputs[1][i] = i;
    fields.outputs[0][i] = 0;
  }

   CeedKernelHipRefQFunction_mass_cpu(ctx, Q, fields);
  float cpu_sum = 0;
  for (int i = 0; i < Q; i++) {
    cpu_sum += fields.outputs[0][i];
    fields.outputs[0][i] = 0;
  }

  hipLaunchKernelGGL(CeedKernelHipRefQFunction_mass, dim3(1), dim3(1), 0, 0,
                     ctx, Q, fields);
  hipDeviceSynchronize();
  float gpu_sum = 0;
  for (int i = 0; i < Q; i++) {
    gpu_sum += fields.outputs[0][i];
  }

  if(abs(cpu_sum - gpu_sum) > 1e-6) {
    printf("FAILED\n");
  } else {
    printf("PASSED\n");
  }
  // printFields(fields, Q);
  return 0;
}