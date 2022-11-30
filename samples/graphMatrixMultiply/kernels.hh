#include <hip/hip_runtime.h>
#include <chrono>

// the required shared memory is (2 * 4 * THREADS_PER_BLOCK * THREADS_PER_BLOCK)
#define THREADS_PER_BLOCK 16

// configure matrix size here. Must be power of 4 at least 64
#define WIDTH 1024
#define NUM (WIDTH * WIDTH)

hipError_t err;

// Simple version, myGEMM2
__global__ void gpuMatrixMul(const float *__restrict A,
                             const float *__restrict B, float *__restrict C,
                             uint M, uint N, uint K)

{
  // Thread identifiers
  const uint globalRow =
      hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; // Row ID of C (0..M)
  const uint globalCol =
      hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y; // Col ID of C (0..N)

  // Compute a single element (loop over K)
  float acc = 0.0f;
  for (uint k = 0; k < K; k++) {
    acc += A[k * M + globalRow] * B[globalCol * K + k];
  }

  // Store the result
  C[globalCol * M + globalRow] = acc;
}

// CPU implementation of matrix transpose
void matrixMultiplyCPUReference(const float *__restrict A,
                                const float *__restrict B,
                                float *__restrict C) {
  for (uint i = 0; i < WIDTH; i++) {
    for (uint j = 0; j < WIDTH; j++) {
      float acc = 0.0f;
      for (uint k = 0; k < WIDTH; k++) {
        acc += B[i * WIDTH + k] * A[k * WIDTH + j];
      }
      C[i * WIDTH + j] = acc;
    }
  }
}

float matrixMultiplyBasic(float *gpuMatrix1, float *gpuMatrix2, float *Matrix1,
                          float *Matrix2, float *gpuMultiplyMatrix,
                          float *MultiplyMatrix, hipEvent_t *events,
                          int ITERS) {
  auto timeGPU1 = std::chrono::high_resolution_clock::now();

  // Memory transfer from host to device
  err = hipMemcpy(gpuMatrix1, Matrix1, NUM * sizeof(float),
                  hipMemcpyHostToDevice);
  ERR_CHECK;

  err = hipMemcpy(gpuMatrix2, Matrix2, NUM * sizeof(float),
                  hipMemcpyHostToDevice);
  ERR_CHECK;

  for (int i = 0; i < ITERS; ++i) {
    err = hipEventRecord(events[i * 2], NULL);
    ERR_CHECK;
    // Lauching kernel from host
    hipLaunchKernelGGL(
        gpuMatrixMul,
        dim3(WIDTH / THREADS_PER_BLOCK, WIDTH / THREADS_PER_BLOCK),
        dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK), 0, 0, gpuMatrix1,
        gpuMatrix2, gpuMultiplyMatrix, WIDTH, WIDTH, WIDTH);
    err = hipEventRecord(events[i * 2 + 1], NULL);
    ERR_CHECK;
  }

  // Memory transfer from device to host
  err = hipMemcpy(MultiplyMatrix, gpuMultiplyMatrix, NUM * sizeof(float),
                  hipMemcpyDeviceToHost);
  ERR_CHECK;

  err = hipDeviceSynchronize();
  ERR_CHECK;

  auto timeGPU2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> gpu_fp_ms = timeGPU2 - timeGPU1;
  return gpu_fp_ms.count();
}

float matrixMultiplyGraphBasic(float *gpuMatrix1, float *gpuMatrix2,
                               float *Matrix1, float *Matrix2,
                               float *gpuMultiplyMatrix, float *MultiplyMatrix,
                               hipEvent_t *events, int ITERS) {
  auto timeGPU1 = std::chrono::high_resolution_clock::now();

  hipGraph_t graph;

  err = hipGraphCreate(&graph, 0);
  ERR_CHECK;

  // Memory transfer from host to device
  // err = hipMemcpy(gpuMatrix1, Matrix1, NUM * sizeof(float),
  // hipMemcpyHostToDevice);
  // ERR_CHECK;
  hipGraphNode_t memcpyHostToDev1node;
  err = hipGraphAddMemcpyNode1D(&memcpyHostToDev1node, graph, nullptr, 0,
                                gpuMatrix1, Matrix1, NUM * sizeof(float),
                                hipMemcpyHostToDevice);
  ERR_CHECK;

  // err = hipMemcpy(gpuMatrix2, Matrix2, NUM * sizeof(float),
  //                 hipMemcpyHostToDevice);
  // ERR_CHECK;
  hipGraphNode_t memcpyHostToDev2node;
  err = hipGraphAddMemcpyNode1D(&memcpyHostToDev2node, graph, nullptr, 0,
                                gpuMatrix2, Matrix2, NUM * sizeof(float),
                                hipMemcpyHostToDevice);
  ERR_CHECK;

  hipGraphNode_t kernelNodes[ITERS];
  for (int i = 0; i < ITERS; ++i) {
    // TODO
    // err = hipEventRecord(events[i * 2], NULL);
    // ERR_CHECK;

    // Lauching kernel from host
    // hipLaunchKernelGGL(
    //     gpuMatrixMul,
    //     dim3(WIDTH / THREADS_PER_BLOCK, WIDTH / THREADS_PER_BLOCK),
    //     dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK), 0, 0, gpuMatrix1,
    //     gpuMatrix2, gpuMultiplyMatrix, WIDTH, WIDTH, WIDTH);

    uint width = WIDTH;
    hipKernelNodeParams kernelParams;
    kernelParams.func = (void *)gpuMatrixMul;
    kernelParams.gridDim =
        dim3(WIDTH / THREADS_PER_BLOCK, WIDTH / THREADS_PER_BLOCK);
    kernelParams.blockDim = dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    kernelParams.sharedMemBytes = 0;

    void *kernelArgs[6] = {
        (void *)&gpuMatrix1, (void *)&gpuMatrix2, (void *)&gpuMultiplyMatrix,
        (void *)&width,      (void *)&width,      (void *)&width};
    kernelParams.kernelParams = reinterpret_cast<void **>(kernelArgs);
    kernelParams.extra = nullptr;

    hipGraphNode_t kernelDependencies[2] = {memcpyHostToDev1node,
                                            memcpyHostToDev2node};

    err = hipGraphAddKernelNode(&(kernelNodes[i]), graph, kernelDependencies, 2,
                                &kernelParams);
    ERR_CHECK;
    // TODO
    // err = hipEventRecord(events[i * 2 + 1], NULL);
    // ERR_CHECK;
  }

  // Memory transfer from device to host
  // err = hipMemcpy(MultiplyMatrix, gpuMultiplyMatrix, NUM * sizeof(float),
  //                 hipMemcpyDeviceToHost);
  // ERR_CHECK;
  hipGraphNode_t memcpyDevToHostnode;
  err = hipGraphAddMemcpyNode1D(&memcpyDevToHostnode, graph, kernelNodes, ITERS,
                                MultiplyMatrix, gpuMultiplyMatrix,
                                NUM * sizeof(float), hipMemcpyDeviceToHost);
  ERR_CHECK;

  // err = hipDeviceSynchronize();
  // ERR_CHECK;

  hipGraphExec_t instance;
  err = hipGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
  ERR_CHECK;

  err = hipGraphLaunch(instance, 0);
  ERR_CHECK;

  auto timeGPU2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> gpu_fp_ms = timeGPU2 - timeGPU1;
  return gpu_fp_ms.count();
}