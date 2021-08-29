
#include <iostream>
#include <random>
#include <functional>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cfloat>

// hip header file
#include "hip/hip_runtime.h"

#define SEED 19284975223

// Defined = use the MatMul with shared memory, not defined = use simplest possible MatMul
#define MM_SHARED

// Defined = use FMA instruction
//#define USE_FMA

// the required shared memory is (2 * 4 * THREADS_PER_BLOCK * THREADS_PER_BLOCK) bytes
#define THREADS_PER_BLOCK 16

// configure matrix size here. Must be power of 4 at least 64
#define WIDTH 1024
#define NUM (WIDTH * WIDTH)

// how many times to run the matmul kernel
#define ITERS 30

#define ERR_CHECK_2 \
  do { \
  err = hipGetLastError(); \
    if (err != hipSuccess) { \
      std::cerr << "HIP API error\n"; \
      return -1; \
    } \
  } while (0)


#define ERR_CHECK \
  do { \
    if (err != hipSuccess) { \
      std::cerr << "HIP API error\n"; \
      return -1; \
    } \
  } while (0)


/*****************************************************************************/

#ifndef MM_SHARED

// Simple version, myGEMM2
__global__ void
gpuMatrixMul (const float * __restrict A, const float * __restrict B, float * __restrict C,
         uint M, uint N, uint K)

{
  // Thread identifiers
  const uint globalRow = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; // Row ID of C (0..M)
  const uint globalCol = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y; // Col ID of C (0..N)

  // Compute a single element (loop over K)
  float acc = 0.0f;
  for (uint k = 0; k < K; k++)
    {
#ifdef USE_FMA
      acc = fmaf (A[k * M + globalRow], B[globalCol * K + k], acc);
#else
      acc += A[k * M + globalRow] * B[globalCol * K + k];
#endif
    }

  // Store the result
  C[globalCol * M + globalRow] = acc;
}


#else
#define TS THREADS_PER_BLOCK
/* work per thread */
#define WPT (THREADS_PER_BLOCK / 4)

// TS/WPT == RTS
#define RTS 4

// Tiled and coalesced version, myGEMM4
__global__ void
gpuMatrixMul (const float * __restrict A,
              const float * __restrict B,
              float * __restrict C,
              uint M, uint N, uint K)
{

  // Thread identifiers
  const uint row = hipThreadIdx_x; // Local row ID (max: TS)
  const uint col = hipThreadIdx_y; // Local col ID (max: TS/WPT == RTS)
  const uint globalRow = TS * hipBlockIdx_x + row; // Row ID of C (0..M)
  const uint globalCol = TS * hipBlockIdx_y + col; // Col ID of C (0..N)

  // Local memory to fit a tile of TS*TS elements of A and B
  __shared__ float Asub[TS][TS];
  __shared__ float Bsub[TS][TS];

  // Initialise the accumulation registers
  float acc[WPT];
  for (uint w = 0; w < WPT; w++)
    {
      acc[w] = 0.0f;
    }

  // Loop over all tiles
  const uint numTiles = K / TS;
  for (uint t = 0; t < numTiles; t++)
    {

      // Load one tile of A and B into local memory
      for (uint w = 0; w < WPT; w++)
        {
          const uint tiledRow = TS * t + row;
          const uint tiledCol = TS * t + col;
          Asub[col + w * RTS][row] = A[(tiledCol + w * RTS) * M + globalRow];
          Bsub[col + w * RTS][row] = B[(globalCol + w * RTS) * K + tiledRow];
        }

      // Synchronise to make sure the tile is loaded
      __syncthreads();

      // Perform the computation for a single tile
      for (uint k = 0; k < TS; k++)
        {
          for (uint w = 0; w < WPT; w++)
            {
#ifdef USE_FMA
              acc[w] = fmaf (Asub[k][row], Bsub[col + w * RTS][k], acc[w]);
#else
              acc[w] += Asub[k][row] * Bsub[col + w * RTS][k];
#endif
            }
        }

      // Synchronise before loading the next tile
      __syncthreads();
    }

  // Store the final results in C
  for (uint w = 0; w < WPT; w++)
    {
      C[(globalCol + w * RTS) * M + globalRow] = acc[w];
    }
}
#endif

/*****************************************************************************/


// CPU implementation of matrix transpose
void matrixMultiplyCPUReference(const float * __restrict A,
                                const float * __restrict B,
                                float * __restrict C) {
  for (uint i = 0; i < WIDTH; i++) {
    for (uint j = 0; j < WIDTH; j++) {
          float acc = 0.0f;
          for (uint k = 0; k < WIDTH; k++) {
#ifdef USE_FMA
            acc = __builtin_fmaf (A[k*WIDTH + j], B[i*WIDTH + k], acc);
#else
            acc += B[i*WIDTH + k] * A[k*WIDTH + j];
#endif
          }
          C[i*WIDTH + j] = acc;
        }
    }
}

/*****************************************************************************/


int main() {

    hipError_t err;
    std::mt19937 gen(SEED);
    auto rnd = std::bind(std::uniform_real_distribution<float>{100.0f, 1000.0f}, gen);

    float* Matrix1;
    float* Matrix2;
    float* MultiplyMatrix;
    float* cpuMultiplyMatrix;

    float* gpuMatrix1;
    float* gpuMatrix2;
    float* gpuMultiplyMatrix;

    hipDeviceProp_t devProp;
    err = hipGetDeviceProperties(&devProp, 0);
    ERR_CHECK;

    std::cout << "Device name " << devProp.name << std::endl;

    hipEvent_t events[ITERS*2];
    for (uint w = 0; w < (ITERS*2); w++) {
      err = hipEventCreate(&events[w]);
      ERR_CHECK;
    }

    size_t i, j;

    Matrix1 = new float [NUM];
    Matrix2 = new float [NUM];
    MultiplyMatrix = new float [NUM];
    cpuMultiplyMatrix = new float [NUM];

    // initialize the input data
    for (i = 0; i < NUM; i++) {
        Matrix1[i] = rnd();
        Matrix2[i] = rnd();
    }

    float minMs,tempMs;

    // allocate the memory on the device side
    err = hipMalloc((void**)&gpuMatrix1, NUM * sizeof(float));
    ERR_CHECK;
    err = hipMalloc((void**)&gpuMatrix2, NUM * sizeof(float));
    ERR_CHECK;
    err = hipMalloc((void**)&gpuMultiplyMatrix, NUM * sizeof(float));
    ERR_CHECK;

    auto timeGPU1 = std::chrono::high_resolution_clock::now();

    // Memory transfer from host to device
    err = hipMemcpy(gpuMatrix1, Matrix1, NUM * sizeof(float), hipMemcpyHostToDevice);
    ERR_CHECK;

    err = hipMemcpy(gpuMatrix2, Matrix2, NUM * sizeof(float), hipMemcpyHostToDevice);
    ERR_CHECK;

    for (i = 0; i < ITERS; ++i) {
      err = hipEventRecord(events[i*2], NULL);
      ERR_CHECK;
      // Lauching kernel from host
      hipLaunchKernelGGL(gpuMatrixMul,
#ifndef MM_SHARED
                       dim3(WIDTH / THREADS_PER_BLOCK, WIDTH / THREADS_PER_BLOCK),
                       dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK),
#else
                       dim3((WIDTH / THREADS_PER_BLOCK), (WIDTH / THREADS_PER_BLOCK)),
                       dim3(THREADS_PER_BLOCK, 4),
#endif
                       0, 0,
                       gpuMatrix1, gpuMatrix2, gpuMultiplyMatrix, WIDTH, WIDTH, WIDTH);
      ERR_CHECK_2;
      err = hipEventRecord(events[i*2+1], NULL);
      ERR_CHECK;
    }

    // Memory transfer from device to host
    err = hipMemcpy(MultiplyMatrix, gpuMultiplyMatrix, NUM * sizeof(float), hipMemcpyDeviceToHost);
    ERR_CHECK;

    err = hipDeviceSynchronize();
    ERR_CHECK;

    auto timeGPU2 = std::chrono::high_resolution_clock::now();

    std::cout << "Running " << ITERS << " iterations \n";

    minMs = 1e30f;
    for (i = 0; i < ITERS; ++i) {
        err = hipEventElapsedTime(&tempMs, events[i*2], events[i*2+1]);
        ERR_CHECK;
        std::cout << "hipLaunchKernel " << i << " time taken: " << tempMs << "\n";
        if (tempMs < minMs) minMs = tempMs;
    }

    std::cout << "hipLaunchKernel BEST TIME: " << minMs << "\n";

    for (uint w = 0; w < (ITERS*2); w++) {
      err = hipEventDestroy(events[w]);
      ERR_CHECK;
    }

    std::chrono::duration<double, std::milli> gpu_fp_ms = timeGPU2 - timeGPU1;
    std::cout << "GPU real time taken(ms): " <<  gpu_fp_ms.count() << "\n";

    auto time1 = std::chrono::high_resolution_clock::now();
    // CPU MatrixTranspose computation
    matrixMultiplyCPUReference(Matrix1, Matrix2, cpuMultiplyMatrix);
    auto time2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = time2 - time1;
    std::cout << "matrixMultiplyCPUReference time taken(ms): " <<  fp_ms.count() << "\n";

    // verify the results
    size_t errors = 0;
    float eps = FLT_EPSILON * 4;
    for (i = 0; i < WIDTH; i++) {
        for (j = 0; j < WIDTH; j++) {
          float cpu = cpuMultiplyMatrix[i*WIDTH+j];
          float gpu = MultiplyMatrix[i*WIDTH+j];
          float diff = std::fabs(gpu - cpu);
          if (diff > std::max(std::fabs(cpu), std::fabs(gpu)) * eps) {
            errors++;
            std::cout << "E[" << i << "][" << j << "]: M1 "
                      << Matrix1[i*WIDTH+j] << " M2 " << Matrix1[i*WIDTH+j]
                      << " CPU: " << cpu << " GPU: "
                      << gpu << " ERROR: " << diff << "\n";
          }
        }
    }

    // free the resources on device side
    err = hipFree(gpuMatrix1);
    ERR_CHECK;
    err = hipFree(gpuMatrix2);
    ERR_CHECK;
    err = hipFree(gpuMultiplyMatrix);
    ERR_CHECK;

    // free the resources on host side
    delete [] Matrix1;
    delete [] Matrix2;
    delete [] MultiplyMatrix;
    delete [] cpuMultiplyMatrix;

    if (errors != 0) {
      std::cout << "Verification FAILED: " << errors << "  errors\n";
      return 1;
    } else {
      std::cout << "Verification PASSED!\n";
      return 0;
    }
}
