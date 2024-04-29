// Check for the ability of enqueueing a lot of async operations.
// If last event dependency chain is mismanaged, this will create
// an excessively long linked list of events resulting in a stack overflow.
#include <hip/hip_runtime.h>
#include <iostream>

#define CHECK(cmd)                                                             \
  {                                                                            \
    hipError_t error = cmd;                                                    \
    if (error != hipSuccess) {                                                 \
      std::cerr << "Error: " << hipGetErrorString(error) << std::endl;         \
      exit(-1);                                                                \
    }                                                                          \
  }

int main(int argc, char *argv[]) {
  const size_t size = 1; // Size of memory copy in bytes
  const int N = 500000;

  char *hostData;
  char *deviceData;

  // Allocate host memory
  CHECK(hipHostMalloc(&hostData, size, hipHostMallocDefault));

  // Allocate device memory
  CHECK(hipMalloc(&deviceData, size));

  // Create a stream for asynchronous operations
  hipStream_t stream;
  CHECK(hipStreamCreate(&stream));

  // Initialize host data
  *hostData = 'x';

  // Perform N asynchronous memory copies from host to device
  for (int i = 0; i < N; ++i)
    CHECK(hipMemcpyAsync(deviceData, hostData, size, hipMemcpyHostToDevice,
                         stream));

  // Wait for all operations in the stream to complete
  CHECK(hipStreamSynchronize(stream));

  // Cleanup
  CHECK(hipStreamDestroy(stream));
  CHECK(hipFree(deviceData));
  CHECK(hipHostFree(hostData));

  std::cout << "Completed " << N << " iterations of asynchronous memory copies."
            << std::endl;
  std::cout << "PASSED!" << std::endl;

  return 0;
}
