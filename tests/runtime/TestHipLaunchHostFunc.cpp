#include <hip/hip_runtime.h>
#include <iostream>
#include <atomic>
#include <cassert>

std::atomic<int> hostFuncCallCount{0};
std::atomic<int> executionOrder{0};

void testHostFunc(void* userData) {
  int* order = static_cast<int*>(userData);
  *order = executionOrder.fetch_add(1) + 1;
  hostFuncCallCount.fetch_add(1);
}

__global__ void dummyKernel(int* data) {
  if (threadIdx.x == 0) {
    *data = 42;
  }
}

int testNullStream() {
  hostFuncCallCount = 0;
  executionOrder = 0;
  
  int order1 = 0, order2 = 0;
  int* d_data;
  int h_data = 0;
  
  if (hipMalloc(&d_data, sizeof(int)) != hipSuccess) {
    std::cerr << "FAIL: hipMalloc failed" << std::endl;
    return 1;
  }
  
  // Launch kernel on default stream (null stream)
  dummyKernel<<<1, 1, 0, nullptr>>>(d_data);
  
  // Launch host function on default stream (null stream)
  if (hipLaunchHostFunc(nullptr, testHostFunc, &order1) != hipSuccess) {
    std::cerr << "FAIL: hipLaunchHostFunc with null stream failed" << std::endl;
    hipFree(d_data);
    return 1;
  }
  
  // Launch another host function
  if (hipLaunchHostFunc(nullptr, testHostFunc, &order2) != hipSuccess) {
    std::cerr << "FAIL: second hipLaunchHostFunc with null stream failed" << std::endl;
    hipFree(d_data);
    return 1;
  }
  
  // Synchronize to ensure all work completes
  if (hipDeviceSynchronize() != hipSuccess) {
    std::cerr << "FAIL: hipDeviceSynchronize failed" << std::endl;
    hipFree(d_data);
    return 1;
  }
  
  // Verify host functions were called
  if (hostFuncCallCount.load() != 2) {
    std::cerr << "FAIL: Expected 2 host function calls, got " << hostFuncCallCount.load() << std::endl;
    hipFree(d_data);
    return 1;
  }
  
  // Verify execution order (host functions execute after kernel, in order)
  if (order1 != 1 || order2 != 2) {
    std::cerr << "FAIL: Execution order incorrect. order1=" << order1 << ", order2=" << order2 << std::endl;
    hipFree(d_data);
    return 1;
  }
  
  // Verify kernel completed
  if (hipMemcpy(&h_data, d_data, sizeof(int), hipMemcpyDeviceToHost) != hipSuccess) {
    std::cerr << "FAIL: hipMemcpy failed" << std::endl;
    hipFree(d_data);
    return 1;
  }
  
  if (h_data != 42) {
    std::cerr << "FAIL: Kernel did not execute correctly. Expected 42, got " << h_data << std::endl;
    hipFree(d_data);
    return 1;
  }
  
  hipFree(d_data);
  return 0;
}

int testExplicitStream() {
  hostFuncCallCount = 0;
  executionOrder = 0;
  
  int order1 = 0;
  hipStream_t stream;
  
  if (hipStreamCreate(&stream) != hipSuccess) {
    std::cerr << "FAIL: hipStreamCreate failed" << std::endl;
    return 1;
  }
  
  // Launch host function on explicit stream
  if (hipLaunchHostFunc(stream, testHostFunc, &order1) != hipSuccess) {
    std::cerr << "FAIL: hipLaunchHostFunc with explicit stream failed" << std::endl;
    hipStreamDestroy(stream);
    return 1;
  }
  
  // Synchronize stream
  if (hipStreamSynchronize(stream) != hipSuccess) {
    std::cerr << "FAIL: hipStreamSynchronize failed" << std::endl;
    hipStreamDestroy(stream);
    return 1;
  }
  
  // Verify host function was called
  if (hostFuncCallCount.load() != 1) {
    std::cerr << "FAIL: Expected 1 host function call, got " << hostFuncCallCount.load() << std::endl;
    hipStreamDestroy(stream);
    return 1;
  }
  
  if (order1 != 1) {
    std::cerr << "FAIL: Host function execution order incorrect. order1=" << order1 << std::endl;
    hipStreamDestroy(stream);
    return 1;
  }
  
  hipStreamDestroy(stream);
  return 0;
}

int main() {
  int result = 0;
  
  result |= testNullStream();
  result |= testExplicitStream();
  
  if (result == 0) {
    std::cout << "PASS" << std::endl;
    return 0;
  } else {
    std::cout << "FAIL" << std::endl;
    return 1;
  }
}
