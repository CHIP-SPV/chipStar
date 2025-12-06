#include <hip/hip_runtime.h>
#include <iostream>

#include "CHIPBackend.hh"

// Simple compute-bound kernel - no arguments, no device memory access
__global__ void slowKernel() {
  volatile int x = 0;
  for (int i = 0; i < 1000000; i++) {
    x += i;
  }
}

int main() {
  hipStream_t streamA = nullptr, streamB = nullptr;
  hipEvent_t streamAKernelEvent = nullptr;

  if (hipStreamCreate(&streamA) != hipSuccess ||
      hipStreamCreate(&streamB) != hipSuccess ||
      hipEventCreate(&streamAKernelEvent) != hipSuccess) {
    std::cerr << "FAIL: setup failed" << std::endl;
    return 1;
  }

  // Launch a slow kernel in streamA and record an event
  slowKernel<<<1, 1, 0, streamA>>>();
  hipEventRecord(streamAKernelEvent, streamA);

  auto *chipEventA = static_cast<chipstar::Event *>(streamAKernelEvent);

  hipStreamWaitEvent(streamB, streamAKernelEvent, 0);

  auto *chipQueueB = Backend->findQueue(static_cast<chipstar::Queue *>(streamB));
  auto lastEventB = chipQueueB->getLastEvent();
  
  bool foundCorrectDependency = false;
  
  if (lastEventB) {
    LOCK(Backend->EventsMtx);
    std::lock_guard<std::mutex> lockB(lastEventB->DependsOnListMtx);
    std::lock_guard<std::mutex> lockA(chipEventA->DependsOnListMtx);
    
    for (const auto &dep : lastEventB->DependsOnList) {
      if (dep.get() == chipEventA) {
        foundCorrectDependency = true;
        break;
      }
    }
    
    if (!foundCorrectDependency && !chipEventA->DependsOnList.empty()) {
      for (const auto &userEventDep : chipEventA->DependsOnList) {
        for (const auto &barrierDep : lastEventB->DependsOnList) {
          if (userEventDep.get() == barrierDep.get()) {
            foundCorrectDependency = true;
            break;
          }
        }
        if (foundCorrectDependency) break;
      }
    }
  }

  std::cout << (foundCorrectDependency ? "PASS: barrier correctly depends on foreign event" 
                                       : "FAIL: barrier missing cross-stream dependency") << std::endl;

  hipStreamSynchronize(streamA);
  hipStreamSynchronize(streamB);

  hipEventDestroy(streamAKernelEvent);
  hipStreamDestroy(streamA);
  hipStreamDestroy(streamB);
  return foundCorrectDependency ? 0 : 1;
}