#include <hip/hip_runtime.h>
#include <iostream>

#include "CHIPBackend.hh"

__global__ void dummyKernel() {
  // no-op
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

  // Launch a dummy kernel in streamA and record an event
  dummyKernel<<<1, 1, 0, streamA>>>();
  hipEventRecord(streamAKernelEvent, streamA);

  // Get the chipStar event from the HIP event handle
  auto *chipEventA = static_cast<chipstar::Event *>(streamAKernelEvent);

  // StreamB waits on the event from streamA
  hipStreamWaitEvent(streamB, streamAKernelEvent, 0);

  // Get the chipStar queue corresponding to streamB
  auto *chipQueueB = Backend->findQueue(static_cast<chipstar::Queue *>(streamB));
  
  // The last event in streamB should be the barrier we just created
  auto lastEventB = chipQueueB->getLastEvent();
  
  bool foundCorrectDependency = false;
  
  if (lastEventB) {
    // Check if the barrier event has the correct dependency
    for (const auto &dep : lastEventB->DependsOnList) {
      if (dep.get() == chipEventA) {
        foundCorrectDependency = true;
        break;
      }
    }
  }

  std::cout << (foundCorrectDependency ? "PASS: barrier correctly depends on foreign event" 
                                       : "FAIL: barrier missing cross-stream dependency") << std::endl;

  // Cleanup
  hipStreamSynchronize(streamA);
  hipStreamSynchronize(streamB);

  hipEventDestroy(streamAKernelEvent);
  hipStreamDestroy(streamA);
  hipStreamDestroy(streamB);
  return foundCorrectDependency ? 0 : 1;
} 