#include <iostream>
#include <thread>
#include "hip/hip_runtime.h"

static constexpr int numThreads = 1000;

void threadFunction(int threadId) {
  int src = threadId; // Source data unique to each thread
  int dst;

  // Use hipStreamPerThread to get a per-thread default stream
  hipMemcpyAsync(&dst, &src, sizeof(int), hipMemcpyHostToHost,
                 hipStreamPerThread);
  // std::cout << "Thread " << threadId << " completed memcpy." << std::endl;
}

int main() {
  std::thread threads[numThreads];

  for (int i = 0; i < numThreads; ++i) {
    threads[i] = std::thread(threadFunction, i);
  }

  // Main thread exits without waiting for the threads to finish
  std::cout << "Main thread exiting." << std::endl;

  // Threads are detached, allowing them to continue independently
  for (int i = 0; i < numThreads; ++i) {
    threads[i].detach();
  }

  return 0;
}
