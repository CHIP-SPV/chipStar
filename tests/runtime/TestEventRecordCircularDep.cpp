#include <hip/hip_runtime.h>
#include <iostream>

void test(int i) {
    std::cout << "Test " << i << std::endl;
    hipInit(0);
    hipSetDevice(0);
    
    hipStream_t s1, s2;
    hipStreamCreate(&s1);
    hipStreamCreate(&s2);
    
    hipEvent_t e1, e2;
    hipEventCreate(&e1);
    hipEventCreate(&e2);
    
    // Create circular dependency
    hipEventRecord(e1, s1);
    hipStreamWaitEvent(s2, e1, 0);
    hipEventRecord(e2, s2);
    hipStreamWaitEvent(s1, e2, 0);
    hipEventRecord(e1, s1);  // Reuse triggers issue
}

int main() {
    for (int i = 0; i < 5000; i++)
      test(i);

    std::cout << "PASSED" << std::endl;
    return 0;
} 