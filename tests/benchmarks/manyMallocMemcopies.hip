#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(cmd)                                                        \
    do {                                                                  \
        hipError_t e = (cmd);                                             \
        if (e != hipSuccess) {                                            \
            fprintf(stderr, "HIP error %s:%d: %s\n",                     \
                    __FILE__, __LINE__, hipGetErrorString(e));            \
            return 1;                                                     \
        }                                                                 \
    } while (0)

int main() {
    const int    ITERS     = 15000;
    const size_t BUF_SIZE  = 192;
    double samples[ITERS];
    char host_buf[BUF_SIZE];
    std::vector<void *> dev_ptrs(ITERS, nullptr);

    // time how long the memcopy takes for many allocated pieces
    for (int i = 0; i < ITERS; ++i) {
      if(i%1000 == 0) printf( "iteration %d\n", i);
      CHECK(hipMalloc(&dev_ptrs[i], BUF_SIZE));
      auto t0 = std::chrono::steady_clock::now();
      CHECK(hipMemcpy(dev_ptrs[i], host_buf, BUF_SIZE, hipMemcpyHostToDevice));
      auto t1 = std::chrono::steady_clock::now();
      samples[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
    }

    // free everything
    for (int i = 0; i < ITERS; ++i) {
      CHECK(hipFree(dev_ptrs[i]));
    }
    
    // starting at 1 to let the first be a warmup
    // fail if the last iteration is > double the first 
    if( samples[ITERS-1] / samples[1] > 2 ) {
      printf( "Fail, the %d th iteration was %lf and the second was %lf. They should be close\n", ITERS-1, samples[ITERS-1], samples[1]);
      return 1;
    }
    else {
      printf( "Pass\n");
      return 0;
    }
    return 0;
}
