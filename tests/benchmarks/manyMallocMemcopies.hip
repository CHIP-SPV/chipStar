#include <hip/hip_runtime.h>
#include <algorithm>
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
    
    // Compare the median of a window of early iterations against the median of
    // a window of late ones. The point of this test is that a memcpy must not
    // get slower as the number of live allocations grows -- a linear scan over
    // the allocation list would show up here -- and medians over a window
    // measure that just as well as endpoints do.
    //
    // Comparing single samples (samples[ITERS-1] against samples[1]) does not
    // work: each is one 192 byte copy timed at nanosecond resolution, so one
    // descheduled iteration at either end decides the result. That made this
    // test fail intermittently on a loaded CI machine while passing on the
    // other backends of the same run.
    const int WINDOW = 100;
    auto medianOf = [&](int First, int Last) -> double {
      std::vector<double> Window(samples + First, samples + Last);
      std::sort(Window.begin(), Window.end());
      return Window[Window.size() / 2];
    };

    // Skip index 0: the first copy pays one-off setup costs.
    double early = medianOf(1, 1 + WINDOW);
    double late = medianOf(ITERS - WINDOW, ITERS);

    if (late / early > 2) {
      printf("Fail, the median of the last %d iterations was %lf ns and of the "
             "first %d was %lf ns. They should be close\n",
             WINDOW, late, WINDOW, early);
      return 1;
    }
    printf("Pass (early median %lf ns, late median %lf ns)\n", early, late);
    return 0;
}
