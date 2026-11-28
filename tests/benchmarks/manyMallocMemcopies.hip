#include <hip/hip_runtime.h>
#include <hip/hip_interop.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

// This benchmark guards a chipStar property: that chipStar does not scan its
// own allocation list on the copy path. It measures total hipMemcpy time,
// which is only a proxy for that, and the proxy breaks where the vendor driver
// itself scans a list of live allocations per copy. chipStar hands one
// clSVMAlloc to the driver per hipMalloc when coarse-grain SVM is the strategy
// in use, and on Mali-G52 that makes a 192 byte copy cost 1.3 ms once 15000
// allocations are live, all of it inside libmali. See
// https://github.com/CHIP-SPV/chipStar/issues/1322.
//
// So skip where the measurement cannot speak about chipStar. Ask the runtime
// which strategy it settled on rather than reconstructing that from device
// capabilities: the selection also depends on how chipStar was built and on
// CHIP_OCL_USE_ALLOC_STRATEGY, and a reconstruction has to track every one of
// those rules to stay true.
// HAVE_OPENCL, not __has_include: the header is on the include path of every
// build, since this target inherits CHIP's own include directories, but the
// backend it declares is compiled only when CMake enabled it. Guarding on the
// header would put a dynamic_cast to CHIPContextOpenCL into a Level Zero only
// build, whose RTTI is not there to link against. src/backend/backends.hh
// gates on the same flag.
#include <chipStarConfig.hh>
#ifdef HAVE_OPENCL
#include "backend/OpenCL/CHIPBackendOpenCL.hh"
#endif

// True when every hipMalloc becomes one allocation the driver tracks itself.
static bool allocationsAreCoarseGrainSVM() {
#ifdef HAVE_OPENCL
  // Only meaningful once the runtime has initialised and chosen a strategy.
  if (hipFree(nullptr) != hipSuccess)
    return false;
  auto *Ctx = dynamic_cast<CHIPContextOpenCL *>(
      Backend->getActiveDevice()->getContext());
  return Ctx && Ctx->getAllocStrategy() == AllocationStrategy::CoarseGrainSVM;
#else
  return false; // no OpenCL backend compiled in, so no coarse-grain SVM path
#endif
}

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
    if (allocationsAreCoarseGrainSVM()) {
        printf("HIP_SKIP_THIS_TEST: this build allocates through coarse-grained "
               "SVM, so every allocation is one the vendor driver tracks itself "
               "and the copy time measured here is not chipStar's to hold flat "
               "(chipStar issue #1322)\n");
        return 0;
    }

    const int    ITERS     = 15000;
    const int    FEW       = 1000;
    const int    WINDOW    = 100;
    const size_t BUF_SIZE  = 192;
    char host_buf[BUF_SIZE];
    std::vector<void *> dev_ptrs(ITERS, nullptr);

    // The guarded property: a memcpy must not get slower as the number of
    // live allocations grows. A linear scan over the allocation list on the
    // copy path (the regression PR #1241 fixed) shows up here as a ratio
    // that grows with the number of live allocations.
    //
    // The timed copy goes into ONE scratch buffer allocated up front, not
    // into a buffer just created: on Intel dGPUs the driver charges a
    // one-off first-touch cost to the first copy into every new allocation,
    // which is driver behaviour, not the regression this test catches.
    void *scratch = nullptr;
    CHECK(hipMalloc(&scratch, BUF_SIZE));
    CHECK(hipMemcpy(scratch, host_buf, BUF_SIZE, hipMemcpyHostToDevice));

    // Median time of WINDOW copies; medians so one descheduled copy does not decide.
    auto medianCopyNs = [&](double &Out) -> int {
      std::vector<double> T(WINDOW);
      for (int i = 0; i < WINDOW; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        CHECK(hipMemcpy(scratch, host_buf, BUF_SIZE, hipMemcpyHostToDevice));
        auto t1 = std::chrono::steady_clock::now();
        T[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
      }
      std::sort(T.begin(), T.end());
      Out = T[WINDOW / 2];
      return 0;
    };

    for (int i = 0; i < ITERS; ++i) {
      if (i % 1000 == 0) printf("iteration %d\n", i);
      CHECK(hipMalloc(&dev_ptrs[i], BUF_SIZE));
      CHECK(hipMemcpy(scratch, host_buf, BUF_SIZE, hipMemcpyHostToDevice));
    }

    // Both windows are taken back to back, so a change in host load between
    // them cannot read as a slowdown.
    double late = 0, early = 0;
    if (medianCopyNs(late))
      return 1;
    for (int i = FEW; i < ITERS; ++i)
      CHECK(hipFree(dev_ptrs[i]));
    if (medianCopyNs(early))
      return 1;
    for (int i = 0; i < FEW; ++i)
      CHECK(hipFree(dev_ptrs[i]));
    CHECK(hipFree(scratch));

    if (late / early > 2) {
      printf("Fail, the median copy with %d live allocations was %lf ns and "
             "with %d was %lf ns. Copies must not slow down as live "
             "allocations accumulate\n",
             ITERS, late, FEW, early);
      return 1;
    }
    printf("Pass (median %lf ns with %d live, %lf ns with %d live)\n", late, ITERS, early, FEW);
    return 0;
}
