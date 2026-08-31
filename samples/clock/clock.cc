/*
 * Copyright (c) 2022-23 chipStar developers
 * Copyright (c) 2022-23 Sarbojit Sarkar <sarkar.iitr@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

// Exercises clock64() as a device side spin: a work item busy waits until
// clock64() has advanced by a tick budget, so the value has to keep advancing
// while the kernel runs, and a larger budget has to keep the kernel running
// longer. Both are checked.
//
// The spin also stops at an iteration cap, so a clock64() that stops advancing
// ends the kernel with a tick count at or below its budget, which is reported
// as a failure, instead of spinning until the test framework's timeout.
//
// Measuring "longer" is the delicate part, because the only portable clock here
// is the host's and the interval it sees is the spin plus a launch and a
// hipDeviceSynchronize(). That overhead is around 0.1 ms on an idle GPU and
// reaches tens of milliseconds when several processes share one, so:
// the large budget is sized to spend tens of milliseconds spinning, a warm up
// launch takes the module load out of the way, and each interval is measured
// several times and reduced to its minimum, since load only ever adds time. The
// short spin, which measures that overhead on its own, is sampled the most
// often because it is the cheap one and because it is what moves under load.
// Comparing one unwarmed measurement of a 10000 tick spin against one of a 10
// tick spin, as this sample used to, inverts on a busy machine
// (https://github.com/CHIP-SPV/chipStar/issues/1547). The host clock is
// steady_clock because system_clock can step backwards.
//
// Device side timing would drop the launch overhead out of the measurement and
// is not usable here: hipEventRecord plus hipEventSynchronize hangs on Mali
// under OpenCL (https://github.com/CHIP-SPV/chipStar/issues/1538), and on an
// Intel iGPU under OpenCL the two markers' CL_PROFILING_COMMAND_END values come
// back out of order, so hipEventElapsedTime reports microseconds for a kernel
// that takes milliseconds (https://github.com/CHIP-SPV/chipStar/issues/1549).

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <limits>

// Ticks each spin waits for.
static constexpr unsigned long long LongBudget = 200000;
static constexpr unsigned long long ShortBudget = 10;

// Bound on the spin's iterations, so that a clock64() which stops advancing
// leaves the loop.
static constexpr unsigned long long IterationCap = 100000000;

// How often each interval is measured before the minimum is taken.
static constexpr int LongMeasurements = 3;
static constexpr int ShortMeasurementsPerLong = 7;

// Separation required between the two spins. The long budget outlasts the short
// one by more than a factor of fifty on every device this was measured on, idle
// or loaded, so a factor of two rejects timing noise without being tight.
static constexpr double RequiredRatio = 2.0;

struct Spin {
  unsigned long long Ticks;
  double Ms;
};

__global__ void waitKernel(unsigned long long Budget, unsigned long long Cap,
                           unsigned long long *ObservedTicks) {
  unsigned long long Start = clock64();
  unsigned long long Elapsed = 0;
  for (unsigned long long I = 0; I < Cap && Elapsed <= Budget; ++I)
    Elapsed = clock64() - Start;
  *ObservedTicks = Elapsed;
}

#define RETURN_ON_ERROR(Expr)                                                  \
  do {                                                                         \
    hipError_t Err_ = (Expr);                                                  \
    if (Err_ != hipSuccess)                                                    \
      return Err_;                                                             \
  } while (0)

static hipError_t runSpin(unsigned long long Budget,
                          unsigned long long *DevTicks, Spin &Result) {
  auto Start = std::chrono::steady_clock::now();
  waitKernel<<<1, 1>>>(Budget, IterationCap, DevTicks);
  RETURN_ON_ERROR(hipGetLastError());
  RETURN_ON_ERROR(hipDeviceSynchronize());
  Result.Ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - Start)
                  .count();
  return hipMemcpy(&Result.Ticks, DevTicks, sizeof(Result.Ticks),
                   hipMemcpyDeviceToHost);
}

#define CHECK(Expr)                                                            \
  do {                                                                         \
    hipError_t Err_ = (Expr);                                                  \
    if (Err_ != hipSuccess) {                                                  \
      printf("%s returned %s\nFAILED\n", #Expr, hipGetErrorString(Err_));      \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static bool ticksAdvanced(const char *Name, const Spin &Measured,
                          unsigned long long Budget) {
  if (Measured.Ticks > Budget)
    return true;
  printf("  the %s spin left its loop after %llu ticks, which does not exceed "
         "its %llu tick budget: clock64() stopped advancing\n",
         Name, Measured.Ticks, Budget);
  return false;
}

int main() {
  unsigned long long *DevTicks = nullptr;
  CHECK(hipMalloc(&DevTicks, sizeof(*DevTicks)));

  // The first launch also loads the module, a one time cost that is not part of
  // the spin.
  Spin Warmup;
  CHECK(runSpin(ShortBudget, DevTicks, Warmup));

  bool Ok = true;
  Spin Long = {0, std::numeric_limits<double>::infinity()};
  Spin Short = Long;
  for (int I = 0; I < LongMeasurements; ++I) {
    Spin LongRun;
    CHECK(runSpin(LongBudget, DevTicks, LongRun));
    Ok = ticksAdvanced("long", LongRun, LongBudget) && Ok;
    if (LongRun.Ms < Long.Ms)
      Long = LongRun;
    for (int J = 0; J < ShortMeasurementsPerLong; ++J) {
      Spin ShortRun;
      CHECK(runSpin(ShortBudget, DevTicks, ShortRun));
      Ok = ticksAdvanced("short", ShortRun, ShortBudget) && Ok;
      if (ShortRun.Ms < Short.Ms)
        Short = ShortRun;
    }
  }

  printf("long spin:  budget %llu ticks, observed %llu, %.3f ms\n", LongBudget,
         Long.Ticks, Long.Ms);
  printf("short spin: budget %llu ticks, observed %llu, %.3f ms\n", ShortBudget,
         Short.Ticks, Short.Ms);

  if (!(Long.Ms > Short.Ms * RequiredRatio)) {
    printf("  the %llu tick spin took %.3f ms, not the expected %.1fx more "
           "than the %llu tick spin's %.3f ms\n",
           LongBudget, Long.Ms, RequiredRatio, ShortBudget, Short.Ms);
    Ok = false;
  }

  CHECK(hipFree(DevTicks));

  printf("%s\n", Ok ? "PASSED" : "FAILED");
  return Ok ? 0 : 1;
}
