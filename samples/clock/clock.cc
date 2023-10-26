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
#include <hip/hip_runtime.h>
#include <chrono>

__global__ void wait_kernel(int timeout) {
  unsigned long long start = clock64();
  unsigned long long end = start;
  while (true) {
    auto diff = clock64() - start;
    if ((diff) > timeout) {
      break;
    }
  };
}

void long_wait_kernel() { wait_kernel<<<1, 1>>>(10000); }

void short_wait_kernel() { wait_kernel<<<1, 1>>>(10); }

int main() {
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  long_wait_kernel();
  hipDeviceSynchronize();
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_time_secs = end - start;
  auto time_spent_in_long_kernel = elapsed_time_secs.count();

  start = std::chrono::system_clock::now();
  short_wait_kernel();
  hipDeviceSynchronize();
  end = std::chrono::system_clock::now();
  elapsed_time_secs = end - start;
  auto time_spent_in_short_kernel = elapsed_time_secs.count();

  if (time_spent_in_short_kernel > time_spent_in_long_kernel) {
    printf("FAILED\n");
  } else {
    printf("PASSED\n");
  }
  return 0;
}