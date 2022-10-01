/*
 * Copyright (c) 2021-22 CHIP-SPV developers
 * Copyright (c) 2021-22 Paulius Velesko <pvelesko@pglc.io>
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
#include <unistd.h>

#define CHECK(cmd)                                                             \
  {                                                                            \
    hipError_t error = cmd;                                                    \
    if (error != hipSuccess) {                                                 \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error),  \
              error, __FILE__, __LINE__);                                      \
      exit(1);                                                                 \
    }                                                                          \
  }

__global__ void addOne(int *__restrict A) {
  const uint i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  A[i] = A[i] + 1;
}

void callback_sleep2(hipStream_t stream, hipError_t status, void* user_data) {
    int *data = (int*)user_data;
    printf("callback_sleep2: Going to sleep for 2sec\n");
    sleep(2);
    *data = 2;
    printf("callback_sleep2: Exiting now\n");
}

void callback_sleep10(hipStream_t stream, hipError_t status, void* user_data) {
    int *data = (int*)user_data;
    printf("callback_sleep10: Going to sleep for 10sec\n");
    sleep(10);
    *data = 2;
    printf("callback_sleep10: Exiting now\n");
}

/*
 * Intent : Verify hipStreamQuery returns right queue status
 */
bool TestStreamSemantics_1() {
    printf("------------------------------------------------------------------\n");
    int* stream2_shared_data = nullptr;
    hipError_t status;
    hipStream_t stream;
    CHECK(hipStreamCreate(&stream));
    stream2_shared_data = (int*)malloc(sizeof(int));
    CHECK(hipStreamAddCallback(stream, callback_sleep10, stream2_shared_data, 0));
    status = hipStreamQuery(stream);
    bool testStatus = true;
    printf("%s : ", __FUNCTION__);
    if (status != hipErrorNotReady) {
        printf("%s%s%s\n","\033[0;31m", "Failed", "\033[0m");
        testStatus = false;
        //printf("Failed, queue status is %s but expected is hipErrorNotReady\n", hipGetErrorName(status));
    } else {
        printf("Passed\n");
    }

    // Wait for all tasks to be finished
    CHECK(hipDeviceSynchronize());
    // Clean-up
    CHECK(hipStreamDestroy(stream));
    free(stream2_shared_data);

    return testStatus;
}

/*
 * Intent : Verify non-blocking stream is indeed non-blocking
 */
bool TestStreamSemantics_2() {
    printf("------------------------------------------------------------------\n");
    // Init
    hipStream_t stream_non_blocking;
    CHECK(hipStreamCreateWithFlags(&stream_non_blocking, hipStreamNonBlocking));
    int* stream_shared_data = nullptr;
    stream_shared_data = (int*)malloc(sizeof(int));
    *stream_shared_data = 1;

    int *host_ptr = nullptr;
    int *dev_ptr = nullptr;
    size_t size = sizeof(int);
    CHECK(hipMalloc(&dev_ptr, size));
    host_ptr = (int*)malloc(size);

    // Push a 10sec long taks into the stream
    CHECK(hipStreamAddCallback(stream_non_blocking, callback_sleep10, stream_shared_data, 0));

    //printf("Starting task on null stream\n");
    *host_ptr = 100; // init value
    CHECK(hipMemcpy(dev_ptr, host_ptr, size, hipMemcpyDefault));
    hipLaunchKernelGGL(addOne, 1, 1, 0, 0, dev_ptr);
    CHECK(hipGetLastError());
    CHECK(hipMemcpyAsync(host_ptr, dev_ptr, size, hipMemcpyDefault));
    CHECK(hipStreamSynchronize(0));
    //printf("End of null stream task\n");fflush(stdout);

    bool testStatus = true;
    printf("%s : ", __FUNCTION__);
    if (*host_ptr == 101 && *stream_shared_data == 2) {
        testStatus = false;
        printf("%s %s %s\n","\033[0;31m", "Failed", "\033[0m");
        //printf("host_ptr = %d, stream_shared_data = %d\n", *host_ptr, *stream_shared_data);fflush(stdout);
    } else {
        printf("Passed\n");
    }
    
    // Wait for all tasks to be finished
    CHECK(hipDeviceSynchronize());

    // Clean-up
    CHECK(hipStreamDestroy(stream_non_blocking));
    CHECK(hipFree(dev_ptr));
    free(stream_shared_data);
    free(host_ptr);
    return testStatus;
}

/*
 * Intent : Verify streams work independently
 */
bool TestStreamSemantics_3() {
    printf("------------------------------------------------------------------\n");
    int* stream1_shared_data = nullptr;
    int* stream2_shared_data = nullptr;
    stream1_shared_data = (int*)malloc(sizeof(int));
    stream2_shared_data = (int*)malloc(sizeof(int));
    hipStream_t stream1, stream2;
    CHECK(hipStreamCreate(&stream1));
    CHECK(hipStreamCreate(&stream2));

    *stream1_shared_data = 1;
    CHECK(hipStreamAddCallback(stream1, callback_sleep2, stream1_shared_data, 0));

    int* dev_ptr = nullptr;
    CHECK(hipMalloc(&dev_ptr, sizeof(int)));
    hipLaunchKernelGGL(addOne, 1, 1, 0, 0, dev_ptr);
    CHECK(hipGetLastError());

    *stream2_shared_data = 1;
    CHECK(hipStreamAddCallback(stream2, callback_sleep10, stream2_shared_data, 0));

    printf("Going to sync stream1\n");
    CHECK(hipStreamSynchronize(stream1));
    printf("Going to call query\n");

    hipError_t status = hipStreamQuery(stream2);
    bool testStatus = true;
    printf("%s : ", __FUNCTION__);
    if (status != hipErrorNotReady) {
        printf("%s %s %s\n","\033[0;31m", "Failed", "\033[0m");
        testStatus = false;
        //printf("Failed, queue status is %s but expected is hipErrorNotReady\n", hipGetErrorName(status));
    } else {
        printf("Passed\n");
    }

    CHECK(hipDeviceSynchronize());
    
    // clean-up
    CHECK(hipStreamDestroy(stream1));
    CHECK(hipStreamDestroy(stream2));
    free(stream1_shared_data);
    free(stream2_shared_data);

    return testStatus;
}

int main() {
    TestStreamSemantics_1();
    TestStreamSemantics_2();
    TestStreamSemantics_3();
    return 0;
}

