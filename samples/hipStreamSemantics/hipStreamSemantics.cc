/*
 * Copyright (c) 2022-23 CHIP-SPV developers
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
#include <unistd.h>

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

int main() {
    int* stream1_shared_data = nullptr;
    int* stream2_shared_data = nullptr;
    stream1_shared_data = (int*)malloc(sizeof(int));
    stream2_shared_data = (int*)malloc(sizeof(int));
    hipStream_t stream1, stream2;
    hipStreamCreate(&stream1);
    hipStreamCreate(&stream2);

    *stream1_shared_data = 1;
    hipStreamAddCallback(stream1, callback_sleep2, stream1_shared_data, 0);
    *stream2_shared_data = 1;
    hipStreamAddCallback(stream2, callback_sleep10, stream2_shared_data, 0);

    printf("Going to sync stream1\n");
    hipStreamSynchronize(stream1);
    printf("Going to call query\n");
    printf("Stream2 status %s\n", hipGetErrorName(hipStreamQuery(stream2)));

    hipDeviceSynchronize();
    if (*stream1_shared_data != 2 || *stream1_shared_data != 2) {
        printf("Failed\n");
    } else {
        printf("Passed\n");
    }

    return 0;
}