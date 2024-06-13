#include <hip/hip_runtime.h>
#include <iostream>

#define NUM 1024  // Assuming a constant for the number of elements

// Callback function to be called after the stream operation is done
void TestCallback(hipStream_t stream, hipError_t status, void *userData) {
    std::cout << "Callback executed. Data at: " << userData << std::endl;
}

// Main function to demonstrate multiple streams in HIP
int main() {
    const int num_streams = 2;
    const int width = 256;  // Assuming width for matrix transpose
    float* data[num_streams];
    float* randArray;
    float* gpuTransposeMatrix[num_streams];
    float* TransposeMatrix[num_streams];
    hipStream_t streams[num_streams];

    // Allocate host memory
    randArray = new float[NUM];
    for (int i = 0; i < NUM; i++) {
        randArray[i] = static_cast<float>(i);
    }

    // Create streams
    for (int i = 0; i < num_streams; i++) {
        hipStreamCreate(&streams[i]);
    }

    // Allocate and initialize GPU memory
    for (int i = 0; i < num_streams; i++) {
        hipMalloc((void**)&data[i], NUM * sizeof(float));
        hipMalloc((void**)&gpuTransposeMatrix[i], NUM * sizeof(float));
        hipMalloc((void**)&TransposeMatrix[i], NUM * sizeof(float));
        hipMemcpyAsync(data[i], randArray, NUM * sizeof(float), hipMemcpyHostToDevice, streams[i]);
    }

    // Example operation: Copy data from one GPU memory area to another
    hipMemcpyAsync(TransposeMatrix[0], gpuTransposeMatrix[0], NUM * sizeof(float), hipMemcpyDeviceToHost, streams[0]);

    // Add callback to the first stream
    hipStreamAddCallback(streams[0], TestCallback, (void*)TransposeMatrix[0], 0);

    // Synchronize and clean up
    for (int i = 0; i < num_streams; i++) {
        hipStreamSynchronize(streams[i]);
        hipStreamDestroy(streams[i]);
        hipFree(data[i]);
        hipFree(gpuTransposeMatrix[i]);
        hipFree(TransposeMatrix[i]);
    }

    delete[] randArray;

    return 0;
}