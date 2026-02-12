#include <unistd.h>
#include <hip/hip_runtime.h>
#include <iostream>

void host_function(void* userData) {
    int* data = static_cast<int*>(userData);
    std::cout << "Host function executed. Received data: " << *data << std::endl;
    sleep(1);
}

__global__ void simple_kernel123(int* device_data) {
    // Simple kernel to perform some device-side work
    device_data[0] = 123;
}
__global__ void simple_kernel456(int* device_data) {
    // Simple kernel to perform some device-side work
    device_data[0] = 456;
}

int main() {
    int* host_data;
    int* device_data;

    // Allocate device memory
    hipMalloc(&device_data, sizeof(int));
    hipHostMalloc(&host_data, sizeof(int));

    // Create HIP streams
    hipStream_t stream1, stream2;
    hipStreamCreate(&stream1); hipStreamCreate(&stream2);

    hipEvent_t event1, event2;
    hipEventCreateWithFlags(&event1,  hipEventDisableTiming);
    hipEventCreateWithFlags(&event2,  hipEventDisableTiming);


    // Launch a kernel on the stream
    hipLaunchKernelGGL(simple_kernel123, dim3(1), dim3(1), 0, stream1, device_data);

    hipMemcpyAsync(host_data, device_data, sizeof(int), hipMemcpyDeviceToHost, stream1);
    hipEventRecord(event1, stream1);

    hipLaunchHostFunc(stream1, host_function, host_data);

    hipStreamWaitEvent(stream2, event1, 0 /*flags*/ );
    // Launch a kernel on the stream1
    hipLaunchKernelGGL(simple_kernel456, dim3(1), dim3(1), 0, stream1, device_data);

    hipMemcpyAsync(host_data, device_data, sizeof(int), hipMemcpyDeviceToHost, stream1);

    hipLaunchHostFunc(stream1, host_function, host_data);

    hipStreamSynchronize(stream1);

    // Clean up
    hipEventDestroy(event1);
    hipEventDestroy(event2);
    hipFree(device_data);
    hipStreamDestroy(stream1);
    hipStreamDestroy(stream2);

    std::cout << "PASS" << std::endl;
    return 0;
}
