#include <hip/hip_runtime.h>
#include <hip/hip_interop.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>

#define HIP_CHECK(status) \
    if (status != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(status) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    }

// For hipGetBackendNativeHandles which returns int, not hipError_t
#define INTEROP_CHECK(status) \
    if (status != 0) { \
        std::cerr << "Interop error: " << status << " at line " << __LINE__ << std::endl; \
        exit(1); \
    }

// Handle indices for debugging
const char* HandleNames[] = {
    "BACKEND_NAME",
    "PLATFORM_DRIVER", 
    "DEVICE", 
    "CONTEXT", 
    "QUEUE", 
    "COMMAND_LIST"
};

// More detailed handle printing with names
void print_handles(const char* prefix, const std::vector<uintptr_t>& handles) {
    std::cout << "\n=== " << prefix << " (" << handles.size() << " handles) ===" << std::endl;
    
    for (size_t i = 0; i < handles.size(); i++) {
        const char* name = (i < sizeof(HandleNames)/sizeof(HandleNames[0])) ? 
                       HandleNames[i] : "UNKNOWN";
        std::cout << "Handle[" << i << "] " << name << ": 0x" << std::hex 
                 << handles[i] << std::dec << std::endl;
    }
    std::cout << std::endl;
}

// Compare two sets of handles and print differences
void compare_handles(const char* name1, const std::vector<uintptr_t>& handles1, 
                     const char* name2, const std::vector<uintptr_t>& handles2) {
    std::cout << "\n=== Comparing " << name1 << " vs " << name2 << " ===" << std::endl;
    
    size_t minSize = std::min(handles1.size(), handles2.size());
    size_t maxSize = std::max(handles1.size(), handles2.size());
    
    bool hasDifference = false;
    
    // First check if sizes are different
    if (handles1.size() != handles2.size()) {
        std::cout << "DIFFERENCE: Handle count mismatch - " << name1 << ": " 
                 << handles1.size() << ", " << name2 << ": " << handles2.size() << std::endl;
        hasDifference = true;
    }
    
    // Compare handles that both arrays have
    for (size_t i = 0; i < minSize; i++) {
        const char* name = (i < sizeof(HandleNames)/sizeof(HandleNames[0])) ? 
                       HandleNames[i] : "UNKNOWN";
        
        if (handles1[i] != handles2[i]) {
            std::cout << "DIFFERENCE: Handle[" << i << "] " << name << " - " 
                     << name1 << ": 0x" << std::hex << handles1[i] << ", " 
                     << name2 << ": 0x" << handles2[i] << std::dec << std::endl;
            hasDifference = true;
        }
    }
    
    // Report on any extra handles
    for (size_t i = minSize; i < maxSize; i++) {
        const char* name = (i < sizeof(HandleNames)/sizeof(HandleNames[0])) ? 
                       HandleNames[i] : "UNKNOWN";
        
        if (i < handles1.size()) {
            std::cout << "DIFFERENCE: Extra handle in " << name1 << " - Handle[" 
                     << i << "] " << name << ": 0x" << std::hex << handles1[i] << std::dec << std::endl;
            hasDifference = true;
        } else {
            std::cout << "DIFFERENCE: Extra handle in " << name2 << " - Handle[" 
                     << i << "] " << name << ": 0x" << std::hex << handles2[i] << std::dec << std::endl;
            hasDifference = true;
        }
    }
    
    if (!hasDifference) {
        std::cout << "All handles match between " << name1 << " and " << name2 << "." << std::endl;
    }
    
    std::cout << std::endl;
}

// Implement a minimal SASUM-like function that just demonstrates handle management
float minimal_sasum(const float* x, int n, int incx, uintptr_t* handles, int numHandles) {
    std::cout << "\n=== Running minimal_sasum with " << numHandles << " handles ===" << std::endl;
    print_handles("Operation", std::vector<uintptr_t>(handles, handles + numHandles));
    
    // Simulate the computation (just add absolute values)
    float result = 0.0f;
    std::vector<float> host_x(n);
    HIP_CHECK(hipMemcpy(host_x.data(), x, n * sizeof(float), hipMemcpyDeviceToHost));
    
    for (int i = 0; i < n; i += incx) {
        result += std::abs(host_x[i]);
    }
    
    return result;
}

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "=== Backend Mismatch CPU Reproducer (Stream Handle Test) ===" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    // Print HIP device information
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    std::cout << "Number of HIP devices found: " << deviceCount << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, i));
        std::cout << "Device " << i << ": " << props.name << std::endl;
        std::cout << "  - Architecture: " << props.gcnArchName << std::endl;
        std::cout << "  - Compute capability: " << props.major << "." << props.minor << std::endl;
        std::cout << "  - Total global memory: " << (props.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "  - Backend type: ";
        if (strstr(props.gcnArchName, "gfx") != nullptr) {
            std::cout << "AMD ROCm" << std::endl;
        } else if (strstr(props.name, "Intel") != nullptr) {
            std::cout << "Intel SYCL" << std::endl;
        } else if (strstr(props.name, "NVIDIA") != nullptr) {
            std::cout << "NVIDIA CUDA" << std::endl;
        } else {
            std::cout << "Unknown" << std::endl;
        }
    }
    
    // Get current device
    int currentDevice = 0;
    HIP_CHECK(hipGetDevice(&currentDevice));
    hipDeviceProp_t currentProps;
    HIP_CHECK(hipGetDeviceProperties(&currentProps, currentDevice));
    std::cout << "Current active device: " << currentDevice << " (" << currentProps.name << ")" << std::endl;
    
    // Get handles for default stream
    int nHandles = 0;
    int status = hipGetBackendNativeHandles((uintptr_t)nullptr, nullptr, &nHandles);
    INTEROP_CHECK(status);
    std::cout << "Number of backend handles for default stream: " << nHandles << std::endl;
    
    std::vector<uintptr_t> defaultHandles(nHandles);
    status = hipGetBackendNativeHandles((uintptr_t)nullptr, defaultHandles.data(), 0);
    INTEROP_CHECK(status);
    print_handles("Default stream", defaultHandles);
    
    // Store default handles for simulating context
    void* contextHandle = malloc(sizeof(uintptr_t) * nHandles);
    if (!contextHandle) {
        std::cerr << "Failed to allocate memory for context handle" << std::endl;
        return 1;
    }
    memcpy(contextHandle, defaultHandles.data(), nHandles * sizeof(uintptr_t));
    std::cout << "Simulated context created successfully" << std::endl;
    
    // Create a new stream
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    std::cout << "New stream created successfully" << std::endl;
    
    // Get handles for new stream
    int streamHandleCount = 0;
    status = hipGetBackendNativeHandles(reinterpret_cast<uintptr_t>(stream), nullptr, &streamHandleCount);
    INTEROP_CHECK(status);
    std::cout << "Number of backend handles for new stream: " << streamHandleCount << std::endl;
    
    std::vector<uintptr_t> streamHandles(streamHandleCount);
    status = hipGetBackendNativeHandles(reinterpret_cast<uintptr_t>(stream), streamHandles.data(), 0);
    INTEROP_CHECK(status);
    print_handles("New stream", streamHandles);
    
    // Compare default and stream handles
    compare_handles("Default stream", defaultHandles, "New stream", streamHandles);
    
    // Simulate updating context with new stream (just update our local copy)
    free(contextHandle);
    contextHandle = malloc(sizeof(uintptr_t) * streamHandleCount);
    if (!contextHandle) {
        std::cerr << "Failed to allocate memory for updated context handle" << std::endl;
        return 1;
    }
    memcpy(contextHandle, streamHandles.data(), streamHandleCount * sizeof(uintptr_t));
    std::cout << "Simulated context updated successfully" << std::endl;
    
    // Prepare test data
    const int n = 5;
    std::vector<float> hx = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f};
    float* dx = nullptr;
    
    HIP_CHECK(hipMalloc((void**)&dx, n * sizeof(float)));
    HIP_CHECK(hipMemcpy(dx, hx.data(), n * sizeof(float), hipMemcpyHostToDevice));
    
    // Test direct SASUM-like operation with raw handles
    std::cout << "\nTesting minimal SASUM with default stream handles..." << std::endl;
    float result1 = minimal_sasum(dx, n, 1, defaultHandles.data(), defaultHandles.size());
    std::cout << "Default stream SASUM result: " << result1 << std::endl;
    
    std::cout << "\nTesting minimal SASUM with new stream handles..." << std::endl;
    float result2 = minimal_sasum(dx, n, 1, streamHandles.data(), streamHandles.size());
    std::cout << "New stream SASUM result: " << result2 << std::endl;
    
    // Create multiple streams to see if there's a pattern to handle changes
    std::cout << "\n=== Testing multiple streams for pattern analysis ===" << std::endl;
    const int numStreams = 3;
    hipStream_t streams[numStreams];
    std::vector<std::vector<uintptr_t>> streamHandlesArray(numStreams);
    
    for (int i = 0; i < numStreams; i++) {
        HIP_CHECK(hipStreamCreate(&streams[i]));
        
        int nHandlesMulti = 0;
        status = hipGetBackendNativeHandles(reinterpret_cast<uintptr_t>(streams[i]), nullptr, &nHandlesMulti);
        INTEROP_CHECK(status);
        
        streamHandlesArray[i].resize(nHandlesMulti);
        status = hipGetBackendNativeHandles(reinterpret_cast<uintptr_t>(streams[i]), 
                                          streamHandlesArray[i].data(), 0);
        INTEROP_CHECK(status);
        
        char prefix[32];
        snprintf(prefix, sizeof(prefix), "Stream %d", i+1);
        print_handles(prefix, streamHandlesArray[i]);
        
        // Compare with previous stream if available
        if (i > 0) {
            char prefix1[32], prefix2[32];
            snprintf(prefix1, sizeof(prefix1), "Stream %d", i);
            snprintf(prefix2, sizeof(prefix2), "Stream %d", i+1);
            compare_handles(prefix1, streamHandlesArray[i-1], prefix2, streamHandlesArray[i]);
        }
    }
    
    // Cleanup
    HIP_CHECK(hipFree(dx));
    for (int i = 0; i < numStreams; i++) {
        HIP_CHECK(hipStreamDestroy(streams[i]));
    }
    HIP_CHECK(hipStreamDestroy(stream));
    free(contextHandle);
    
    std::cout << "=== Test Complete ===" << std::endl;
    return 0;
} 