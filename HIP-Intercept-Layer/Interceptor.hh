#ifndef HIP_INTERCEPT_LAYER_INTERCEPTOR_HH
#define HIP_INTERCEPT_LAYER_INTERCEPTOR_HH

#include "Tracer.hh"
#include <vector>
#include <map>
#include <memory>
#include <string>
#include <fstream>
#include <utility>
#include <cstring>
#include <unordered_map>
#include "Util.hh"
#include <sstream>
#include <iostream>
#include <dlfcn.h>
#include <link.h>
#include <unordered_map>
#include <algorithm>
#include <regex>
#include <unistd.h>
#include <linux/limits.h>
#include <chrono>
#include <filesystem>
#include <sys/stat.h>

// Forward declarations
struct dim3;
struct hipDeviceProp_t;

// Use hip_intercept namespace
using namespace hip_intercept;

// GPU allocation tracking
class AllocationInfo {
public:
    size_t size;
    std::unique_ptr<char[]> shadow_copy;
    
    explicit AllocationInfo(size_t s) : size(s), shadow_copy(new char[s]) {}
};

// Global state - declare extern variable
extern std::unordered_map<void*, AllocationInfo> gpu_allocations;

// Helper function declarations
std::pair<void*, AllocationInfo*> findContainingAllocation(void* ptr);

// External C interface declarations
extern "C" {
    hipError_t hipMalloc(void **ptr, size_t size);
    hipError_t hipLaunchKernel(const void *function_address, dim3 numBlocks,
                              dim3 dimBlocks, void **args, size_t sharedMemBytes,
                              hipStream_t stream);
    hipError_t hipDeviceSynchronize(void);
    hipError_t hipFree(void* ptr);
    hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind);
    hipError_t hipMemset(void *dst, int value, size_t sizeBytes);
    hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX,
                                    unsigned int gridDimY, unsigned int gridDimZ,
                                    unsigned int blockDimX, unsigned int blockDimY,
                                    unsigned int blockDimZ, unsigned int sharedMemBytes,
                                    hipStream_t stream, void** kernelParams,
                                    void** extra);
}

#endif // HIP_INTERCEPT_LAYER_INTERCEPTOR_HH
