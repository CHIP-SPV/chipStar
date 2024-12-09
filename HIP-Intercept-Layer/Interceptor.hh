#ifndef HIP_INTERCEPT_LAYER_INTERCEPTOR_HH
#define HIP_INTERCEPT_LAYER_INTERCEPTOR_HH

#define __HIP_PLATFORM_SPIRV__
#include "hip/hip_runtime_api.h"
#include "Tracer.hh"

#include <vector>
#include <map>
#include <memory>
#include <string>
#include <fstream>
#include <utility>
#include <cstring>
#include <unordered_map>

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
std::string getKernelSignature(const void* function_address);
std::string getKernelName(const void* function_address);
size_t countKernelArgs(void** args);
std::string getArgTypeFromSignature(const std::string& signature, size_t arg_index);
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
}

#endif // HIP_INTERCEPT_LAYER_INTERCEPTOR_HH
