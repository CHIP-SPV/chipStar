#ifndef HIP_INTERCEPT_LAYER_INTERCEPTOR_HH
#define HIP_INTERCEPT_LAYER_INTERCEPTOR_HH

#define __HIP_PLATFORM_SPIRV__
#include "hip/hip_runtime_api.h"
#include <vector>
#include <map>
#include <memory>
#include <string>
#include <fstream>
#include <utility>
#include <cstring>  // For memcpy

// Forward declarations
struct dim3;
struct hipDeviceProp_t;

// Memory state tracking
struct MemoryState {
    std::unique_ptr<char[]> data;
    size_t size;
    
    explicit MemoryState(size_t s) : data(new char[s]), size(s) {}
    MemoryState(const char* src, size_t s) : data(new char[s]), size(s) {
        memcpy(data.get(), src, s);  // Using C's memcpy instead of std::memcpy
    }
    MemoryState() : size(0) {} // Default constructor for std::map
};

// Kernel execution tracking
struct KernelExecution {
    void* function_address;
    std::string kernel_name;
    dim3 grid_dim;
    dim3 block_dim;
    size_t shared_mem;
    hipStream_t stream;
    uint64_t execution_order;
    
    std::map<void*, MemoryState> pre_state;
    std::map<void*, MemoryState> post_state;
    std::vector<std::pair<void*, size_t>> changes;
    std::vector<void*> arg_ptrs;
    std::vector<size_t> arg_sizes;
    std::map<int, std::vector<std::pair<size_t, std::pair<float, float>>>> changes_by_arg;
};

// Memory operation tracking
enum class MemoryOpType {
    COPY,
    SET
};

struct MemoryOperation {
    MemoryOpType type;
    void* dst;
    const void* src;
    size_t size;
    int value;
    hipMemcpyKind kind;
    uint64_t execution_order;
    
    std::shared_ptr<MemoryState> pre_state;
    std::shared_ptr<MemoryState> post_state;
};

// Global state
extern std::vector<MemoryOperation> memory_operations;
extern std::vector<KernelExecution> kernel_executions;

// Kernel argument info
struct KernelArgInfo {
    bool is_vector;
    size_t size;
};

struct KernelInfo {
    std::vector<KernelArgInfo> args;
};

// Trace file format
struct TraceHeader {
    uint32_t magic;
    uint32_t version;
    static const uint32_t MAGIC = 0x48495054; // "HIPT"
    static const uint32_t VERSION = 1;
};

struct TraceEvent {
    enum Type : uint32_t {
        KERNEL_LAUNCH = 1,
        MEMORY_COPY = 2,
        MEMORY_SET = 3
    } type;
    
    uint64_t timestamp;
    uint32_t size;
};

// Trace file management
class TraceFile {
public:
    TraceFile(const std::string& path);
    ~TraceFile();
    
    void writeEvent(TraceEvent::Type type, const void* data, size_t data_size);
    void readAndProcessTrace();

private:
    void writeKernelExecution(const KernelExecution& exec);
    
    std::ofstream trace_file_;
    std::string path_;
};

// Helper functions declarations
std::string getTraceFilePath();
void registerKernelArg(const std::string& kernel_name, size_t arg_index, bool is_vector, size_t size);

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
