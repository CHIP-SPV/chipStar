#ifndef HIP_INTERCEPT_LAYER_TRACER_HH
#define HIP_INTERCEPT_LAYER_TRACER_HH

#include <string>
#include <fstream>
#include <memory>
#include <vector>
#include <map>

#define __HIP_PLATFORM_SPIRV__
#include "hip/hip_runtime_api.h"

#include <cxxabi.h> // For demangling
#include <regex>
#include <sstream>
#include <unordered_map>

// Forward declarations
struct dim3;

namespace hip_intercept {

// Memory state tracking
struct MemoryState {
    std::unique_ptr<char[]> data;
    size_t size;
    
    explicit MemoryState(size_t s);
    MemoryState(const char* src, size_t s);
    MemoryState(); // Default constructor
};

// Kernel execution record
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

// Memory operation record
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

// Add helper functions declarations
std::string dim3ToString(const dim3& d);
const char* memcpyKindToString(hipMemcpyKind kind);
std::string devicePropsToString(const hipDeviceProp_t* props);
std::string demangle(const char* name);
std::string getKernelSignature(const void* function_address);
std::string getKernelName(const void* function_address);

// Add kernel argument tracking
struct KernelArgInfo {
    bool is_vector;
    size_t size;
};

struct KernelInfo {
    std::vector<KernelArgInfo> args;
};

class Tracer {
public:
    static Tracer& instance();
    
    void recordKernelLaunch(const KernelExecution& exec);
    void recordMemoryOperation(const MemoryOperation& op);
    void flush(); // Write current trace to disk
    
    // Disable copy/move
    Tracer(const Tracer&) = delete;
    Tracer& operator=(const Tracer&) = delete;
    Tracer(Tracer&&) = delete;
    Tracer& operator=(Tracer&&) = delete;
    
    void registerKernelArg(const std::string& kernel_name, size_t arg_index, 
                          bool is_vector, size_t size);
    bool isVectorType(const std::string& type_name) const;
    void printKernelArgs(void** args, const std::string& kernelName, 
                        const void* function_address);
    
private:
    Tracer(); // Private constructor for singleton
    ~Tracer();
    
    void initializeTraceFile();
    void writeEvent(uint32_t type, const void* data, size_t size);
    void writeKernelExecution(const KernelExecution& exec);
    void writeMemoryOperation(const MemoryOperation& op);
    
    std::string getTraceFilePath() const;
    
    std::ofstream trace_file_;
    std::string trace_path_;
    bool initialized_;
    static constexpr uint32_t TRACE_MAGIC = 0x48495054; // "HIPT"
    static constexpr uint32_t TRACE_VERSION = 1;
    
    std::unordered_map<std::string, KernelInfo> kernel_registry_;
};

} // namespace hip_intercept

#endif // HIP_INTERCEPT_LAYER_TRACER_HH
