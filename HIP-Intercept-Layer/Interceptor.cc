#include "Interceptor.hh"
#include <sstream>
#include <iostream>
#include <dlfcn.h>
#include <link.h>
#include <unordered_map>
#include <algorithm>
#include <cxxabi.h>
#include <regex>
#include <unistd.h>
#include <linux/limits.h>
#include <chrono>
#include <filesystem>
#include <sys/stat.h>

// Define the global state
std::vector<MemoryOperation> memory_operations;
std::vector<KernelExecution> kernel_executions;

namespace {
// Function pointer types
typedef hipError_t (*hipMalloc_fn)(void**, size_t);
typedef hipError_t (*hipMemcpy_fn)(void*, const void*, size_t, hipMemcpyKind);
typedef hipError_t (*hipLaunchKernel_fn)(const void*, dim3, dim3, void**, size_t, hipStream_t);
typedef hipError_t (*hipDeviceSynchronize_fn)(void);
typedef hipError_t (*hipFree_fn)(void*);
typedef hipError_t (*hipMemset_fn)(void*, int, size_t);

// Get the real function pointers
void* getOriginalFunction(const char* name) {
    std::cout << "Looking for symbol: " << name << std::endl;
    
    // Try to find the symbol in any loaded library
    void* sym = dlsym(RTLD_NEXT, name);
    if (!sym) {
        std::cerr << "ERROR: Could not find implementation of " << name 
                  << ": " << dlerror() << std::endl;
        
        // Print currently loaded libraries for debugging
        void* handle = dlopen(NULL, RTLD_NOW);
        if (handle) {
            link_map* map;
            dlinfo(handle, RTLD_DI_LINKMAP, &map);
            std::cerr << "Loaded libraries:" << std::endl;
            while (map) {
                std::cerr << "  " << map->l_name << std::endl;
                map = map->l_next;
            }
        }
        
        std::cerr << "Make sure the real HIP runtime library is loaded." << std::endl;
        exit(1);
    }
    
    std::cout << "Found symbol " << name << " at " << sym << std::endl;
    return sym;
}

// Lazy function pointer getters
hipMalloc_fn get_real_hipMalloc() {
    static auto fn = (hipMalloc_fn)getOriginalFunction("hipMalloc");
    return fn;
}

hipMemcpy_fn get_real_hipMemcpy() {
    static auto fn = (hipMemcpy_fn)getOriginalFunction("hipMemcpy");
    return fn;
}

hipLaunchKernel_fn get_real_hipLaunchKernel() {
    static auto fn = (hipLaunchKernel_fn)getOriginalFunction("hipLaunchKernel");
    return fn;
}

hipDeviceSynchronize_fn get_real_hipDeviceSynchronize() {
    static auto fn = (hipDeviceSynchronize_fn)getOriginalFunction("hipDeviceSynchronize");
    return fn;
}

hipFree_fn get_real_hipFree() {
    static auto fn = (hipFree_fn)getOriginalFunction("hipFree");
    return fn;
}

hipMemset_fn get_real_hipMemset() {
    static auto fn = (hipMemset_fn)getOriginalFunction("hipMemset");
    return fn;
}

// Helper function to convert dim3 to string
static std::string dim3ToString(dim3 d) {
  std::stringstream ss;
  ss << "{" << d.x << "," << d.y << "," << d.z << "}";
  return ss.str();
}

// Helper function to convert hipMemcpyKind to string
static const char* memcpyKindToString(hipMemcpyKind kind) {
  switch(kind) {
    case hipMemcpyHostToHost: return "hipMemcpyHostToHost";
    case hipMemcpyHostToDevice: return "hipMemcpyHostToDevice"; 
    case hipMemcpyDeviceToHost: return "hipMemcpyDeviceToHost";
    case hipMemcpyDeviceToDevice: return "hipMemcpyDeviceToDevice";
    case hipMemcpyDefault: return "hipMemcpyDefault";
    default: return "Unknown";
  }
}

// Helper for hipDeviceProp_t
static std::string devicePropsToString(const hipDeviceProp_t* props) {
  if (!props) return "null";
  std::stringstream ss;
  ss << "{name=" << props->name << ", totalGlobalMem=" << props->totalGlobalMem << "}";
  return ss.str();
}

// Helper to get type name as string
template<typename T>
static std::string getTypeName() {
  std::string name = typeid(T).name();
  
  // Clean up type name
  if constexpr (std::is_pointer_v<T>) {
    name = getTypeName<std::remove_pointer_t<T>>() + "*";
  }
  else if constexpr (std::is_const_v<T>) {
    name = "const " + getTypeName<std::remove_const_t<T>>();
  }
  else {
    // Map common types to readable names
    if (name == "f") name = "float";
    else if (name == "i") name = "int";
    else if (name == "d") name = "double";
    // Add more type mappings as needed
  }
  return name;
}

template<typename... Args>
static std::string getArgTypes() {
  std::vector<std::string> typeNames;
  (typeNames.push_back(getTypeName<Args>()), ...);
  
  std::stringstream ss;
  ss << "(";
  for (size_t i = 0; i < typeNames.size(); i++) {
    if (i > 0) ss << ", ";
    ss << typeNames[i];
  }
  ss << ")";
  return ss.str();
}

// Helper to demangle C++ names
static std::string demangle(const char* name) {
    int status;
    std::unique_ptr<char, void(*)(void*)> demangled(
        abi::__cxa_demangle(name, nullptr, nullptr, &status),
        std::free
    );
    return status == 0 ? demangled.get() : name;
}

// Helper to extract kernel signature from binary
static std::string getKernelSignature(const void* function_address) {
    // Get the current executable's path from /proc
    char exe_path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path)-1);
    if (len == -1) {
        std::cerr << "Failed to read /proc/self/exe: " << strerror(errno) << std::endl;
        return "";
    }
    exe_path[len] = '\0';
    
    // Use nm to get symbol information from the executable
    std::string cmd = "nm -C " + std::string(exe_path) + " | grep __device_stub_";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        std::cerr << "Failed to run nm command: " << strerror(errno) << std::endl;
        return "";
    }
    
    char buffer[1024];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    int status = pclose(pipe);
    if (status == -1) {
        std::cerr << "Failed to close pipe: " << strerror(errno) << std::endl;
    }
    
    // Parse the nm output to extract function signature
    std::regex signature_regex(R"(__device_stub_([^\(]+)\((.*?)\))");
    std::smatch matches;
    if (std::regex_search(result, matches, signature_regex)) {
        return matches[1].str() + "(" + matches[2].str() + ")";
    }
    
    std::cerr << "No kernel signature found in binary " << exe_path << std::endl;
    return "";
}

// Helper to print kernel arguments
static void printKernelArgs(void** args, const std::string& kernelName, const void* function_address) {
    std::string signature = getKernelSignature(function_address);
    std::cout << "    kernel signature: " << signature << "\n";
    
    if (!args) {
        std::cout << "    args: nullptr\n";
        return;
    }
    
    // Parse signature to get argument types
    std::vector<std::string> argTypes;
    size_t start = signature.find('(');
    size_t end = signature.find(')');
    if (start != std::string::npos && end != std::string::npos) {
        std::string argsStr = signature.substr(start + 1, end - start - 1);
        
        // Handle template arguments more carefully
        size_t pos = 0;
        int template_depth = 0;
        std::string current_arg;
        
        for (char c : argsStr) {
            if (c == '<') template_depth++;
            else if (c == '>') template_depth--;
            else if (c == ',' && template_depth == 0) {
                // Only split on commas outside of template arguments
                if (!current_arg.empty()) {
                    // Trim whitespace
                    current_arg.erase(0, current_arg.find_first_not_of(" "));
                    current_arg.erase(current_arg.find_last_not_of(" ") + 1);
                    argTypes.push_back(current_arg);
                    current_arg.clear();
                }
                continue;
            }
            current_arg += c;
        }
        if (!current_arg.empty()) {
            current_arg.erase(0, current_arg.find_first_not_of(" "));
            current_arg.erase(current_arg.find_last_not_of(" ") + 1);
            argTypes.push_back(current_arg);
        }
    }
    
    std::cout << "    args:\n";
    for (size_t i = 0; i < argTypes.size() && args[i] != nullptr; i++) {
        std::cout << "      arg[" << i << "]: " << argTypes[i] << " = ";
        
        try {
            // Handle different argument types
            if (argTypes[i].find("*") != std::string::npos) {
                // Pointer type
                void* ptr = *(void**)args[i];
                std::cout << ptr;
            } else if (argTypes[i].find("HIP_vector_type") != std::string::npos) {
                // For vector types, print the first value
                float* values = (float*)args[i];
                std::cout << values[0];  // Print just the first component
            } else if (argTypes[i].find("int") != std::string::npos) {
                // Integer type
                std::cout << *(int*)args[i];
            } else if (argTypes[i].find("float") != std::string::npos) {
                // Float type
                std::cout << *(float*)args[i];
            } else {
                // Unknown type - show raw pointer
                std::cout << "[unknown type at " << args[i] << "]";
            }
        } catch (...) {
            std::cout << "[failed to read argument]";
        }
        std::cout << "\n";
    }
}

// Add this helper function to get kernel name from function address
static std::string getKernelName(const void* function_address) {
    Dl_info info;
    if (dladdr(function_address, &info) && info.dli_sname) {
        return info.dli_sname;
    }
    // If we can't get the name, return the address as hex
    std::stringstream ss;
    ss << "kernel_" << std::hex << (uintptr_t)function_address;
    return ss.str();
}

struct AllocationInfo {
    size_t size;
    std::unique_ptr<char[]> shadow_copy;
    
    AllocationInfo(size_t s) : size(s), shadow_copy(new char[s]) {}
};

// Track GPU allocations: device_ptr -> AllocationInfo
static std::unordered_map<void*, AllocationInfo> gpu_allocations;

// Helper to find which allocation a pointer belongs to
static std::pair<void*, AllocationInfo*> findContainingAllocation(void* ptr) {
    for (auto& [base_ptr, info] : gpu_allocations) {
        char* start = static_cast<char*>(base_ptr);
        char* end = start + info.size;
        if (ptr >= base_ptr && ptr < end) {
            return {base_ptr, &info};
        }
    }
    return {nullptr, nullptr};
}

// Helper to create shadow copy of an allocation
static void createShadowCopy(void* base_ptr, AllocationInfo& info) {
    // Copy current GPU memory to shadow copy
    hipError_t err = get_real_hipMemcpy()(
        info.shadow_copy.get(), 
        base_ptr,
        info.size,
        hipMemcpyDeviceToHost);
    
    if (err != hipSuccess) {
        std::cerr << "Failed to create shadow copy for allocation at " 
                  << base_ptr << " of size " << info.size << std::endl;
    } else {
        // Print first 3 values for debugging
        float* values = reinterpret_cast<float*>(info.shadow_copy.get());
        std::cout << "Shadow copy for " << base_ptr << " first 3 values: "
                  << values[0] << ", " << values[1] << ", " << values[2] 
                  << std::endl;
    }
}

// Move this helper function declaration to the top with other helper functions
// Add before recordMemoryChanges
static int getArgumentIndex(void* ptr, const std::vector<void*>& arg_ptrs) {
    for (size_t i = 0; i < arg_ptrs.size(); i++) {
        if (arg_ptrs[i] == ptr) return i;
    }
    return -1;
}

// Then the recordMemoryChanges function that uses it
static void recordMemoryChanges(KernelExecution& exec) {
    for (const auto& [ptr, pre] : exec.pre_state) {
        const auto& post = exec.post_state[ptr];
        
        // For output arguments (arg2 in MatrixMul), record all values
        int arg_idx = getArgumentIndex(ptr, exec.arg_ptrs);
        if (arg_idx == 2) {  // Output matrix C
            // Record all values for output arguments
            for (size_t i = 0; i < pre.size; i += sizeof(float)) {
                exec.changes.push_back({(char*)ptr + i, i});
                float* pre_val = (float*)(pre.data.get() + i);
                float* post_val = (float*)(post.data.get() + i);
                exec.changes_by_arg[arg_idx].push_back({i/sizeof(float), {*pre_val, *post_val}});
            }
        } else {
            // For input arguments, only record actual changes
            for (size_t i = 0; i < pre.size; i += sizeof(float)) {
                float* pre_val = (float*)(pre.data.get() + i);
                float* post_val = (float*)(post.data.get() + i);
                
                if (*pre_val != *post_val) {
                    exec.changes.push_back({(char*)ptr + i, i});
                    exec.changes_by_arg[arg_idx].push_back({i/sizeof(float), {*pre_val, *post_val}});
                }
            }
        }
    }
}

// Add this to the printKernelSummary function
static void printMemoryOperations() {
    std::cout << "\n=== Memory Operations Summary ===\n";
    
    for (const auto& op : memory_operations) {
        if (op.type == MemoryOpType::COPY) {
            std::cout << "\nMemcpy: " << op.size << " bytes\n"
                     << "  dst: " << op.dst << "\n"
                     << "  src: " << op.src << "\n"
                     << "  kind: " << memcpyKindToString(op.kind) << "\n";
        } else {
            std::cout << "\nMemset: " << op.size << " bytes\n"
                     << "  dst: " << op.dst << "\n"
                     << "  value: " << op.value << "\n";
        }
        
        // Print first few values before/after
        float* pre_vals = (float*)op.pre_state->data.get();
        float* post_vals = (float*)op.post_state->data.get();
        size_t num_floats = std::min(size_t(5), op.size / sizeof(float));
        
        std::cout << "  First " << num_floats << " values:\n";
        for (size_t i = 0; i < num_floats; i++) {
            std::cout << "    [" << i << "]: " << pre_vals[i] 
                     << " -> " << post_vals[i] << "\n";
        }
    }
}

// Move the implementation functions here
static hipError_t hipMemcpy_impl(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind) {
    std::cout << "\n=== INTERCEPTED hipMemcpy ===\n";
    std::cout << "hipMemcpy(dst=" << dst << ", src=" << src 
              << ", size=" << sizeBytes << ", kind=" << memcpyKindToString(kind) << ")\n";
    
    MemoryOperation op;
    op.type = MemoryOpType::COPY;
    op.dst = dst;
    op.src = src;
    op.size = sizeBytes;
    op.kind = kind;
    static uint64_t op_count = 0;
    op.execution_order = op_count++;
    
    // Initialize pre_state and post_state
    op.pre_state = std::make_shared<MemoryState>(sizeBytes);
    op.post_state = std::make_shared<MemoryState>(sizeBytes);
    
    // Capture pre-copy state if destination is GPU memory
    if (kind != hipMemcpyHostToHost) {
        auto [base_ptr, info] = findContainingAllocation(dst);
        if (base_ptr && info) {
            createShadowCopy(base_ptr, *info);
            memcpy(op.pre_state->data.get(), info->shadow_copy.get(), sizeBytes);
        }
    }
    
    // Perform the copy
    hipError_t result = get_real_hipMemcpy()(dst, src, sizeBytes, kind);
    
    // Capture post-copy state
    if (kind != hipMemcpyHostToHost) {
        auto [base_ptr, info] = findContainingAllocation(dst);
        if (base_ptr && info) {
            createShadowCopy(base_ptr, *info);
            memcpy(op.post_state->data.get(), info->shadow_copy.get(), sizeBytes);
        }
    }
    
    memory_operations.push_back(std::move(op));
    return result;
}

static hipError_t hipMemset_impl(void *dst, int value, size_t sizeBytes) {
    std::cout << "hipMemset(dst=" << dst << ", value=" << value 
              << ", size=" << sizeBytes << ")\n";
    
    MemoryOperation op;
    op.type = MemoryOpType::SET;
    op.dst = dst;
    op.size = sizeBytes;
    op.value = value;
    static uint64_t op_count = 0;
    op.execution_order = op_count++;
    
    // Initialize states
    op.pre_state = std::make_shared<MemoryState>(sizeBytes);
    op.post_state = std::make_shared<MemoryState>(sizeBytes);
    
    // Capture pre-set state
    auto [base_ptr, info] = findContainingAllocation(dst);
    if (base_ptr && info) {
        createShadowCopy(base_ptr, *info);
        memcpy(op.pre_state->data.get(), info->shadow_copy.get(), sizeBytes);
    }
    
    // Perform the memset
    hipError_t result = get_real_hipMemset()(dst, value, sizeBytes);
    
    // Capture post-set state
    if (base_ptr && info) {
        createShadowCopy(base_ptr, *info);
        memcpy(op.post_state->data.get(), info->shadow_copy.get(), sizeBytes);
    }
    
    memory_operations.push_back(std::move(op));
    return result;
}

// Replace the printKernelSummary function
static void __attribute__((destructor)) printKernelSummary() {
    std::cout << "\n=== HIP API Execution Trace ===\n";
    std::cout << "Found " << kernel_executions.size() << " kernel executions\n";
    std::cout << "Found " << memory_operations.size() << " memory operations\n";
    
    // Combine kernel executions and memory operations into a single timeline
    struct TimelineEvent {
        enum Type { KERNEL, MEMCPY, MEMSET } type;
        uint64_t order;
        const void* data;  // Pointer to KernelExecution or MemoryOperation
        
        TimelineEvent(Type t, uint64_t o, const void* d) 
            : type(t), order(o), data(d) {}
    };
    
    std::vector<TimelineEvent> timeline;
    
    // Add kernel executions to timeline
    for (const auto& exec : kernel_executions) {
        timeline.emplace_back(TimelineEvent::KERNEL, exec.execution_order, &exec);
    }
    
    // Add memory operations to timeline
    for (const auto& op : memory_operations) {
        timeline.emplace_back(
            op.type == MemoryOpType::COPY ? TimelineEvent::MEMCPY : TimelineEvent::MEMSET,
            op.execution_order, 
            &op
        );
    }
    
    // Sort by execution order
    std::sort(timeline.begin(), timeline.end(),
              [](const TimelineEvent& a, const TimelineEvent& b) {
                  return a.order < b.order;
              });
    
    // Print timeline
    for (const auto& event : timeline) {
        switch (event.type) {
            case TimelineEvent::KERNEL: {
                const auto& exec = *static_cast<const KernelExecution*>(event.data);
                std::cout << "\nhipLaunchKernel(\n"
                          << "    kernel: " << exec.kernel_name 
                          << " at " << exec.function_address << "\n"
                          << "    gridDim: " << dim3ToString(exec.grid_dim) << "\n"
                          << "    blockDim: " << dim3ToString(exec.block_dim) << "\n"
                          << "    sharedMem: " << exec.shared_mem << "\n"
                          << "    stream: " << exec.stream << ")\n";
                
                // Print memory changes
                for (const auto& [arg_idx, changes] : exec.changes_by_arg) {
                    std::cout << "  Memory changes for argument " << arg_idx << ":\n";
                    size_t num_to_print = std::min<size_t>(10, changes.size());
                    
                    for (size_t i = 0; i < num_to_print; i++) {
                        const auto& [element_index, values] = changes[i];
                        const auto& [pre_val, post_val] = values;
                        
                        std::cout << "    arg" << arg_idx << " float* " 
                                 << exec.arg_ptrs[arg_idx] << "[" << element_index << "] "
                                 << "changed from " << pre_val 
                                 << " to " << post_val << "\n";
                    }
                    
                    if (changes.size() > 10) {
                        std::cout << "    ... and " << (changes.size() - 10) 
                                 << " more changes\n";
                    }
                }
                break;
            }
            case TimelineEvent::MEMCPY: {
                const auto& op = *static_cast<const MemoryOperation*>(event.data);
                std::cout << "\nhipMemcpy(\n"
                          << "    dst: " << op.dst << "\n"
                          << "    src: " << op.src << "\n"
                          << "    size: " << op.size << " bytes\n"
                          << "    kind: " << memcpyKindToString(op.kind) << ")\n";
                
                // Print first few values that changed
                std::cout << "  Memory changes:\n";
                float* pre_vals = (float*)op.pre_state->data.get();
                float* post_vals = (float*)op.post_state->data.get();
                size_t num_floats = std::min(size_t(5), op.size / sizeof(float));
                
                for (size_t i = 0; i < num_floats; i++) {
                    std::cout << "    [" << i << "]: " << pre_vals[i] 
                             << " -> " << post_vals[i] << "\n";
                }
                break;
            }
            case TimelineEvent::MEMSET: {
                const auto& op = *static_cast<const MemoryOperation*>(event.data);
                std::cout << "\nhipMemset(\n"
                          << "    dst: " << op.dst << "\n"
                          << "    value: " << op.value << "\n"
                          << "    size: " << op.size << " bytes)\n";
                
                // Print first few values that changed
                std::cout << "  Memory changes:\n";
                float* pre_vals = (float*)op.pre_state->data.get();
                float* post_vals = (float*)op.post_state->data.get();
                size_t num_floats = std::min(size_t(5), op.size / sizeof(float));
                
                for (size_t i = 0; i < num_floats; i++) {
                    std::cout << "    [" << i << "]: " << pre_vals[i] 
                             << " -> " << post_vals[i] << "\n";
                }
                break;
            }
        }
    }
}

// Helper to extract argument type from kernel signature
static std::string getArgTypeFromSignature(const std::string& signature, size_t arg_index) {
    size_t start = signature.find('(');
    size_t end = signature.find(')');
    if (start == std::string::npos || end == std::string::npos) {
        return "";
    }
    
    std::string args_str = signature.substr(start + 1, end - start - 1);
    
    // Parse arguments handling nested template brackets
    std::vector<std::string> args;
    std::string current_arg;
    int template_depth = 0;
    
    for (char c : args_str) {
        if (c == '<') {
            template_depth++;
            current_arg += c;
        }
        else if (c == '>') {
            template_depth--;
            current_arg += c;
        }
        else if (c == ',' && template_depth == 0) {
            // Only split on commas outside of template brackets
            if (!current_arg.empty()) {
                // Trim whitespace
                current_arg.erase(0, current_arg.find_first_not_of(" "));
                current_arg.erase(current_arg.find_last_not_of(" ") + 1);
                args.push_back(current_arg);
                current_arg.clear();
            }
        }
        else {
            current_arg += c;
        }
    }
    
    // Add the last argument
    if (!current_arg.empty()) {
        current_arg.erase(0, current_arg.find_first_not_of(" "));
        current_arg.erase(current_arg.find_last_not_of(" ") + 1);
        args.push_back(current_arg);
    }
    
    // Return the requested argument type if index is valid
    if (arg_index < args.size()) {
        return args[arg_index];
    }
    
    return "";
}

// Helper to count number of arguments
static size_t countKernelArgs(void** args) {
    if (!args) return 0;
    
    // For VecAdd kernel, we expect exactly 4 arguments
    // This is a temporary fix - ideally we would parse this from the kernel signature
    return 4;
}

// First define KernelArgInfo
struct KernelArgInfo {
    bool is_vector;
    size_t size;
    // other fields...
};

// Then define KernelInfo which uses KernelArgInfo
struct KernelInfo {
    std::vector<KernelArgInfo> args;
};

// Finally define the kernel registry map
static std::unordered_map<std::string, KernelInfo> kernel_registry;

// When registering kernel arguments
void registerKernelArg(const std::string& kernel_name, size_t arg_index, 
                      bool is_vector, size_t size) {
    auto& info = kernel_registry[kernel_name];
    // Ensure the args vector is large enough
    if (info.args.size() <= arg_index) {
        info.args.resize(arg_index + 1);
    }
    info.args[arg_index].is_vector = is_vector;
    info.args[arg_index].size = size;
}

// Add this helper function to detect vector types
static bool isVectorType(const std::string& type_name) {
    static const std::vector<std::string> vector_types = {
        "float4", "float3", "float2",
        "int4", "int3", "int2",
        "uint4", "uint3", "uint2",
        "double4", "double3", "double2",
        "long4", "long3", "long2",
        "ulong4", "ulong3", "ulong2",
        "char4", "char3", "char2",
        "uchar4", "uchar3", "uchar2",
        "HIP_vector_type"
    };
    
    for (const auto& vtype : vector_types) {
        if (type_name.find(vtype) != std::string::npos) {
            return true;
        }
    }
    return false;
}

// Add this to automatically register kernel arguments when first seen
static void registerKernelIfNeeded(const std::string& kernel_name, const std::string& signature) {
    if (kernel_registry.find(kernel_name) != kernel_registry.end()) {
        return;  // Already registered
    }
    
    // Parse signature to get argument types
    size_t start = signature.find('(');
    size_t end = signature.find(')');
    if (start != std::string::npos && end != std::string::npos) {
        std::string args_str = signature.substr(start + 1, end - start - 1);
        std::stringstream ss(args_str);
        std::string arg_type;
        size_t arg_index = 0;
        
        while (std::getline(ss, arg_type, ',')) {
            // Trim whitespace
            arg_type.erase(0, arg_type.find_first_not_of(" "));
            arg_type.erase(arg_type.find_last_not_of(" ") + 1);
            
            bool is_vector = isVectorType(arg_type);
            size_t size = is_vector ? sizeof(float4) : sizeof(void*);  // Approximate size
            registerKernelArg(kernel_name, arg_index++, is_vector, size);
        }
    }
}

// Add these structures for binary trace file
struct TraceHeader {
    uint32_t magic;  // Magic number to identify file format
    uint32_t version;  // Version number
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
    uint32_t size;  // Size of the event-specific data that follows
};

// Structure to represent timeline events for replay
struct TimelineEvent {
    enum Type {
        KERNEL,
        MEMCPY,
        MEMSET
    } type;
    
    uint64_t order;
    std::vector<char> data;  // Store the actual data, not just a pointer
    std::shared_ptr<KernelExecution> kernel_exec;  // Keep kernel execution data alive
    
    TimelineEvent(Type t, uint64_t o, const void* d, size_t size) 
        : type(t), order(o), data(size) {
        memcpy(data.data(), d, size);
    }
};

// Add these serialization helper functions
struct SerializedKernelExecution {
    void* function_address;
    char kernel_name[256];  // Fixed size buffer for the name
    dim3 grid_dim;
    dim3 block_dim;
    size_t shared_mem;
    hipStream_t stream;
    uint64_t execution_order;
    uint32_t num_changes;  // Number of memory changes
    uint32_t num_args;     // Number of arguments
    // Followed by arg_ptrs data in the file
    // Then followed by changes data
};

struct SerializedMemoryChange {
    int arg_idx;
    size_t element_index;
    float pre_val;
    float post_val;
};

// Class to manage the trace file
class TraceFile {
public:
    TraceFile(const std::string& path) : path_(path) {
        if (path.empty()) {
            // Skip tracing for this process
            return;
        }
        
        trace_file_.open(path, std::ios::binary);
        if (!trace_file_) {
            std::cerr << "Failed to open trace file for writing" << std::endl;
            return;
        }
        std::cout << "Trace file opened successfully: " << path << std::endl;
        TraceHeader header{TraceHeader::MAGIC, TraceHeader::VERSION};
        trace_file_.write(reinterpret_cast<char*>(&header), sizeof(header));
    }

    ~TraceFile() {
        if (trace_file_.is_open()) {
            trace_file_.close();
        }
        // Read and process the trace file
        readAndProcessTrace();
    }

    void writeEvent(TraceEvent::Type type, const void* data, size_t data_size) {
        if (!trace_file_) return;
        
        TraceEvent event;
        event.type = type;
        event.timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
        event.size = data_size;
        
        trace_file_.write(reinterpret_cast<char*>(&event), sizeof(event));
        
        // Use specialized serialization for kernel executions
        if (type == TraceEvent::KERNEL_LAUNCH) {
            writeKernelExecution(*static_cast<const KernelExecution*>(data));
        } else {
            trace_file_.write(reinterpret_cast<const char*>(data), data_size);
        }
    }

    void readAndProcessTrace() {
        std::ifstream in(path_, std::ios::binary);
        if (!in) {
            std::cerr << "Failed to open trace file for reading" << std::endl;
            return;
        }
        
        // Read header
        TraceHeader header;
        in.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (header.magic != TraceHeader::MAGIC || header.version != TraceHeader::VERSION) {
            std::cerr << "Invalid trace file format" << std::endl;
            return;
        }
        
        std::vector<TimelineEvent> timeline;
        
        // Read events
        while (in.good()) {
            TraceEvent event;
            in.read(reinterpret_cast<char*>(&event), sizeof(event));
            if (!in.good()) break;
            
            if (event.type == TraceEvent::KERNEL_LAUNCH) {
                SerializedKernelExecution serialized;
                in.read(reinterpret_cast<char*>(&serialized), sizeof(serialized));
                
                // Create a new KernelExecution with only the serialized data
                auto exec = std::make_shared<KernelExecution>();
                exec->function_address = serialized.function_address;
                exec->kernel_name = serialized.kernel_name;
                exec->grid_dim = serialized.grid_dim;
                exec->block_dim = serialized.block_dim;
                exec->shared_mem = serialized.shared_mem;
                exec->stream = serialized.stream;
                exec->execution_order = serialized.execution_order;
                
                // Read arg_ptrs
                for (uint32_t i = 0; i < serialized.num_args; i++) {
                    void* ptr;
                    in.read(reinterpret_cast<char*>(&ptr), sizeof(void*));
                    exec->arg_ptrs.push_back(ptr);
                }
                
                // Read changes
                for (uint32_t i = 0; i < serialized.num_changes; i++) {
                    SerializedMemoryChange mem_change;
                    in.read(reinterpret_cast<char*>(&mem_change), sizeof(mem_change));
                    
                    exec->changes_by_arg[mem_change.arg_idx].push_back({
                        mem_change.element_index,
                        {mem_change.pre_val, mem_change.post_val}
                    });
                }
                
                // Store the shared_ptr in the timeline
                timeline.emplace_back(TimelineEvent::KERNEL, exec->execution_order, exec.get(), sizeof(*exec));
                timeline.back().kernel_exec = exec;  // Keep the shared_ptr alive
            }
            // ... handle other event types ...
        }
        
        // Print timeline
        std::cout << "\n=== HIP API Execution Trace ===\n";
        std::cout << "Found " << timeline.size() << " events\n";
        
        for (const auto& event : timeline) {
            switch (event.type) {
                case TimelineEvent::KERNEL: {
                    const auto& exec = *event.kernel_exec;
                    std::cout << "\nhipLaunchKernel(\n"
                             << "    kernel: " << exec.kernel_name 
                             << " at " << exec.function_address << "\n"
                             << "    gridDim: " << dim3ToString(exec.grid_dim) << "\n"
                             << "    blockDim: " << dim3ToString(exec.block_dim) << "\n"
                             << "    sharedMem: " << exec.shared_mem << "\n"
                             << "    stream: " << exec.stream << ")\n";
                    
                    // Print memory changes
                    for (const auto& [arg_idx, changes] : exec.changes_by_arg) {
                        std::cout << "  Memory changes for argument " << arg_idx << ":\n";
                        size_t num_to_print = std::min<size_t>(10, changes.size());
                        
                        for (size_t i = 0; i < num_to_print; i++) {
                            const auto& [element_index, values] = changes[i];
                            const auto& [pre_val, post_val] = values;
                            
                            std::cout << "    arg" << arg_idx << " float* " 
                                     << exec.arg_ptrs[arg_idx] << "[" << element_index << "] "
                                     << "changed from " << pre_val 
                                     << " to " << post_val << "\n";
                        }
                        
                        if (changes.size() > 10) {
                            std::cout << "    ... and " << (changes.size() - 10) 
                                     << " more changes\n";
                        }
                    }
                    break;
                }
                // ... handle other event types ...
            }
        }
    }

    void writeKernelExecution(const KernelExecution& exec) {
        SerializedKernelExecution serialized;
        serialized.function_address = exec.function_address;
        strncpy(serialized.kernel_name, exec.kernel_name.c_str(), sizeof(serialized.kernel_name) - 1);
        serialized.kernel_name[sizeof(serialized.kernel_name) - 1] = '\0';
        serialized.grid_dim = exec.grid_dim;
        serialized.block_dim = exec.block_dim;
        serialized.shared_mem = exec.shared_mem;
        serialized.stream = exec.stream;
        serialized.execution_order = exec.execution_order;
        
        // Count total changes
        uint32_t total_changes = 0;
        for (const auto& [arg_idx, changes] : exec.changes_by_arg) {
            total_changes += changes.size();
        }
        serialized.num_changes = total_changes;
        serialized.num_args = exec.arg_ptrs.size();
        
        // Write the main structure
        trace_file_.write(reinterpret_cast<const char*>(&serialized), sizeof(serialized));
        
        // Write arg_ptrs
        for (void* ptr : exec.arg_ptrs) {
            trace_file_.write(reinterpret_cast<const char*>(&ptr), sizeof(void*));
        }
        
        // Write all changes
        for (const auto& [arg_idx, changes] : exec.changes_by_arg) {
            for (const auto& change : changes) {
                SerializedMemoryChange mem_change;
                mem_change.arg_idx = arg_idx;
                mem_change.element_index = change.first;
                mem_change.pre_val = change.second.first;
                mem_change.post_val = change.second.second;
                trace_file_.write(reinterpret_cast<const char*>(&mem_change), sizeof(mem_change));
            }
        }
    }

private:
    std::ofstream trace_file_;
    std::string path_;  // Store the path for reading later
};

std::string getTraceFilePath() {
    static int traceId = 0;
    const char* home = getenv("HOME");
    if (!home) {
        home = "/tmp";
    }

    // Get the binary name from /proc/self/exe
    char selfPath[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", selfPath, sizeof(selfPath) - 1);
    if (len == -1) {
        return std::string(home) + "/hipTracer/unknown-" + std::to_string(traceId++) + ".trace";
    }
    selfPath[len] = '\0';
    
    // Extract just the binary name from the full path
    std::string binaryName = std::string(selfPath);
    size_t lastSlash = binaryName.find_last_of('/');
    if (lastSlash != std::string::npos) {
        binaryName = binaryName.substr(lastSlash + 1);
    }

    // Skip system utilities and only trace actual HIP programs
    static const std::vector<std::string> ignore_list = {
        "grep", "dash", "nm", "x86_64-linux-gnu-nm",
        "ld", "as", "objdump", "readelf", "addr2line"
    };
    
    for (const auto& ignored : ignore_list) {
        if (binaryName.find(ignored) != std::string::npos) {
            return "";  // Return empty string to skip tracing for these programs
        }
    }

    // Create the hipTracer directory if it doesn't exist
    std::string tracerDir = std::string(home) + "/hipTracer";
    mkdir(tracerDir.c_str(), 0755);

    // Find the next available trace ID
    std::string basePath = tracerDir + "/" + binaryName + "-";
    while (access((basePath + std::to_string(traceId) + ".trace").c_str(), F_OK) != -1) {
        traceId++;
    }

    return basePath + std::to_string(traceId++) + ".trace";
}

// Global instance of TraceFile with dynamic path
static TraceFile trace_file(getTraceFilePath());

// Modify hipLaunchKernel_impl to write to trace instead of storing in memory
static hipError_t hipLaunchKernel_impl(const void *function_address, dim3 numBlocks,
                                     dim3 dimBlocks, void **args, size_t sharedMemBytes,
                                     hipStream_t stream) {
    std::cout << "hipLaunchKernel(\n"
              << "    function=" << function_address 
              << "\n    numBlocks=" << dim3ToString(numBlocks)
              << "\n    dimBlocks=" << dim3ToString(dimBlocks)
              << "\n    sharedMem=" << sharedMemBytes
              << "\n    stream=" << (void*)stream << "\n";
              
    // Print kernel arguments with types
    std::string kernelName = getKernelName(function_address);
    std::string signature = getKernelSignature(function_address);
    registerKernelIfNeeded(kernelName, signature);
    printKernelArgs(args, kernelName, function_address);
    
    // Create execution record
    KernelExecution exec;
    exec.function_address = (void*)function_address;
    exec.kernel_name = kernelName;
    exec.grid_dim = numBlocks;
    exec.block_dim = dimBlocks;
    exec.shared_mem = sharedMemBytes;
    exec.stream = stream;
    static uint64_t kernel_count = 0;
    exec.execution_order = kernel_count++;

    // Get kernel name and look up kernel info
    auto kernel_it = kernel_registry.find(kernelName);
    bool have_kernel_info = (kernel_it != kernel_registry.end());

    // Store argument pointers for later analysis
    if (args) {
        std::string signature = getKernelSignature(function_address);
        size_t num_args = countKernelArgs(args);
        
        std::cout << "\nProcessing " << num_args << " kernel arguments\n";
        std::cout << "Kernel signature: " << signature << "\n";
        
        for (size_t i = 0; i < num_args; i++) {
            std::cout << "\nProcessing argument " << i << ":\n";
            
            if (!args[i]) {
                std::cout << "  Argument is nullptr, skipping\n";
                continue;
            }
            
            // Get argument type
            std::string arg_type = getArgTypeFromSignature(signature, i);
            std::cout << "  Argument type: '" << arg_type << "'\n";
            
            // Check if this is a vector type
            bool is_vector = isVectorType(arg_type);
            std::cout << "  Is vector type: " << (is_vector ? "yes" : "no") << "\n";
            
            void* arg_ptr = nullptr;
            size_t arg_size = 0;
            
            try {
                std::cout << "  Raw argument address: " << args[i] << "\n";
                
                if (is_vector) {
                    // For vector types, use the argument directly
                    arg_ptr = args[i];
                    arg_size = 16;  // HIP_vector_type<float,2> is 16 bytes
                    std::cout << "  Vector argument of size " << arg_size << "\n";
                } else if (arg_type.find("*") != std::string::npos) {
                    // For pointer types, dereference to get the actual pointer
                    arg_ptr = *(void**)args[i];
                    arg_size = sizeof(void*);
                    std::cout << "  Pointer argument pointing to " << arg_ptr << "\n";
                } else {
                    // For scalar types, use directly
                    arg_ptr = args[i];
                    arg_size = 16;  // HIP_vector_type is passed by value
                    std::cout << "  Scalar argument of size " << arg_size << "\n";
                }
                
                std::cout << "  Adding argument to execution record\n";
                exec.arg_ptrs.push_back(arg_ptr);
                exec.arg_sizes.push_back(arg_size);
                
                // Only track GPU memory for pointer types
                if (!is_vector && arg_type.find("*") != std::string::npos) {
                    std::cout << "  Checking for GPU memory allocation\n";
                    auto [base_ptr, info] = findContainingAllocation(arg_ptr);
                    if (base_ptr && info) {
                        std::cout << "  Found GPU allocation at " << base_ptr 
                                 << " of size " << info->size << "\n";
                        createShadowCopy(base_ptr, *info);
                        exec.pre_state.emplace(base_ptr, 
                            MemoryState(info->shadow_copy.get(), info->size));
                    }
                }
                
            } catch (...) {
                std::cerr << "  Failed to process argument " << i << std::endl;
                continue;
            }
        }
    }

    // Rest of the function remains the same...
    hipError_t result = get_real_hipLaunchKernel()(function_address, numBlocks, 
                                                  dimBlocks, args, sharedMemBytes, stream);
    get_real_hipDeviceSynchronize()();
    
    // Record post-execution state and changes...
    for (const auto& [ptr, pre_state] : exec.pre_state) {
        auto [base_ptr, info] = findContainingAllocation(ptr);
        if (base_ptr && info) {
            createShadowCopy(base_ptr, *info);
            exec.post_state.emplace(ptr, 
                MemoryState(info->shadow_copy.get(), info->size));
        }
    }

    recordMemoryChanges(exec);
    
    // Write to trace file instead of storing in memory
    trace_file.writeEvent(TraceEvent::KERNEL_LAUNCH, &exec, sizeof(exec));
    
    return result;
}

} // namespace

extern "C" {

hipError_t hipMalloc(void **ptr, size_t size) {
    std::cout << "hipMalloc(ptr=" << (void*)ptr << ", size=" << size << ")\n";
    
    hipError_t result = get_real_hipMalloc()(ptr, size);
    
    if (result == hipSuccess && ptr && *ptr) {
        // Track the allocation
        gpu_allocations.emplace(*ptr, AllocationInfo(size));
        std::cout << "Tracking GPU allocation at " << *ptr 
                  << " of size " << size << std::endl;
    }
    
    return result;
}

hipError_t hipLaunchKernel(const void *function_address, dim3 numBlocks,
                          dim3 dimBlocks, void **args, size_t sharedMemBytes,
                          hipStream_t stream) {
    return hipLaunchKernel_impl(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream);
}

hipError_t hipDeviceSynchronize(void) {
    return get_real_hipDeviceSynchronize()();
}

hipError_t hipFree(void* ptr) {
    if (ptr) {
        auto it = gpu_allocations.find(ptr);
        if (it != gpu_allocations.end()) {
            std::cout << "Removing tracked GPU allocation at " << ptr 
                     << " of size " << it->second.size << std::endl;
            gpu_allocations.erase(it);
        }
    }
    return get_real_hipFree()(ptr);
}

__attribute__((visibility("default")))
hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind) {
    return hipMemcpy_impl(dst, src, sizeBytes, kind);
}

__attribute__((visibility("default")))
hipError_t hipMemset(void *dst, int value, size_t sizeBytes) {
    return hipMemset_impl(dst, value, sizeBytes);
}

} // extern "C"