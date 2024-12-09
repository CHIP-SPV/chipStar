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
#include "Tracer.hh"
using namespace hip_intercept;

std::unordered_map<void*, AllocationInfo> gpu_allocations;

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

// Helper to find which allocation a pointer belongs to
static std::pair<void*, AllocationInfo*> findContainingAllocation(void* ptr) {
    for (auto& [base_ptr, info] : gpu_allocations) {
        char* start = static_cast<char*>(base_ptr);
        char* end = start + info.size;
        if (ptr >= base_ptr && ptr < end) {
            return std::make_pair(base_ptr, &info);
        }
    }
    return std::make_pair(nullptr, nullptr);
}

// Keep the static function definition
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
static void recordMemoryChanges(hip_intercept::KernelExecution& exec) {
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

// Add this function definition before it's used
static size_t countKernelArgs(void** args) {
    if (!args) return 0;
    
    size_t count = 0;
    while (args[count] != nullptr) {
        count++;
        // Safety check to prevent infinite loop
        if (count > 100) {  // reasonable max number of kernel arguments
            std::cerr << "Warning: Exceeded maximum expected kernel arguments\n";
            break;
        }
    }
    return count;
}

// Move the implementation functions here
static hipError_t hipMemcpy_impl(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind) {
    std::cout << "\n=== INTERCEPTED hipMemcpy ===\n";
    std::cout << "hipMemcpy(dst=" << dst << ", src=" << src 
              << ", size=" << sizeBytes << ", kind=" << memcpyKindToString(kind) << ")\n";
    
    hip_intercept::MemoryOperation op;
    op.type = hip_intercept::MemoryOpType::COPY;
    op.dst = dst;
    op.src = src;
    op.size = sizeBytes;
    op.kind = kind;
    static uint64_t op_count = 0;
    op.execution_order = op_count++;
    
    // Initialize pre_state and post_state
    op.pre_state = std::make_shared<hip_intercept::MemoryState>(sizeBytes);
    op.post_state = std::make_shared<hip_intercept::MemoryState>(sizeBytes);
    
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
    
    // Record operation using Tracer
    Tracer::instance().recordMemoryOperation(op);
    return result;
}

static hipError_t hipMemset_impl(void *dst, int value, size_t sizeBytes) {
    std::cout << "hipMemset(dst=" << dst << ", value=" << value 
              << ", size=" << sizeBytes << ")\n";
    
    hip_intercept::MemoryOperation op;
    op.type = hip_intercept::MemoryOpType::SET;
    op.dst = dst;
    op.size = sizeBytes;
    op.value = value;
    static uint64_t op_count = 0;
    op.execution_order = op_count++;
    
    // Initialize states
    op.pre_state = std::make_shared<hip_intercept::MemoryState>(sizeBytes);
    op.post_state = std::make_shared<hip_intercept::MemoryState>(sizeBytes);
    
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
    
    // Record operation using Tracer
    Tracer::instance().recordMemoryOperation(op);
    return result;
}

// Remove the 'static' keyword since this is implementing the forward-declared function
std::string getArgTypeFromSignature(const std::string& signature, size_t arg_index) {
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

// Modify hipLaunchKernel_impl to use Tracer directly:
static hipError_t hipLaunchKernel_impl(const void *function_address, dim3 numBlocks,
                                     dim3 dimBlocks, void **args, size_t sharedMemBytes,
                                     hipStream_t stream) {
    std::cout << "hipLaunchKernel(\n"
              << "    function=" << function_address 
              << "\n    numBlocks=" << dim3ToString(numBlocks)
              << "\n    dimBlocks=" << dim3ToString(dimBlocks)
              << "\n    sharedMem=" << sharedMemBytes
              << "\n    stream=" << (void*)stream << "\n";
              
    // Get kernel name and print args using Tracer
    std::string kernelName = getKernelName(function_address);
    Tracer::instance().printKernelArgs(args, kernelName, function_address);
    
    // Create execution record
    hip_intercept::KernelExecution exec;
    exec.function_address = (void*)function_address;
    exec.kernel_name = kernelName;
    exec.grid_dim = numBlocks;
    exec.block_dim = dimBlocks;
    exec.shared_mem = sharedMemBytes;
    exec.stream = stream;
    static uint64_t kernel_count = 0;
    exec.execution_order = kernel_count++;

    // Store argument pointers and capture pre-execution state
    if (args) {
        size_t num_args = countKernelArgs(args);
        for (size_t i = 0; i < num_args; i++) {
            if (!args[i]) continue;
            
            std::string arg_type = getArgTypeFromSignature(getKernelSignature(function_address), i);
            bool is_vector = Tracer::instance().isVectorType(arg_type);
            
            void* arg_ptr = nullptr;
            size_t arg_size = 0;
            
            if (is_vector) {
                arg_ptr = args[i];
                arg_size = 16;  // HIP_vector_type size
            } else if (arg_type.find("*") != std::string::npos) {
                arg_ptr = *(void**)args[i];
                arg_size = sizeof(void*);
            } else {
                arg_ptr = args[i];
                arg_size = 16;
            }
            
            exec.arg_ptrs.push_back(arg_ptr);
            exec.arg_sizes.push_back(arg_size);
            
            if (!is_vector && arg_type.find("*") != std::string::npos) {
                auto [base_ptr, info] = findContainingAllocation(arg_ptr);
                if (base_ptr && info) {
                    createShadowCopy(base_ptr, *info);
                    exec.pre_state.emplace(base_ptr, 
                        hip_intercept::MemoryState(info->shadow_copy.get(), info->size));
                }
            }
        }
    }

    // Launch kernel
    hipError_t result = get_real_hipLaunchKernel()(function_address, numBlocks, 
                                                  dimBlocks, args, sharedMemBytes, stream);
    get_real_hipDeviceSynchronize()();
    
    // Capture post-execution state
    for (const auto& [ptr, pre_state] : exec.pre_state) {
        auto [base_ptr, info] = findContainingAllocation(ptr);
        if (base_ptr && info) {
            createShadowCopy(base_ptr, *info);
            exec.post_state.emplace(ptr, 
                hip_intercept::MemoryState(info->shadow_copy.get(), info->size));
        }
    }

    recordMemoryChanges(exec);
    
    // Record kernel execution using Tracer
    Tracer::instance().recordKernelLaunch(exec);
    
    return result;
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