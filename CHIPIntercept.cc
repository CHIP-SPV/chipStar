/*
 * Copyright (c) 2021-24 chipStar developers
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

#define __HIP_PLATFORM_SPIRV__
#include "hip/hip_runtime_api.h"
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <link.h>
#include <map>
#include <unordered_map>
#include <memory>
#include <cstring>  // For memcpy
#include <algorithm>  // for std::sort, std::min
#include <utility>   // for std::pair

// At the top level (outside any namespace)
struct MemoryState {
    std::unique_ptr<char[]> data;
    size_t size;
    
    MemoryState(size_t s) : data(new char[s]), size(s) {}
    MemoryState(const char* src, size_t s) : data(new char[s]), size(s) {
        memcpy(data.get(), src, s);
    }
    
    // Add default constructor required by std::map
    MemoryState() : size(0) {}
};

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
    
    // Add this to store changes grouped by argument
    std::map<int, std::vector<std::pair<size_t, std::pair<float, float>>>> changes_by_arg;
};

// Track all kernel executions
static std::vector<KernelExecution> kernel_executions;

// Add this after the KernelExecution struct
enum class MemoryOpType {
    COPY,
    SET
};

struct MemoryOperation {
    MemoryOpType type;
    void* dst;
    const void* src;  // Only used for COPY
    size_t size;
    int value;        // Only used for SET
    hipMemcpyKind kind;  // Only used for COPY
    uint64_t execution_order;
    
    // Memory state before/after operation
    std::shared_ptr<MemoryState> pre_state;
    std::shared_ptr<MemoryState> post_state;
};

// Track all memory operations
static std::vector<MemoryOperation> memory_operations;

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

// Helper to print kernel arguments
static void printKernelArgs(void** args, const std::string& kernelName) {
    // Extract function signature from kernel name
    if (kernelName.find("MatrixMul") != std::string::npos) {
        // Known signature: (float const*, float const*, float*, unsigned int, unsigned int, unsigned int)
        float const* A = *(float const**)args[0];
        float const* B = *(float const**)args[1];
        float* C = *(float**)args[2];
        unsigned int M = *(unsigned int*)args[3];
        unsigned int N = *(unsigned int*)args[4];
        unsigned int K = *(unsigned int*)args[5];
        
        std::cout << "    args:\n"
                  << "      A: float const* = " << (void*)A << "\n"
                  << "      B: float const* = " << (void*)B << "\n"
                  << "      C: float* = " << (void*)C << "\n"
                  << "      M: unsigned int = " << M << "\n"
                  << "      N: unsigned int = " << N << "\n"
                  << "      K: unsigned int = " << K << "\n";
    } else {
        // For unknown kernels, print raw arg pointers
        std::cout << "    raw args:\n";
        for (int i = 0; args[i] != nullptr; i++) {
            std::cout << "      arg[" << i << "]: <unknown_type> = " 
                     << args[i] << " -> " << *(void**)args[i] << "\n";
        }
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
                    size_t num_to_print = std::min(size_t(10), changes.size());
                    
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
    std::cout << "hipLaunchKernel(\n"
              << "    function=" << function_address 
              << "\n    numBlocks=" << dim3ToString(numBlocks)
              << "\n    dimBlocks=" << dim3ToString(dimBlocks)
              << "\n    sharedMem=" << sharedMemBytes
              << "\n    stream=" << (void*)stream << "\n";
    
    KernelExecution exec;
    exec.function_address = (void*)function_address;
    exec.kernel_name = getKernelName(function_address);
    exec.grid_dim = numBlocks;
    exec.block_dim = dimBlocks;
    exec.shared_mem = sharedMemBytes;
    exec.stream = stream;
    static uint64_t kernel_count = 0;
    exec.execution_order = kernel_count++;
    
    // Clear any previous state
    exec.pre_state.clear();
    exec.post_state.clear();
    exec.changes.clear();
    exec.arg_ptrs.clear();
    
    // Create shadow copies of GPU memory before kernel execution
    if (args) {
        // First 3 args are pointers (float* or float const*)
        for (int i = 0; i < 3; i++) {
            if (!args[i]) continue;
            
            void* arg_ptr = *(void**)args[i];
            if (!arg_ptr) continue;
            
            exec.arg_ptrs.push_back(arg_ptr);  // Store argument pointer
            
            // Try to find if this points to GPU memory
            auto [base_ptr, info] = findContainingAllocation(arg_ptr);
            if (base_ptr && info) {
                // Create shadow copy first
                createShadowCopy(base_ptr, *info);
                // Then record pre-execution state using the shadow copy
                exec.pre_state.emplace(base_ptr, 
                    MemoryState(info->shadow_copy.get(), info->size));
                
                std::cout << "Created shadow copy for GPU memory at " 
                          << base_ptr << " referenced by arg " << i << std::endl;
            }
        }
    }
    
    // Launch the kernel
    hipError_t result = get_real_hipLaunchKernel()(function_address, numBlocks, 
                                                  dimBlocks, args, sharedMemBytes, stream);
    
    // Synchronize and capture post-execution state
    get_real_hipDeviceSynchronize()();
    
    // Record post-execution state
    for (const auto& [ptr, pre_state] : exec.pre_state) {
        auto [base_ptr, info] = findContainingAllocation(ptr);
        if (base_ptr && info) {
            createShadowCopy(base_ptr, *info);
            exec.post_state.emplace(ptr, 
                MemoryState(info->shadow_copy.get(), info->size));
        }
    }
    
    // Record changes
    recordMemoryChanges(exec);
    
    // Store the execution record
    kernel_executions.push_back(std::move(exec));
    
    return result;
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