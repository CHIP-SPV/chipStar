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

namespace {
// Function pointer types
typedef hipError_t (*hipGetDeviceProperties_fn)(hipDeviceProp_t*, int);
typedef hipError_t (*hipMalloc_fn)(void**, size_t);
typedef hipError_t (*hipMemcpy_fn)(void*, const void*, size_t, hipMemcpyKind);
typedef hipError_t (*hipLaunchKernel_fn)(const void*, dim3, dim3, void**, size_t, hipStream_t);
typedef hipError_t (*hipDeviceSynchronize_fn)(void);
typedef hipError_t (*hipGetDevice_fn)(int*);
typedef hipError_t (*hipSetDevice_fn)(int);
typedef hipError_t (*hipEventCreate_fn)(hipEvent_t*);
typedef hipError_t (*hipEventDestroy_fn)(hipEvent_t);
typedef hipError_t (*hipEventRecord_fn)(hipEvent_t, hipStream_t);
typedef hipError_t (*hipEventElapsedTime_fn)(float*, hipEvent_t, hipEvent_t);
typedef hipError_t (*hipFree_fn)(void*);
typedef hipError_t (*hipGetLastError_fn)(void);
typedef const char* (*hipGetErrorString_fn)(hipError_t);
typedef hipError_t (*hipLaunchKernelGGL_fn)(const void*, dim3, dim3, size_t, hipStream_t, ...);

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
hipGetDeviceProperties_fn get_real_hipGetDeviceProperties() {
    static auto fn = (hipGetDeviceProperties_fn)dlsym(RTLD_NEXT, "hipGetDevicePropertiesR0600");
    return fn;
}

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

hipGetDevice_fn get_real_hipGetDevice() {
    static auto fn = (hipGetDevice_fn)getOriginalFunction("hipGetDevice");
    return fn;
}

hipSetDevice_fn get_real_hipSetDevice() {
    static auto fn = (hipSetDevice_fn)getOriginalFunction("hipSetDevice");
    return fn;
}

hipEventCreate_fn get_real_hipEventCreate() {
    static auto fn = (hipEventCreate_fn)dlsym(RTLD_NEXT, "hipEventCreate");
    return fn;
}

hipEventDestroy_fn get_real_hipEventDestroy() {
    static auto fn = (hipEventDestroy_fn)getOriginalFunction("hipEventDestroy");
    return fn;
}

hipEventRecord_fn get_real_hipEventRecord() {
    static auto fn = (hipEventRecord_fn)getOriginalFunction("hipEventRecord");
    return fn;
}

hipEventElapsedTime_fn get_real_hipEventElapsedTime() {
    static auto fn = (hipEventElapsedTime_fn)getOriginalFunction("hipEventElapsedTime");
    return fn;
}

hipFree_fn get_real_hipFree() {
    static auto fn = (hipFree_fn)getOriginalFunction("hipFree");
    return fn;
}

hipGetLastError_fn get_real_hipGetLastError() {
    static auto fn = (hipGetLastError_fn)getOriginalFunction("hipGetLastError");
    return fn;
}

hipGetErrorString_fn get_real_hipGetErrorString() {
    static auto fn = (hipGetErrorString_fn)getOriginalFunction("hipGetErrorString");
    return fn;
}

hipLaunchKernelGGL_fn get_real_hipLaunchKernelGGL() {
    static auto fn = (hipLaunchKernelGGL_fn)getOriginalFunction("hipLaunchKernelGGL");
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

// Template function needs to be outside both namespace and extern "C"
template <typename F, typename... Args>
hipError_t hipLaunchKernelGGL_impl(F func, dim3 gridDim, dim3 blockDim, 
                                  size_t sharedMem, hipStream_t stream,
                                  Args... args) {
    std::cout << "hipLaunchKernelGGL(\n"
              << "    func=" << (void*)func << "\n"
              << "    gridDim=" << dim3ToString(gridDim) << "\n"
              << "    blockDim=" << dim3ToString(blockDim) << "\n"
              << "    sharedMem=" << sharedMem << "\n"
              << "    stream=" << (void*)stream << "\n"
              << "    arg_types=" << getArgTypes<Args...>() << "\n";

    // Print argument values using proper fold expression
    int argIndex = 0;
    ((std::cout << "    arg[" << argIndex++ << "]=" << (void*)&args << "\n"), ...);
              
    std::cout << ")\n";

    return get_real_hipLaunchKernelGGL()(func, gridDim, blockDim, sharedMem, stream,
                                        std::forward<Args>(args)...);
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
    }
}

// Add these new structures at the top of the anonymous namespace after the includes
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
    std::map<void*, MemoryState> pre_state;
    std::map<void*, MemoryState> post_state;
    std::vector<std::pair<void*, size_t>> changes;
    std::vector<void*> arg_ptrs;
};

// Track all kernel executions
static std::vector<KernelExecution> kernel_executions;

// Add this helper function
static void recordMemoryChanges(KernelExecution& exec) {
    for (const auto& [ptr, pre] : exec.pre_state) {
        const auto& post = exec.post_state[ptr];
        
        // Compare pre and post states
        for (size_t i = 0; i < pre.size; i += sizeof(float)) {
            float* pre_val = (float*)(pre.data.get() + i);
            float* post_val = (float*)(post.data.get() + i);
            
            if (*pre_val != *post_val) {
                exec.changes.push_back({(char*)ptr + i, i});
            }
        }
    }
}

// Add this helper function near the other helpers
static int getArgumentIndex(void* ptr, const std::vector<void*>& arg_ptrs) {
    for (size_t i = 0; i < arg_ptrs.size(); i++) {
        if (arg_ptrs[i] == ptr) return i;
    }
    return -1;
}

// Replace the printKernelSummary function
static void __attribute__((destructor)) printKernelSummary() {
    std::cout << "\n=== Kernel Execution Summary ===\n";
    
    for (const auto& exec : kernel_executions) {
        std::cout << "\nKernel: " << exec.kernel_name 
                  << " at " << exec.function_address << "\n";
        
        // Track changes per argument
        std::map<int, std::vector<std::pair<size_t, std::pair<float, float>>>> changes_by_arg;
        
        // Group changes by argument
        for (const auto& [ptr, offset] : exec.changes) {
            void* base_ptr = (char*)ptr - offset;
            float pre_val = *(float*)(exec.pre_state.at(base_ptr).data.get() + offset);
            float post_val = *(float*)(exec.post_state.at(base_ptr).data.get() + offset);
            
            int arg_idx = getArgumentIndex(base_ptr, exec.arg_ptrs);
            if (arg_idx >= 0) {
                size_t element_index = offset / sizeof(float);
                changes_by_arg[arg_idx].push_back({element_index, {pre_val, post_val}});
            }
        }
        
        // Print up to 10 changes for each argument
        for (const auto& [arg_idx, changes] : changes_by_arg) {
            std::cout << "\n  Changes for argument " << arg_idx << ":\n";
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
    }
}

extern "C" {

hipError_t hipGetDeviceProperties(hipDeviceProp_t* props, int deviceId) {
    return get_real_hipGetDeviceProperties()(props, deviceId);
}

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

hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind) {
    return get_real_hipMemcpy()(dst, src, sizeBytes, kind);
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
    exec.kernel_name = "MatrixMul"; // You might want to make this more dynamic
    
    // Create shadow copies of GPU memory before kernel execution
    if (args) {
        // First 3 args are pointers (float* or float const*)
        for (int i = 0; i < 3; i++) {
            if (!args[i]) continue;
            
            void* arg_ptr = *(void**)args[i];
            if (!arg_ptr) continue;
            
            // Try to find if this points to GPU memory
            auto [base_ptr, info] = findContainingAllocation(arg_ptr);
            if (base_ptr && info) {
                // Record pre-execution state
                exec.pre_state.emplace(base_ptr, 
                    MemoryState(info->shadow_copy.get(), info->size));
                
                createShadowCopy(base_ptr, *info);
                std::cout << "Created shadow copy for GPU memory at " 
                         << base_ptr << " referenced by arg " << i << std::endl;
            }
        }
    }
    
    // Store the first 3 pointer arguments
    if (args) {
        for (int i = 0; i < 3; i++) {
            if (!args[i]) continue;
            void* arg_ptr = *(void**)args[i];
            exec.arg_ptrs.push_back(arg_ptr);
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

hipError_t hipGetDevice(int *deviceId) {
    return get_real_hipGetDevice()(deviceId);
}

hipError_t hipSetDevice(int deviceId) {
    return get_real_hipSetDevice()(deviceId);
}

hipError_t hipEventCreate(hipEvent_t* event) {
    return get_real_hipEventCreate()(event);
}

hipError_t hipEventDestroy(hipEvent_t event) {
    return get_real_hipEventDestroy()(event);
}

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
    return get_real_hipEventRecord()(event, stream);
}

hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) {
    return get_real_hipEventElapsedTime()(ms, start, stop);
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

hipError_t hipGetLastError(void) {
    return get_real_hipGetLastError()();
}

const char* hipGetErrorString(hipError_t error) {
    return get_real_hipGetErrorString()(error);
}

} // extern "C"
} // namespace