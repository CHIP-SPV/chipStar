#include "Interceptor.hh"
#include "Tracer.hh"
using namespace hip_intercept;

std::unordered_map<void*, AllocationInfo> gpu_allocations;

std::pair<void*, AllocationInfo*> findContainingAllocation(void* ptr) {
    for (auto& [base_ptr, info] : gpu_allocations) {
        char* start = static_cast<char*>(base_ptr);
        char* end = start + info.size;
        if (ptr >= base_ptr && ptr < end) {
            return std::make_pair(base_ptr, &info);
        }
    }
    return std::make_pair(nullptr, nullptr);
}

namespace {
// Function pointer types
typedef hipError_t (*hipMalloc_fn)(void**, size_t);
typedef hipError_t (*hipMemcpy_fn)(void*, const void*, size_t, hipMemcpyKind);
typedef hipError_t (*hipLaunchKernel_fn)(const void*, dim3, dim3, void**, size_t, hipStream_t);
typedef hipError_t (*hipDeviceSynchronize_fn)(void);
typedef hipError_t (*hipFree_fn)(void*);
typedef hipError_t (*hipMemset_fn)(void*, int, size_t);
typedef hipError_t (*hipModuleLaunchKernel_fn)(hipFunction_t, unsigned int,
                                              unsigned int, unsigned int,
                                              unsigned int, unsigned int,
                                              unsigned int, unsigned int,
                                              hipStream_t, void**, void**);
typedef hipError_t (*hipModuleGetFunction_fn)(hipFunction_t*, hipModule_t, const char*);

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

hipModuleLaunchKernel_fn get_real_hipModuleLaunchKernel() {
    static auto fn = (hipModuleLaunchKernel_fn)getOriginalFunction("hipModuleLaunchKernel");
    return fn;
}

hipModuleGetFunction_fn get_real_hipModuleGetFunction() {
    static auto fn = (hipModuleGetFunction_fn)getOriginalFunction("hipModuleGetFunction");
    return fn;
}

// Helper to find which allocation a pointer belongs to
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

static int getArgumentIndex(void* ptr, const std::vector<void*>& arg_ptrs) {
    for (size_t i = 0; i < arg_ptrs.size(); i++) {
        if (arg_ptrs[i] == ptr) return i;
    }
    return -1;
}

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
    std::cout << "Kernel name: " << kernelName << std::endl;
    printKernelArgs(args, kernelName, function_address);
    
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
            bool is_vector = isVectorType(arg_type);
            
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

// Add a map to store function names for RTC kernels
static std::unordered_map<hipFunction_t, std::string> rtc_kernel_names;

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

hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX,
                                unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY,
                                unsigned int blockDimZ, unsigned int sharedMemBytes,
                                hipStream_t stream, void** kernelParams,
                                void** extra) {
    std::cout << "\n=== INTERCEPTED hipModuleLaunchKernel ===\n";
    std::cout << "hipModuleLaunchKernel(\n"
              << "    function=" << f
              << "\n    gridDim={" << gridDimX << "," << gridDimY << "," << gridDimZ << "}"
              << "\n    blockDim={" << blockDimX << "," << blockDimY << "," << blockDimZ << "}"
              << "\n    sharedMem=" << sharedMemBytes
              << "\n    stream=" << stream << "\n";

    // Create execution record
    hip_intercept::KernelExecution exec;
    exec.function_address = f;
    exec.kernel_name = rtc_kernel_names.count(f) ? 
        rtc_kernel_names[f] : "unknown_rtc_kernel";
    std::cout << "Kernel name: " << exec.kernel_name << std::endl;
    exec.grid_dim = {gridDimX, gridDimY, gridDimZ};
    exec.block_dim = {blockDimX, blockDimY, blockDimZ};
    exec.shared_mem = sharedMemBytes;
    exec.stream = stream;
    static uint64_t kernel_count = 0;
    exec.execution_order = kernel_count++;

    // Store argument pointers and capture pre-execution state
    if (kernelParams) {
        size_t num_args = countKernelArgs(kernelParams);
        for (size_t i = 0; i < num_args; i++) {
            if (!kernelParams[i]) continue;
        
            std::string arg_type = "float*"; //getArgTypeFromSignature(getKernelSignature(f), i);
            bool is_vector = isVectorType(arg_type);
            
            void* arg_ptr = nullptr;
            size_t arg_size = 0;
            
            if (is_vector) {
                arg_ptr = kernelParams[i];
                arg_size = 16;  // HIP_vector_type size
            } else if (arg_type.find("*") != std::string::npos) {
                arg_ptr = *(void**)kernelParams[i];
                arg_size = sizeof(void*);
            } else {
                arg_ptr = kernelParams[i];
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
    hipError_t result = get_real_hipModuleLaunchKernel()(f, gridDimX, gridDimY, gridDimZ,
                                                        blockDimX, blockDimY, blockDimZ,
                                                        sharedMemBytes, stream,
                                                        kernelParams, extra);
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

hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* kname) {
    std::cout << "\n=== INTERCEPTED hipModuleGetFunction ===\n";
    std::cout << "hipModuleGetFunction(function=" << function 
              << ", module=" << module 
              << ", kname=" << kname << ")\n";
              
    hipError_t result = get_real_hipModuleGetFunction()(function, module, kname);
    
    if (result == hipSuccess && function && *function) {
        // Store the kernel name for this function handle
        rtc_kernel_names[*function] = kname;
        std::cout << "Stored RTC kernel name '" << kname 
                  << "' for function handle " << *function << std::endl;
    }
    
    return result;
}

} // extern "C"