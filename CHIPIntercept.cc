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
            << "    arg_types=" << getArgTypes<Args...>() << ")\n";

  return get_real_hipLaunchKernelGGL()(func, gridDim, blockDim, sharedMem, stream,
                           std::forward<Args>(args)...);
}

extern "C" {

hipError_t hipGetDeviceProperties(hipDeviceProp_t* props, int deviceId) {
    std::cout << "hipGetDeviceProperties(props=" << (void*)props << ", deviceId=" << deviceId << ")\n";
    return get_real_hipGetDeviceProperties()(props, deviceId);
}

hipError_t hipMalloc(void **ptr, size_t size) {
    std::cout << "hipMalloc(ptr=" << (void*)ptr << ", size=" << size << ")\n";
    return get_real_hipMalloc()(ptr, size);
}

hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind) {
    std::cout << "hipMemcpy(dst=" << dst << ", src=" << src 
              << ", sizeBytes=" << sizeBytes 
              << ", kind=" << memcpyKindToString(kind) << ")\n";
    return get_real_hipMemcpy()(dst, src, sizeBytes, kind);
}

hipError_t hipLaunchKernel(const void *function_address, dim3 numBlocks,
                          dim3 dimBlocks, void **args, size_t sharedMemBytes,
                          hipStream_t stream) {
  std::cout << "hipLaunchKernel(function=" << function_address 
            << ", numBlocks=" << dim3ToString(numBlocks)
            << ", dimBlocks=" << dim3ToString(dimBlocks)
            << ", args=" << (void*)args 
            << ", sharedMem=" << sharedMemBytes
            << ", stream=" << (void*)stream << ")\n";
  return get_real_hipLaunchKernel()(function_address, numBlocks, dimBlocks, args, 
                        sharedMemBytes, stream);
}

hipError_t hipDeviceSynchronize(void) {
  std::cout << "hipDeviceSynchronize()\n";
  return get_real_hipDeviceSynchronize()();
}

hipError_t hipGetDevice(int *deviceId) {
  std::cout << "hipGetDevice(deviceId=" << (void*)deviceId << ")\n";
  return get_real_hipGetDevice()(deviceId);
}

hipError_t hipSetDevice(int deviceId) {
  std::cout << "hipSetDevice(deviceId=" << deviceId << ")\n";
  return get_real_hipSetDevice()(deviceId);
}

hipError_t hipEventCreate(hipEvent_t* event) {
  std::cout << "hipEventCreate(event=" << (void*)event << ")\n";
  return get_real_hipEventCreate()(event);
}

hipError_t hipEventDestroy(hipEvent_t event) {
  std::cout << "hipEventDestroy(event=" << (void*)event << ")\n";
  return get_real_hipEventDestroy()(event);
}

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  std::cout << "hipEventRecord(event=" << (void*)event << ", stream=" << (void*)stream << ")\n";
  return get_real_hipEventRecord()(event, stream);
}

hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) {
  std::cout << "hipEventElapsedTime(ms=" << (void*)ms << ", start=" << (void*)start << ", stop=" << (void*)stop << ")\n";
  return get_real_hipEventElapsedTime()(ms, start, stop);
}

hipError_t hipFree(void* ptr) {
  std::cout << "hipFree(ptr=" << ptr << ")\n";
  return get_real_hipFree()(ptr);
}

hipError_t hipGetLastError(void) {
  std::cout << "hipGetLastError()\n";
  return get_real_hipGetLastError()();
}

const char* hipGetErrorString(hipError_t error) {
  std::cout << "hipGetErrorString(error=" << (int)error << ")\n";
  return get_real_hipGetErrorString()(error);
}

} // extern "C"
} // namespace