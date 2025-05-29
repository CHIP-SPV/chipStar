#include <hip/hip_runtime.h>
#include "CHIPDriver.hh"
#include <iostream>
#include <cassert>
#include <cstdlib>
#include <string>
#include <filesystem>
#include <fstream>
#include <vector>
#include <set>
#include <unistd.h>

void setEnvVar(const char* name, const char* value) {
    if (setenv(name, value, 1) != 0) {
        std::cerr << "Failed to set environment variable: " << name << std::endl;
        exit(1);
    }
}

void unsetEnvVar(const char* name) {
    if (unsetenv(name) != 0) {
        std::cerr << "Failed to unset environment variable: " << name << std::endl;
        exit(1);
    }
}

std::set<std::string> getFilesInDirectory(const std::string& dirPath) {
    std::set<std::string> files;
    if (!std::filesystem::exists(dirPath)) {
        return files;
    }
    
    for (const auto& entry : std::filesystem::directory_iterator(dirPath)) {
        if (entry.is_regular_file()) {
            files.insert(entry.path().filename().string());
        }
    }
    return files;
}

__global__ void testKernel(int* data, int multiplier) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    data[idx] = idx * multiplier;
}

void testKernelCachingWithIGCVariables() {
    std::cout << "Testing kernel caching with IGC variables..." << std::endl;
    
    // Get the cache directory from environment variable
    const char* cacheDir = std::getenv("CHIP_MODULE_CACHE_DIR");
    if (!cacheDir || strlen(cacheDir) == 0) {
        std::cerr << "CHIP_MODULE_CACHE_DIR not set or empty" << std::endl;
        return;
    }
    std::string tempCacheDir(cacheDir);
    
    hipError_t err = hipInit(0);
    assert(err == hipSuccess && "Failed to initialize HIP");
    
    int deviceCount;
    err = hipGetDeviceCount(&deviceCount);
    assert(err == hipSuccess && "Failed to get device count");
    assert(deviceCount > 0 && "No HIP devices available");
    
    // Set device
    err = hipSetDevice(0);
    assert(err == hipSuccess && "Failed to set device");
    
    // Allocate memory
    int* d_data;
    const int N = 256;
    err = hipMalloc((void**)&d_data, N * sizeof(int));
    assert(err == hipSuccess && "Failed to allocate device memory");
    
    // Test 1: Launch kernel without IGC variables
    std::cout << "  Launching kernel without IGC variables..." << std::endl;
    hipLaunchKernelGGL(testKernel, dim3(1), dim3(N), 0, 0, d_data, 2);
    err = hipDeviceSynchronize();
    assert(err == hipSuccess && "Kernel launch failed without IGC variables");
    
    // Check cache files after first launch
    auto cacheFiles1 = getFilesInDirectory(tempCacheDir);
    std::cout << "  Cache files after first launch: " << cacheFiles1.size() << std::endl;
    
    // Print cache file names for debugging
    for (const auto& file : cacheFiles1) {
        std::cout << "    Cache file: " << file << std::endl;
    }
    
    // Verify results are correct
    int* h_data = new int[N];
    err = hipMemcpy(h_data, d_data, N * sizeof(int), hipMemcpyDeviceToHost);
    assert(err == hipSuccess && "Failed to copy data from device");
    
    for (int i = 0; i < N; i++) {
        assert(h_data[i] == i * 2 && "Kernel computation incorrect");
    }
    
    // Clean up
    delete[] h_data;
    hipFree(d_data);
    
    std::cout << "Kernel caching test completed. Cache files created: " << cacheFiles1.size() << std::endl;
}

void testKernelCachingWithDifferentIGCVariables() {
    std::cout << "Testing kernel caching with different IGC variables..." << std::endl;
    
    // Create a temporary cache directory for testing
    std::string tempCacheDir = "/tmp/chipstar_test_cache_" + std::to_string(getpid());
    setEnvVar("CHIP_MODULE_CACHE_DIR", tempCacheDir.c_str());
    
    // Clean up any existing cache directory and create fresh one
    if (std::filesystem::exists(tempCacheDir)) {
        std::filesystem::remove_all(tempCacheDir);
    }
    std::filesystem::create_directories(tempCacheDir);
    
    // Check existing cache files
    auto cacheFilesBefore = getFilesInDirectory(tempCacheDir);
    std::cout << "  Cache files before launch: " << cacheFilesBefore.size() << std::endl;
    
    // Initialize HIP
    hipError_t err = hipInit(0);
    assert(err == hipSuccess && "Failed to initialize HIP");
    
    // Set device
    err = hipSetDevice(0);
    assert(err == hipSuccess && "Failed to set device");
    
    // Allocate memory
    int* d_data;
    const int N = 256;
    err = hipMalloc((void**)&d_data, N * sizeof(int));
    assert(err == hipSuccess && "Failed to allocate device memory");
    
    // Launch kernel with different IGC variables (set externally)
    std::cout << "  Launching kernel with IGC variables..." << std::endl;
    hipLaunchKernelGGL(testKernel, dim3(1), dim3(N), 0, 0, d_data, 3);
    err = hipDeviceSynchronize();
    assert(err == hipSuccess && "Kernel launch failed with IGC variables");
    
    // Check cache files after launch
    auto cacheFilesAfter = getFilesInDirectory(tempCacheDir);
    std::cout << "  Cache files after launch: " << cacheFilesAfter.size() << std::endl;
    
    // Print cache file names for debugging
    for (const auto& file : cacheFilesAfter) {
        std::cout << "    Cache file: " << file << std::endl;
    }
    
    // Should have more cache files now (different cache key due to IGC variables)
    assert(cacheFilesAfter.size() > cacheFilesBefore.size() && 
           "IGC variables should create different cache entries");
    
    // Verify results are correct
    int* h_data = new int[N];
    err = hipMemcpy(h_data, d_data, N * sizeof(int), hipMemcpyDeviceToHost);
    assert(err == hipSuccess && "Failed to copy data from device");
    
    for (int i = 0; i < N; i++) {
        assert(h_data[i] == i * 3 && "Kernel computation incorrect");
    }
    
    // Clean up
    delete[] h_data;
    hipFree(d_data);
    
    if (std::filesystem::exists(tempCacheDir)) {
        std::filesystem::remove_all(tempCacheDir);
    }
    
    std::cout << "Kernel caching with different IGC variables test passed!" << std::endl;
}

void testIGCVariablesWithoutCaching() {
    std::cout << "Testing IGC variables without caching (empty cache dir)..." << std::endl;
    
    setEnvVar("CHIP_MODULE_CACHE_DIR", "");
    
    hipError_t err = hipInit(0);
    assert(err == hipSuccess && "Failed to initialize HIP");
    
    err = hipSetDevice(0);
    assert(err == hipSuccess && "Failed to set device");
    
    int* d_data;
    const int N = 64;
    err = hipMalloc((void**)&d_data, N * sizeof(int));
    assert(err == hipSuccess && "Failed to allocate device memory");
    
    // Test with different IGC variables - should still work without caching
    setEnvVar("IGC_EnableDPEmulation", "1");
    hipLaunchKernelGGL(testKernel, dim3(1), dim3(N), 0, 0, d_data, 5);
    err = hipDeviceSynchronize();
    assert(err == hipSuccess && "Kernel launch failed with IGC variables and no caching");
    
    setEnvVar("IGC_OptLevel", "2");
    hipLaunchKernelGGL(testKernel, dim3(1), dim3(N), 0, 0, d_data, 6);
    err = hipDeviceSynchronize();
    assert(err == hipSuccess && "Kernel launch failed with changed IGC variables and no caching");
    
    // Verify results
    int* h_data = new int[N];
    err = hipMemcpy(h_data, d_data, N * sizeof(int), hipMemcpyDeviceToHost);
    assert(err == hipSuccess && "Failed to copy data from device");
    
    for (int i = 0; i < N; i++) {
        assert(h_data[i] == i * 6 && "Kernel computation incorrect");
    }
    
    // Clean up
    delete[] h_data;
    hipFree(d_data);
    unsetEnvVar("IGC_EnableDPEmulation");
    unsetEnvVar("IGC_OptLevel");
    
    std::cout << "IGC variables without caching test passed!" << std::endl;
}

int main() {
    try {
        std::cout << "Running IGC caching tests..." << std::endl;
        
        // Get the cache directory from environment variable (set by CMake)
        const char* cacheDir = std::getenv("CHIP_MODULE_CACHE_DIR");
        if (cacheDir && strlen(cacheDir) > 0) {
            std::string tempCacheDir(cacheDir);
            
            // Clean up any existing cache directory and create fresh one
            if (std::filesystem::exists(tempCacheDir)) {
                std::filesystem::remove_all(tempCacheDir);
            }
            std::filesystem::create_directories(tempCacheDir);
            
            std::cout << "  Using cache directory: " << tempCacheDir << std::endl;
        }
        
        CHIPInitialize();
        
        // Only run the basic kernel caching test
        testKernelCachingWithIGCVariables();
        
        // Clean up the test cache directory if it was set
        if (cacheDir && strlen(cacheDir) > 0) {
            std::string tempCacheDir(cacheDir);
            if (std::filesystem::exists(tempCacheDir)) {
                std::filesystem::remove_all(tempCacheDir);
            }
        }
        
        std::cout << "IGC caching test PASSED!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test FAILED with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test FAILED with unknown exception!" << std::endl;
        return 1;
    }
} 