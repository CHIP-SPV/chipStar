#include <CL/cl2.hpp>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

// Helper function for HIP error checking
#define HIP_CHECK(cmd) \
    do { \
        hipError_t error = cmd; \
        if (error != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            exit(1); \
        } \
    } while(0)

// The OpenCL C kernel source.
static const char* kernelSource = R"CLC(
// Emulate __builtin_amdgcn_ds_bpermute
int __builtin_amdgcn_ds_bpermute_emulated(__local int* lmem, int byte_offset, int src_data) {
    // Write source data to local memory at this thread's position
    int lid = get_local_id(0);
    lmem[lid] = src_data;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Convert byte offset to lane index (divide by 4 since we're dealing with ints)
    int target_lane = (byte_offset >> 2) & 63;  // 63 is wavefront size - 1
    
    // Read from the target lane
    return lmem[target_lane];
}

__kernel void ocl_permute(__global const int* inData,
                         __global int* outData)
{
    // Local memory for the wavefront (64 threads)
    __local int lmem[64];

    int lid = get_local_id(0);
    
    // Load input data
    int src_data = inData[get_global_id(0)];
    
    // Calculate byte offset (same as in HIP version)
    int lane = lid & 63;
    int byte_offset = lane * 4;  // Each int is 4 bytes
    
    // Call our emulated version
    int result = __builtin_amdgcn_ds_bpermute_emulated(lmem, byte_offset, src_data);
    
    // Write result
    outData[get_global_id(0)] = result;
}
)CLC";

// HIP kernel using __builtin_amdgcn_ds_bpermute
__global__ void hip_bpermute(const int* inData, int* outData) {
    int tid = threadIdx.x;
    int lane = tid & 63;  // Get lane ID within wavefront
    
    // Load the value this thread will share
    int src_data = inData[tid];
    
    // The byte offset is lane * 4 (each int is 4 bytes)
    int src_lane = lane * 4;
    
    // Call the builtin with byte offset and source data
    int result = __builtin_amdgcn_ds_bpermute(src_lane, src_data);
    outData[tid] = result;
}

int main() {
    // Choose how many threads (global size). For simplicity, use 64.
    const size_t globalSize = 64;
    // We'll also choose the local/work-group size = 64 (one group).
    const size_t localSize  = 64;

    // Prepare input data
    std::vector<int> inData(globalSize);
    for (int i = 0; i < static_cast<int>(globalSize); ++i) {
        // Fill with some pattern
        inData[i] = 1000000000 + i;
    }

    // Arrays for results
    std::vector<int> outDataOpenCL(globalSize, 0);
    std::vector<int> outDataHIP(globalSize, 0);

    // ---------------------------------------------------------
    // OpenCL Implementation
    // ---------------------------------------------------------
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found.\n";
        return 1;
    }

    cl::Platform platform = platforms[0];
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);
    if (devices.empty()) {
        std::cerr << "No OpenCL devices found.\n";
        return 1;
    }
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    cl::Program::Sources sources;
    sources.push_back({kernelSource, strlen(kernelSource)});
    cl::Program program(context, sources);

    try {
        program.build({device});
    } catch (...) {
        std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        std::cerr << "Error building: " << buildLog << "\n";
        return 1;
    }

    cl::Kernel kernel(program, "ocl_permute");

    cl::Buffer bufIn(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(int) * inData.size(), inData.data());

    cl::Buffer bufOut(context, CL_MEM_WRITE_ONLY,
                      sizeof(int) * outDataOpenCL.size());

    kernel.setArg(0, bufIn); 
    kernel.setArg(1, bufOut);

    queue.enqueueNDRangeKernel(kernel,
                               cl::NullRange,
                               cl::NDRange(globalSize),
                               cl::NDRange(localSize));
    queue.finish();

    queue.enqueueReadBuffer(bufOut, CL_TRUE, 0,
                            sizeof(int) * outDataOpenCL.size(),
                            outDataOpenCL.data());

    // ---------------------------------------------------------
    // HIP Implementation
    // ---------------------------------------------------------
    int *d_inData, *d_outData;
    HIP_CHECK(hipMalloc(&d_inData, globalSize * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_outData, globalSize * sizeof(int)));
    
    HIP_CHECK(hipMemcpy(d_inData, inData.data(), globalSize * sizeof(int), hipMemcpyHostToDevice));
    
    hipLaunchKernelGGL(hip_bpermute, 
                       dim3(1), 
                       dim3(globalSize), 
                       0, 0,
                       d_inData, d_outData);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    
    HIP_CHECK(hipMemcpy(outDataHIP.data(), d_outData, globalSize * sizeof(int), hipMemcpyDeviceToHost));
    
    HIP_CHECK(hipFree(d_inData));
    HIP_CHECK(hipFree(d_outData));

    // ---------------------------------------------------------
    // Compare and print results
    // ---------------------------------------------------------
    std::cout << "Results Comparison:\n\n";
    bool mismatch = false;
    for (int i = 0; i < static_cast<int>(globalSize); ++i) {
        std::cout << "Thread " << i << ": OpenCL=" << outDataOpenCL[i] 
                 << " HIP=" << outDataHIP[i];
        if (outDataOpenCL[i] != outDataHIP[i]) {
            std::cout << " [MISMATCH]";
            mismatch = true;
        }
        std::cout << "\n";
    }
    
    if (mismatch) {
        std::cout << "\nWARNING: Mismatches detected between OpenCL and HIP implementations!\n";
    } else {
        std::cout << "\nSUCCESS: Both implementations produced identical results!\n";
    }

    return 0;
}