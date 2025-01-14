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
// We store each work-item's data in a local memory array, then read
// from the "offset" lane within the same sub-group/wave.
static const char* kernelSource = R"CLC(
__kernel void emulate_bpermute(__global const int* inData,
                               __global int* outData,
                               int offset)
{
    // Local memory for up to 256 threads in a work-group:
    __local int lmem[256];

    int lid = get_local_id(0);
    int groupSize = get_local_size(0);

    // Write each thread's data to local memory
    lmem[lid] = inData[get_global_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Sub-group/wave size. For demonstration, we just assume groupSize is our wave size.
    int laneId = lid;  
    int targetLane = (laneId + offset) & (groupSize - 1);

    // Compute the index in local memory
    // If you had multiple sub-groups, you'd do subGroupId * subGroupSize + targetLane.
    // We'll assume just one sub-group or groupSize == wavefront.
    int finalIdx = targetLane;

    // Read from local memory
    int val = lmem[finalIdx];

    // Write result to global
    outData[get_global_id(0)] = val;
}
)CLC";

// HIP kernel using __builtin_amdgcn_ds_bpermute
__global__ void hip_bpermute(const int* inData, int* outData, int offset) {
    int tid = threadIdx.x;
    int lane = tid & 63; // Assuming wave size of 64
    int target_lane = (lane + offset) & 63;
    
    // Convert input data to the required format for ds_bpermute
    int src_data = inData[tid];
    int src_lane = target_lane << 2; // Multiply by 4 as ds_bpermute expects byte offset
    
    // Call the builtin
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
        inData[i] = 1000000000 + i; // e.g., 1000000000 + i
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

    // Pick the first platform and device (adjust to your preference)
    cl::Platform platform = platforms[0];
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);
    if (devices.empty()) {
        std::cerr << "No OpenCL devices found.\n";
        return 1;
    }
    cl::Device device = devices[0];

    // Create a context and command queue
    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    // Build the program from source
    cl::Program::Sources sources;
    sources.push_back({kernelSource, strlen(kernelSource)});
    cl::Program program(context, sources);

    try {
        program.build({device});
    } catch (...) {
        // Print build errors
        std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        std::cerr << "Error building: " << buildLog << "\n";
        return 1;
    }

    // Create the kernel
    cl::Kernel kernel(program, "emulate_bpermute");

    // ---------------------------------------------------------
    // Create device buffers
    // ---------------------------------------------------------
    cl::Buffer bufIn(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(int) * inData.size(), inData.data());

    cl::Buffer bufOut(context, CL_MEM_WRITE_ONLY,
                      sizeof(int) * outDataOpenCL.size());

    // ---------------------------------------------------------
    // Set kernel arguments
    // ---------------------------------------------------------
    int offset = 3; // Example offset
    kernel.setArg(0, bufIn); 
    kernel.setArg(1, bufOut);
    kernel.setArg(2, offset);

    // ---------------------------------------------------------
    // Enqueue kernel
    // ---------------------------------------------------------
    queue.enqueueNDRangeKernel(kernel,
                               cl::NullRange,
                               cl::NDRange(globalSize),
                               cl::NDRange(localSize));
    queue.finish();

    // ---------------------------------------------------------
    // Read results back
    // ---------------------------------------------------------
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
                       d_inData, d_outData, offset);
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