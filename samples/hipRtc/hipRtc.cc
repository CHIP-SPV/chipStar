#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <stdio.h>
#include <string>
#include <vector>

// Helper function to check hipRTC errors
void checkRtcError(hiprtcResult result, const char* file, int line) {
    if (result != HIPRTC_SUCCESS) {
        printf("hipRTC error: %s at %s:%d\n", hiprtcGetErrorString(result), file, line);
        exit(1);
    }
}
#define CHECK_RTC(x) checkRtcError(x, __FILE__, __LINE__)

// Helper function to check HIP errors
void checkHipError(hipError_t result, const char* file, int line) {
    if (result != hipSuccess) {
        printf("HIP error: %s at %s:%d\n", hipGetErrorString(result), file, line);
        exit(1);
    }
}
#define CHECK_HIP(x) checkHipError(x, __FILE__, __LINE__)

int main() {
    // Kernel source code as a string
    const char* kernelSource = R"(
        __global__ void vectorAdd(float* a, float* b, float* c, int n) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
    )";

    // Create a program
    hiprtcProgram prog;
    CHECK_RTC(hiprtcCreateProgram(&prog,    // prog
                                 kernelSource,    // buffer
                                 "vector_add.cu", // name
                                 0,               // numHeaders
                                 NULL,            // headers
                                 NULL));          // includeNames

    // Add name expression before compilation
    CHECK_RTC(hiprtcAddNameExpression(prog, "vectorAdd"));

    // Compile the program
    hiprtcResult compileResult = hiprtcCompileProgram(prog,  // prog
                                                     0,      // numOptions
                                                     NULL);  // options

    // Get compilation log
    size_t logSize;
    CHECK_RTC(hiprtcGetProgramLogSize(prog, &logSize));
    if (logSize > 1) {
        std::string log(logSize, '\0');
        CHECK_RTC(hiprtcGetProgramLog(prog, &log[0]));
        printf("Compilation log: %s\n", log.c_str());
    }

    if (compileResult != HIPRTC_SUCCESS) {
        printf("Compilation failed\n");
        return 1;
    }

    // Get the lowered name of the kernel
    const char* loweredName;
    CHECK_RTC(hiprtcGetLoweredName(prog,
                                  "vectorAdd",
                                  &loweredName));

    // Get code object
    size_t codeSize;
    CHECK_RTC(hiprtcGetCodeSize(prog, &codeSize));
    std::vector<char> code(codeSize);
    CHECK_RTC(hiprtcGetCode(prog, code.data()));

    // Load the module and get the kernel function
    hipModule_t module;
    hipFunction_t kernel;
    CHECK_HIP(hipModuleLoadData(&module, code.data()));
    CHECK_HIP(hipModuleGetFunction(&kernel, module, loweredName));

    // Prepare data
    const int N = 1000;
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CHECK_HIP(hipMalloc((void**)&d_a, size));
    CHECK_HIP(hipMalloc((void**)&d_b, size));
    CHECK_HIP(hipMalloc((void**)&d_c, size));

    // Copy data to device
    CHECK_HIP(hipMemcpy(d_a, h_a, size, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_b, h_b, size, hipMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // First kernel launch (existing code with config array)
    struct {
        float* a;
        float* b;
        float* c;
        int n;
    } args = {d_a, d_b, d_c, N};

    size_t size_args = sizeof(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                     HIP_LAUNCH_PARAM_BUFFER_SIZE, &size_args,
                     HIP_LAUNCH_PARAM_END};

    CHECK_HIP(hipModuleLaunchKernel(kernel,
                                   blocksPerGrid, 1, 1,    // grid dims
                                   threadsPerBlock, 1, 1,  // block dims
                                   0,                      // shared mem
                                   nullptr,                // stream
                                   nullptr,                // kernel params
                                   config));               // extra params

    // Second kernel launch using direct parameter passing
    int N_value = N;  // Create a local variable to take its address
    void* kernelArgs[] = {&d_a, &d_b, &d_c, &N_value};
    CHECK_HIP(hipModuleLaunchKernel(kernel,
                                   blocksPerGrid, 1, 1,    // grid dims
                                   threadsPerBlock, 1, 1,  // block dims
                                   0,                      // shared mem
                                   nullptr,                // stream
                                   kernelArgs,             // kernel params
                                   nullptr));              // extra params is nullptr when using direct params

    // Copy result back to host
    CHECK_HIP(hipMemcpy(h_c, d_c, size, hipMemcpyDeviceToHost));

    // Verify results
    bool passed = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Verification failed at index %d: %f != %f + %f\n",
                   i, h_c[i], h_a[i], h_b[i]);
            passed = false;
            break;
        }
    }

    if (passed) {
        printf("PASSED!\n");
    }

    // Cleanup
    CHECK_HIP(hipFree(d_a));
    CHECK_HIP(hipFree(d_b));
    CHECK_HIP(hipFree(d_c));
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    CHECK_RTC(hiprtcDestroyProgram(&prog));
    CHECK_HIP(hipModuleUnload(module));

    return 0;
}
