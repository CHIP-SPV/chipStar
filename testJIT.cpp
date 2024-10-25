#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <level_zero/ze_api.h>
#include <CL/cl.h>
#include <cstdlib>  // for system()
#include <sstream>  // for stringstream

// Function to read SPIR-V file
std::vector<char> readSPIRVFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open SPIR-V file");
    }
    
    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    
    return buffer;
}

// Function to initialize Level Zero
ze_context_handle_t initializeLevelZero() {
    ze_result_t result;
    
    // Initialize the driver
    result = zeInit(0);
    if (result != ZE_RESULT_SUCCESS) {
        throw std::runtime_error("Failed to initialize Level Zero driver");
    }
    
    // Get the device
    uint32_t driverCount = 1;
    ze_driver_handle_t driver;
    result = zeDriverGet(&driverCount, &driver);
    if (result != ZE_RESULT_SUCCESS) {
        throw std::runtime_error("Failed to get Level Zero driver");
    }
    
    uint32_t deviceCount = 1;
    ze_device_handle_t device;
    result = zeDeviceGet(driver, &deviceCount, &device);
    if (result != ZE_RESULT_SUCCESS) {
        throw std::runtime_error("Failed to get Level Zero device");
    }
    
    // Create context
    ze_context_desc_t contextDesc = {};
    contextDesc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    ze_context_handle_t context;
    result = zeContextCreate(driver, &contextDesc, &context);
    if (result != ZE_RESULT_SUCCESS) {
        throw std::runtime_error("Failed to create Level Zero context");
    }
    
    return context;
}

// Function to initialize OpenCL
cl_context initializeOpenCL() {
    cl_int err;
    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to get OpenCL platforms");
    }
    
    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to get OpenCL platform IDs");
    }
    
    cl_device_id device;
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to get OpenCL device");
    }
    
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL context");
    }
    
    return context;
}

// Function to create Level Zero kernel and dump machine code
void createAndDumpLevelZeroKernel(ze_context_handle_t context, const std::vector<char>& spirvData, const std::string& buildFlags) {
    ze_result_t result;
    
    // Create module
    ze_module_desc_t moduleDesc = {};
    moduleDesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    moduleDesc.inputSize = spirvData.size();
    moduleDesc.pInputModule = reinterpret_cast<const uint8_t*>(spirvData.data());
    moduleDesc.pBuildFlags = buildFlags.empty() ? nullptr : buildFlags.c_str();

    // Get the device properly
    uint32_t driverCount = 1;
    ze_driver_handle_t driver;
    result = zeDriverGet(&driverCount, &driver);
    if (result != ZE_RESULT_SUCCESS) {
        throw std::runtime_error("Failed to get Level Zero driver");
    }
    
    uint32_t deviceCount = 1;
    ze_device_handle_t device;
    result = zeDeviceGet(driver, &deviceCount, &device);  // Use driver handle here
    if (result != ZE_RESULT_SUCCESS) {
        throw std::runtime_error("Failed to get Level Zero device");
    }
    
    ze_module_handle_t module;
    result = zeModuleCreate(context, device, &moduleDesc, &module, nullptr);
    if (result != ZE_RESULT_SUCCESS) {
        throw std::runtime_error("Failed to create Level Zero module");
    }
    
    // Dump machine code
    size_t binarySize;
    result = zeModuleGetNativeBinary(module, &binarySize, nullptr);
    if (result != ZE_RESULT_SUCCESS) {
        throw std::runtime_error("Failed to get Level Zero module binary size");
    }

    std::vector<uint8_t> binaryData(binarySize);
    result = zeModuleGetNativeBinary(module, &binarySize, binaryData.data());
    if (result != ZE_RESULT_SUCCESS) {
        throw std::runtime_error("Failed to get Level Zero module binary");
    }

    std::ofstream outFile("level_zero_kernel.bin", std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(binaryData.data()), binarySize);
    outFile.close();
    
    std::cout << "Level Zero kernel machine code dumped to 'level_zero_kernel.bin'" << std::endl;
    
    // Clean up
    zeModuleDestroy(module);
}

// Function to create OpenCL kernel and dump machine code
void createAndDumpOpenCLKernel(cl_context context, const std::vector<char>& spirvData, const std::string& buildFlags) {
    cl_int err;
    
    // Create program
    cl_program program = clCreateProgramWithIL(context, spirvData.data(), spirvData.size(), &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL program");
    }
    
    // Get device from context
    size_t deviceSize;
    err = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, nullptr, &deviceSize);
    if (err != CL_SUCCESS) {
        clReleaseProgram(program);
        throw std::runtime_error("Failed to get context device size");
    }

    std::vector<cl_device_id> devices(deviceSize / sizeof(cl_device_id));
    err = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceSize, devices.data(), nullptr);
    if (err != CL_SUCCESS) {
        clReleaseProgram(program);
        throw std::runtime_error("Failed to get context devices");
    }
    
    // Build program with flags
    err = clBuildProgram(program, 1, devices.data(), 
                        buildFlags.empty() ? nullptr : buildFlags.c_str(), 
                        nullptr, nullptr);
    
    // Always get build log regardless of build success
    size_t logSize;
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
    if (logSize > 1) {  // Size includes null terminator
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cout << "Build log:\n" << log.data() << std::endl;
    }

    if (err != CL_SUCCESS) {
        clReleaseProgram(program);
        throw std::runtime_error("Failed to build OpenCL program");
    }
    
    // Dump machine code
    size_t binarySize;
    err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binarySize, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseProgram(program);
        throw std::runtime_error("Failed to get OpenCL program binary size");
    }
    
    std::vector<unsigned char*> binaries(1);
    binaries[0] = new unsigned char[binarySize];
    
    err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*), binaries.data(), nullptr);
    if (err != CL_SUCCESS) {
        delete[] binaries[0];
        clReleaseProgram(program);
        throw std::runtime_error("Failed to get OpenCL program binary");
    }
    
    std::ofstream outFile("opencl_kernel.bin", std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(binaries[0]), binarySize);
    outFile.close();
    
    delete[] binaries[0];
    std::cout << "OpenCL kernel machine code dumped to 'opencl_kernel.bin'" << std::endl;
    
    // Clean up
    clReleaseProgram(program);
}

// Add this new function
void disassembleWithOcloc(const std::string& inputBinaryPath, const std::string& outputPrefix) {
    std::stringstream cmd;
    cmd << "ocloc disasm -file " << inputBinaryPath << " -dump " << outputPrefix;
    
    int result = system(cmd.str().c_str());
    if (result != 0) {
        std::cerr << "Error: Failed to run ocloc disassembly. Command: " << cmd.str() << std::endl;
        throw std::runtime_error("Failed to run ocloc disassembly");
    }
    
    std::cout << "Disassembly dumped to '" << outputPrefix << "_disasm.txt'" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <SPIR-V file> [JIT flags]" << std::endl;
        return 1;
    }
    
    try {
        // Read SPIR-V file
        std::vector<char> spirvData = readSPIRVFile(argv[1]);
        
        // Get JIT flags (empty if not provided)
        std::string jitFlags = (argc == 3) ? argv[2] : "";
        
        // Initialize Level Zero
        ze_context_handle_t zeContext = initializeLevelZero();
        
        // Initialize OpenCL
        cl_context clContext = initializeOpenCL();
        
        // Create and dump Level Zero kernel
        createAndDumpLevelZeroKernel(zeContext, spirvData, jitFlags);
        disassembleWithOcloc("level_zero_kernel.bin", "level_zero_disasm");
        
        // Create and dump OpenCL kernel
        createAndDumpOpenCLKernel(clContext, spirvData, jitFlags);
        disassembleWithOcloc("opencl_kernel.bin", "opencl_disasm");
        
        // Clean up
        zeContextDestroy(zeContext);
        clReleaseContext(clContext);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
