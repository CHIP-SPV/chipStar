#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <level_zero/ze_api.h>
#include <algorithm> // For std::min
#include <cstdio>    // For std::remove, std::tmpnam (though not using tmpnam directly for safety)
#include <cstdlib>   // For std::system
#include <array>     // For std::array to read magic number

// Helper function to convert ze_result_t to a string for error reporting
std::string zeResultToString(ze_result_t status) {
    switch (status) {
    case ZE_RESULT_SUCCESS: return "ZE_RESULT_SUCCESS";
    case ZE_RESULT_NOT_READY: return "ZE_RESULT_NOT_READY";
    case ZE_RESULT_ERROR_DEVICE_LOST: return "ZE_RESULT_ERROR_DEVICE_LOST";
    case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY: return "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
    case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY: return "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
    case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE: return "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE";
    case ZE_RESULT_ERROR_MODULE_LINK_FAILURE: return "ZE_RESULT_ERROR_MODULE_LINK_FAILURE";
    case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS: return "ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS";
    case ZE_RESULT_ERROR_NOT_AVAILABLE: return "ZE_RESULT_ERROR_NOT_AVAILABLE";
    case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE: return "ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE";
    case ZE_RESULT_ERROR_UNINITIALIZED: return "ZE_RESULT_ERROR_UNINITIALIZED";
    case ZE_RESULT_ERROR_UNSUPPORTED_VERSION: return "ZE_RESULT_ERROR_UNSUPPORTED_VERSION";
    case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE: return "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
    case ZE_RESULT_ERROR_INVALID_ARGUMENT: return "ZE_RESULT_ERROR_INVALID_ARGUMENT";
    case ZE_RESULT_ERROR_INVALID_NULL_HANDLE: return "ZE_RESULT_ERROR_INVALID_NULL_HANDLE";
    case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE: return "ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE";
    case ZE_RESULT_ERROR_INVALID_NULL_POINTER: return "ZE_RESULT_ERROR_INVALID_NULL_POINTER";
    case ZE_RESULT_ERROR_INVALID_SIZE: return "ZE_RESULT_ERROR_INVALID_SIZE";
    case ZE_RESULT_ERROR_UNSUPPORTED_SIZE: return "ZE_RESULT_ERROR_UNSUPPORTED_SIZE";
    case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT: return "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT";
    case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT: return "ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT";
    case ZE_RESULT_ERROR_INVALID_ENUMERATION: return "ZE_RESULT_ERROR_INVALID_ENUMERATION";
    case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION: return "ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION";
    case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT: return "ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT";
    case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY: return "ZE_RESULT_ERROR_INVALID_NATIVE_BINARY";
    case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME: return "ZE_RESULT_ERROR_INVALID_GLOBAL_NAME";
    case ZE_RESULT_ERROR_INVALID_KERNEL_NAME: return "ZE_RESULT_ERROR_INVALID_KERNEL_NAME";
    case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME: return "ZE_RESULT_ERROR_INVALID_FUNCTION_NAME";
    case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION: return "ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION";
    case ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION: return "ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION";
    case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX: return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX";
    case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE: return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE";
    case ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE: return "ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE";
    case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED: return "ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED";
    case ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE: return "ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE";
    case ZE_RESULT_ERROR_OVERLAPPING_REGIONS: return "ZE_RESULT_ERROR_OVERLAPPING_REGIONS";
    case ZE_RESULT_ERROR_UNKNOWN: return "ZE_RESULT_ERROR_UNKNOWN";
    default: return "Unknown Level Zero Error Code";
    }
}

// Helper to check L0 status, print error, and exit if failed
void checkZeResult(ze_result_t status, const std::string& callName) {
    if (status != ZE_RESULT_SUCCESS) {
        std::cerr << "Error: Level Zero call \'" << callName << "\' failed with "
                  << zeResultToString(status) << " (0x" << std::hex << status << std::dec << ")"
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Dumps build log into the error log stream. The 'Log' value must be a valid handle.
// This function will destroy the log handle.
static void dumpZeBuildLog(ze_module_build_log_handle_t buildLog) {
    if (!buildLog) return;

    size_t logSize = 0;
    ze_result_t status = zeModuleBuildLogGetString(buildLog, &logSize, nullptr);
    if (status == ZE_RESULT_SUCCESS && logSize > 0) {
        std::vector<char> logBuffer(logSize);
        status = zeModuleBuildLogGetString(buildLog, &logSize, logBuffer.data());
        if (status == ZE_RESULT_SUCCESS) {
            std::cout << "--- Build Log ---" << std::endl;
            std::cout << std::string(logBuffer.data(), logSize) << std::endl;
            std::cout << "-----------------" << std::endl;
        } else {
            std::cerr << "Warning: zeModuleBuildLogGetString failed to get log content: " << zeResultToString(status) << std::endl;
        }
    } else if (status != ZE_RESULT_SUCCESS) {
         std::cerr << "Warning: zeModuleBuildLogGetString failed to get log size: " << zeResultToString(status) << std::endl;
    }
    
    // Always attempt to destroy the log handle
    status = zeModuleBuildLogDestroy(buildLog);
    if (status != ZE_RESULT_SUCCESS) {
        std::cerr << "Warning: zeModuleBuildLogDestroy failed: " << zeResultToString(status) << std::endl;
    }
}

// Function to read a file into a vector of bytes
std::vector<uint8_t> readFile(const std::string& filePath, bool& success) {
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << filePath << std::endl;
        success = false;
        return {};
    }
    std::streamsize fileSize = file.tellg();
    if (fileSize == 0) {
         std::cerr << "Warning: File is empty: " << filePath << std::endl;
         // allow empty file for now, L0 might handle it or fail later
    }
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(fileSize);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), fileSize)) {
        std::cerr << "Error: Could not read file: " << filePath << std::endl;
        success = false;
        return {};
    }
    file.close();
    success = true;
    return buffer;
}


int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <spirv_file_path>" << std::endl;
        return EXIT_FAILURE;
    }
    std::string spvFilePath = argv[1];
    std::string currentSpvPath = spvFilePath; // Path to the SPV data we will use
    bool deleteTemporarySpv = false;
    std::string temporarySpvPath = "";

    // 1. Initialize Level Zero
    checkZeResult(zeInit(0), "zeInit");
    std::cout << "Level Zero initialized." << std::endl;

    // 2. Discover Driver
    uint32_t driverCount = 0;
    checkZeResult(zeDriverGet(&driverCount, nullptr), "zeDriverGet (count)");
    if (driverCount == 0) {
        std::cerr << "Error: No Level Zero drivers found." << std::endl;
        return EXIT_FAILURE;
    }
    std::vector<ze_driver_handle_t> drivers(driverCount);
    checkZeResult(zeDriverGet(&driverCount, drivers.data()), "zeDriverGet (handles)");
    ze_driver_handle_t driverHandle = drivers[0]; // Using the first driver
    std::cout << "Using driver 0." << std::endl;

    // 3. Discover Device
    uint32_t deviceCount = 0;
    checkZeResult(zeDeviceGet(driverHandle, &deviceCount, nullptr), "zeDeviceGet (count)");
    if (deviceCount == 0) {
        std::cerr << "Error: No Level Zero devices found for the selected driver." << std::endl;
        return EXIT_FAILURE;
    }
    std::vector<ze_device_handle_t> devices(deviceCount);
    checkZeResult(zeDeviceGet(driverHandle, &deviceCount, devices.data()), "zeDeviceGet (handles)");
    
    ze_device_handle_t deviceHandle = nullptr;
    // Prefer GPU device
    for (uint32_t i = 0; i < deviceCount; ++i) {
        ze_device_properties_t deviceProperties{};
        deviceProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        // deviceProperties.pNext = nullptr; // Not strictly necessary as it's POD and zero-initialized
        checkZeResult(zeDeviceGetProperties(devices[i], &deviceProperties), "zeDeviceGetProperties for device " + std::to_string(i));
        if (deviceProperties.type == ZE_DEVICE_TYPE_GPU) {
            deviceHandle = devices[i];
            std::cout << "Selected GPU: " << deviceProperties.name << " (Device " << i << ")" << std::endl;
            break;
        }
    }

    if (!deviceHandle && deviceCount > 0) { // Fallback to the first device if no GPU found
        deviceHandle = devices[0];
        ze_device_properties_t deviceProperties{};
        deviceProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        checkZeResult(zeDeviceGetProperties(deviceHandle, &deviceProperties), "zeDeviceGetProperties for fallback device 0");
        std::cout << "No GPU found. Selected first available device: " << deviceProperties.name << " (Device 0)" << std::endl;
    }

    if (!deviceHandle) { // Should not happen if deviceCount > 0
         std::cerr << "Error: Failed to select a device." << std::endl;
         return EXIT_FAILURE;
    }

    // 4. Create Context
    ze_context_desc_t contextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    ze_context_handle_t context = nullptr;
    checkZeResult(zeContextCreate(driverHandle, &contextDesc, &context), "zeContextCreate");
    std::cout << "Context created." << std::endl;

    // 5. Read SPIR-V file (initial attempt, check magic number)
    std::vector<uint8_t> spirvBuffer;
    bool fileReadSuccess = false;

    // Check magic number
    const uint32_t SPV_MAGIC_NUMBER = 0x07230203;
    std::ifstream initialFileCheck(spvFilePath, std::ios::binary);
    bool isBinarySpv = false;
    if (initialFileCheck.is_open()) {
        std::array<uint8_t, 4> magicBuffer;
        initialFileCheck.read(reinterpret_cast<char*>(magicBuffer.data()), 4);
        if (initialFileCheck.gcount() == 4) {
            uint32_t magicInFile = 0;
            // Assuming little-endian SPIR-V magic number
            magicInFile |= static_cast<uint32_t>(magicBuffer[3]) << 24;
            magicInFile |= static_cast<uint32_t>(magicBuffer[2]) << 16;
            magicInFile |= static_cast<uint32_t>(magicBuffer[1]) << 8;
            magicInFile |= static_cast<uint32_t>(magicBuffer[0]);
            if (magicInFile == SPV_MAGIC_NUMBER) {
                isBinarySpv = true;
            }
        }
        initialFileCheck.close();
    } else {
        std::cerr << "Error: Could not open input file for magic number check: " << spvFilePath << std::endl;
        zeContextDestroy(context);
        return EXIT_FAILURE;
    }

    if (isBinarySpv) {
        std::cout << "Input file appears to be a SPIR-V binary (magic number matches)." << std::endl;
        spirvBuffer = readFile(spvFilePath, fileReadSuccess);
        if (!fileReadSuccess) {
            zeContextDestroy(context);
            return EXIT_FAILURE;
        }
    } else {
        std::cout << "Input file does not have SPIR-V binary magic number. Attempting to convert with spirv-as..." << std::endl;
        // Create a temporary file name. Note: simple, not C++17 <filesystem> robust.
        // Using fixed name for simplicity in example, consider more robust temp naming.
        temporarySpvPath = spvFilePath + ".temp.spv"; 
        std::string command = "spirv-as \"" + spvFilePath + "\" -o \"" + temporarySpvPath + "\"";
        std::cout << "Executing: " << command << std::endl;
        
        int spirvAsRet = std::system(command.c_str());
        if (spirvAsRet == 0) {
            std::cout << "spirv-as conversion successful. Reading temporary binary." << std::endl;
            spirvBuffer = readFile(temporarySpvPath, fileReadSuccess);
            if (!fileReadSuccess) {
                std::remove(temporarySpvPath.c_str()); // Clean up
                zeContextDestroy(context);
                return EXIT_FAILURE;
            }
            currentSpvPath = temporarySpvPath;
            deleteTemporarySpv = true;
        } else {
            std::cerr << "Error: spirv-as command failed with return code " << spirvAsRet << "." << std::endl;
            std::cerr << "Please ensure spirv-as is installed and in your PATH, and the input is valid SPIR-V assembly." << std::endl;
            // No need to remove temporarySpvPath if spirv-as failed to create it or errored.
            zeContextDestroy(context);
            return EXIT_FAILURE;
        }
    }
    
    std::cout << "SPIR-V data from '" << currentSpvPath << "' loaded (" << spirvBuffer.size() << " bytes)." << std::endl;

    // Print SPIR-V entry points using spirv-dis
    std::cout << "\n--- SPIR-V Declared Entry Points (from spirv-dis) ---" << std::endl;
    std::string spirvDisCommand = "spirv-dis \"" + currentSpvPath + "\" | grep OpEntryPoint | cat";
    std::cout << "Executing: " << spirvDisCommand << std::endl;
    int disRet = std::system(spirvDisCommand.c_str());
    if (disRet != 0) {
        std::cerr << "Warning: spirv-dis command potentially failed or found no OpEntryPoints. Return code: " << disRet << std::endl;
        std::cerr << "This might happen if spirv-dis is not in PATH, grep is not available, or the .spv file is invalid/empty." << std::endl;
    }
    std::cout << "----------------------------------------------------\n" << std::endl;

    // 6. Create Module
    ze_module_desc_t moduleDesc = {};
    moduleDesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    moduleDesc.pNext = nullptr;
    moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    moduleDesc.inputSize = spirvBuffer.size();
    moduleDesc.pInputModule = spirvBuffer.data();
    moduleDesc.pBuildFlags = ""; // No specific build flags
    moduleDesc.pConstants = nullptr;

    ze_module_handle_t moduleHandle = nullptr;
    ze_module_build_log_handle_t buildLog = nullptr;
    
    std::cout << "Attempting to create Level Zero module..." << std::endl;
    ze_result_t moduleCreateStatus = zeModuleCreate(context, deviceHandle, &moduleDesc, &moduleHandle, &buildLog);
    
    // IMPORTANT DIAGNOSTIC LINE:
    std::cout << "zeModuleCreate returned: " << zeResultToString(moduleCreateStatus) 
              << ". Build log handle is " << (buildLog ? "non-null" : "null") << "." << std::endl;

    dumpZeBuildLog(buildLog); // Dump log regardless of module creation status, then destroy it.
                             // buildLog is set to nullptr by dumpZeBuildLog if it was valid.
    
    if (moduleCreateStatus != ZE_RESULT_SUCCESS) {
         std::cerr << "Failed to create Level Zero module from '" << currentSpvPath 
                   << "' with error: " << zeResultToString(moduleCreateStatus) << std::endl;
         if (deleteTemporarySpv) {
            std::remove(temporarySpvPath.c_str());
         }
         zeContextDestroy(context);
         return EXIT_FAILURE;
    }
    std::cout << "Module created successfully." << std::endl;

    // 7. Get Kernel Names
    uint32_t kernelCount = 0;
    checkZeResult(zeModuleGetKernelNames(moduleHandle, &kernelCount, nullptr), "zeModuleGetKernelNames (count)");

    if (kernelCount > 0) {
        std::vector<const char*> kernelNamesTemp(kernelCount); // Stores pointers
        checkZeResult(zeModuleGetKernelNames(moduleHandle, &kernelCount, kernelNamesTemp.data()), "zeModuleGetKernelNames (names)");
        
        std::cout << "\nAvailable kernels (" << kernelCount << "):" << std::endl;
        for (uint32_t i = 0; i < kernelCount; ++i) {
            if (kernelNamesTemp[i] != nullptr) {
                std::cout << " - " << kernelNamesTemp[i] << std::endl;
            } else {
                std::cout << " - <Warning: Received null kernel name pointer at index " << i << ">" << std::endl;
            }
        }
    } else {
        std::cout << "No kernels found in the module." << std::endl;
    }

    // 8. Cleanup
    if (deleteTemporarySpv) {
        std::cout << "Deleting temporary file: " << temporarySpvPath << std::endl;
        std::remove(temporarySpvPath.c_str());
    }
    if (moduleHandle) {
        checkZeResult(zeModuleDestroy(moduleHandle), "zeModuleDestroy");
        std::cout << "Module destroyed." << std::endl;
    }
    if (context) {
        checkZeResult(zeContextDestroy(context), "zeContextDestroy");
        std::cout << "Context destroyed." << std::endl;
    }
    
    std::cout << "Execution finished." << std::endl;
    return EXIT_SUCCESS;
}

/*
To compile this file (e.g., on Linux with Intel oneAPI Base Toolkit installed):
g++ -std=c++17 -g l0_compile_list_kernels.cpp -o l0_compile_list_kernels -I${ONEAPI_ROOT}/level-zero/include -L${ONEAPI_ROOT}/level-zero/lib -lze_loader

Then run it with a SPIR-V file (binary or assembly):
./l0_compile_list_kernels my_kernel.spv
./l0_compile_list_kernels my_kernel.spv.asm 

Note: This program expects the input file to be a SPIR-V *binary*. 
If you have SPIR-V assembly text (e.g., a .spv.txt file), you need to compile it 
to binary first using a tool like `spirv-as` from the SPIRV-Tools suite.
For example: `spirv-as my_kernel.spv.txt -o my_kernel.spv`
*/ 