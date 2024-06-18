#include <level_zero/ze_api.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <unistd.h>

int main() {
    ze_result_t result = zeInit(0);
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "zeInit failed with error code: " << std::hex << "0x" << result << std::endl;
        return 1;
    }

    uint32_t driverCount = 0;
    result = zeDriverGet(&driverCount, nullptr);
    if (result != ZE_RESULT_SUCCESS || driverCount == 0) {
        std::cerr << "zeDriverGet failed with error code: " << std::hex << "0x" << result << std::endl;
        return 1;
    }

    std::vector<ze_driver_handle_t> drivers(driverCount);
    result = zeDriverGet(&driverCount, drivers.data());
    ze_driver_handle_t driverHandle = drivers[0];

    uint32_t deviceCount = 0;
    result = zeDeviceGet(driverHandle, &deviceCount, nullptr);
    if (result != ZE_RESULT_SUCCESS || deviceCount == 0) {
        std::cerr << "zeDeviceGet failed with error code: " << std::hex << "0x" << result << std::endl;
        return 1;
    }

    std::vector<ze_device_handle_t> devices(deviceCount);
    result = zeDeviceGet(driverHandle, &deviceCount, devices.data());
    ze_device_handle_t deviceHandle = devices[0];

    ze_context_handle_t context;
    ze_context_desc_t contextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    result = zeContextCreate(driverHandle, &contextDesc, &context);
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "zeContextCreate failed with error code: " << std::hex << "0x" << result << std::endl;
        return 1;
    }

    // Load SPIR-V binary
    char executablePath[1024];
    ssize_t count = readlink("/proc/self/exe", executablePath, sizeof(executablePath) - 1);
    if (count == -1) {
        std::cerr << "Failed to get executable path" << std::endl;
        return 1;
    }
    executablePath[count] = '\0';
    std::string directoryPath = std::string(executablePath).substr(0, std::string(executablePath).find_last_of('/'));
    std::ifstream file(directoryPath + "/inputs/firstTouch.spv", std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << directoryPath + "/inputs/firstTouch.spv" << std::endl;
        return 1;
    }

    size_t fileSize = file.tellg();
    std::vector<uint8_t> spirvBinary(fileSize);
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(spirvBinary.data()), fileSize);
    file.close();

    ze_module_handle_t module;
    ze_module_desc_t moduleDesc = {ZE_STRUCTURE_TYPE_MODULE_DESC, nullptr, ZE_MODULE_FORMAT_IL_SPIRV, spirvBinary.size(), spirvBinary.data(), nullptr, nullptr};
    ze_module_build_log_handle_t buildLog;
    result = zeModuleCreate(context, deviceHandle, &moduleDesc, &module, &buildLog);
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "zeModuleCreate failed with error code: " << std::hex << "0x" << result << std::endl;
        return 1;
    }

    ze_kernel_handle_t kernel;
    ze_kernel_desc_t kernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, ZE_KERNEL_FLAG_FORCE_RESIDENCY, "_Z6setOne4Data"};
    result = zeKernelCreate(module, &kernelDesc, &kernel);
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "zeKernelCreate failed with error code: " << std::hex << "0x" << result << std::endl;
        return 1;
    }

    // Set indirect access flags
    result = zeKernelSetIndirectAccess(kernel, ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST | ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE);
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "zeKernelSetIndirectAccess failed with error code: " << std::hex << "0x" << result << std::endl;
        return 1;
    }

    struct Data {
        int *A_d;
    } typedef Data;

    // Create a Data object and allocate memory for A_d on the device
    Data data;
    ze_device_mem_alloc_desc_t deviceMemAllocDesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
    result = zeMemAllocDevice(context, &deviceMemAllocDesc, sizeof(int), 1, deviceHandle, (void**)&data.A_d);
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "zeMemAllocDevice failed with error code: " << std::hex << "0x" << result << std::endl;
        return 1;
    }

    // Set the kernel argument to the Data object
    result = zeKernelSetArgumentValue(kernel, 0, sizeof(Data), &data);
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "zeKernelSetArgumentValue failed with error code: " << std::hex << "0x" << result << std::endl;
        return 1;
    }

    ze_command_queue_handle_t commandQueue;
    ze_command_queue_desc_t commandQueueDesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS};
    result = zeCommandQueueCreate(context, deviceHandle, &commandQueueDesc, &commandQueue);
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "zeCommandQueueCreate failed with error code: " << std::hex << "0x" << result << std::endl;
        return 1;
    }

    ze_command_list_handle_t commandList;
    ze_command_list_desc_t commandListDesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0};
    result = zeCommandListCreateImmediate(context, deviceHandle, &commandQueueDesc, &commandList);
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "zeCommandListCreateImmediate failed with error code: " << std::hex << "0x" << result << std::endl;
        return 1;
    }

    ze_group_count_t launchArgs = {1, 1, 1};
    result = zeCommandListAppendLaunchKernel(commandList, kernel, &launchArgs, nullptr, 0, nullptr);
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "zeCommandListAppendLaunchKernel failed with error code: " << std::hex << "0x" << result << std::endl;
        return 1;
    }

    result = zeCommandQueueSynchronize(commandQueue, UINT64_MAX);
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "zeCommandQueueSynchronize failed with error code: " << std::hex << "0x" << result << std::endl;
        return 1;
    }

    int A_h[1] = {0};
    result = zeCommandListAppendMemoryCopy(commandList, A_h, data.A_d, sizeof(int), nullptr, 0, nullptr);
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "zeCommandListAppendMemoryCopy failed with error code: " << std::hex << "0x" << result << std::endl;
        return 1;
    }

    result = zeCommandQueueSynchronize(commandQueue, UINT64_MAX);
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "zeCommandQueueSynchronize failed with error code: " << std::hex << "0x" << result << std::endl;
        return 1;
    }

    std::cout << "Result: " << A_h[0] << std::endl;
    if (A_h[0] == 1) {
        std::cout << "PASSED" << std::endl;
    } else {
        std::cout << "FAILED" << std::endl;
    }

    zeMemFree(context, data.A_d);
    zeKernelDestroy(kernel);
    zeModuleDestroy(module);
    zeCommandListDestroy(commandList);
    zeCommandQueueDestroy(commandQueue);
    zeContextDestroy(context);

    return 0;
}