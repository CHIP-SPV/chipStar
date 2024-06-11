#include <level_zero/ze_api.h>
#include <iostream>

int main() {
    ze_result_t result;
    ze_driver_handle_t hDriver = nullptr;
    ze_device_handle_t hDevice = nullptr;
    ze_context_handle_t hContext = nullptr;
    ze_command_queue_handle_t hCommandQueue = nullptr;
    ze_command_list_handle_t hCommandList = nullptr;
    void* pMemAllocDevice = nullptr;

    // Initialize the driver
    result = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "zeInit failed\n";
        return 1;
    }

    // Discover all the driver instances
    uint32_t driverCount = 0;
    result = zeDriverGet(&driverCount, nullptr);
    if (driverCount == 0) {
        std::cerr << "No drivers found\n";
        return 1;
    }

    result = zeDriverGet(&driverCount, &hDriver);
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "zeDriverGet failed\n";
        return 1;
    }

    // Get the first device of the driver
    uint32_t deviceCount = 0;
    result = zeDeviceGet(hDriver, &deviceCount, nullptr);
    if (deviceCount == 0) {
        std::cerr << "No devices found\n";
        return 1;
    }

    result = zeDeviceGet(hDriver, &deviceCount, &hDevice);
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "zeDeviceGet failed\n";
        return 1;
    }

    // Create a context
    ze_context_desc_t contextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    result = zeContextCreate(hDriver, &contextDesc, &hContext);
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "zeContextCreate failed\n";
        return 1;
    }

    // Create a command queue
    ze_command_queue_desc_t commandQueueDesc = {
        ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        nullptr,
        0, // ordinal
        0, // index
        0, // flags
        ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
        ZE_COMMAND_QUEUE_PRIORITY_NORMAL
    };
    result = zeCommandQueueCreate(hContext, hDevice, &commandQueueDesc, &hCommandQueue);
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "zeCommandQueueCreate failed\n";
        return 1;
    }

    // Create a command list
    ze_command_list_desc_t commandListDesc = {
        ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
        nullptr,
        0 // flags
    };
    result = zeCommandListCreateImmediate(hContext, hDevice, &commandQueueDesc, &hCommandList);
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "zeCommandListCreateImmediate failed\n";
        return 1;
    }

    // Allocate device memory
    ze_device_mem_alloc_desc_t deviceMemAllocDesc = {
        ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
        nullptr,
        0, // flags
        0  // ordinal
    };
    size_t size = 4096; // size of the memory to allocate
    result = zeMemAllocDevice(hContext, &deviceMemAllocDesc, size, 1, hDevice, &pMemAllocDevice);
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << "zeMemAllocDevice failed\n";
        return 1;
    }

    // Clean up resources
    zeMemFree(hContext, pMemAllocDevice);
    zeCommandListDestroy(hCommandList);
    zeCommandQueueDestroy(hCommandQueue);
    zeContextDestroy(hContext);

    return 0;
}