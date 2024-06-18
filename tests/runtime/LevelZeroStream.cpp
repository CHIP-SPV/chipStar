#include <level_zero/ze_api.h>
#include <iostream>
#include <thread>
#include <atomic>
#include <unistd.h>
#include <string.h>
#include <cstring>

#define PATH_MAX 4096

#define CHECK_RESULT(res) if (res != ZE_RESULT_SUCCESS) { std::cerr << "Error: " << res << " at line " << __LINE__ << std::endl; exit(res); }

std::atomic<bool> GpuReady1(false);
std::atomic<bool> GpuReady2(false);

void monitorGpuReady(std::atomic<bool>& GpuReady) {
    while (!GpuReady) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << "GPU is now ready." << std::endl;
}

int main() {
    std::cout << "Initializing Level Zero..." << std::endl;
    ze_result_t result = zeInit(0);
    CHECK_RESULT(result);

    std::cout << "Getting driver..." << std::endl;
    uint32_t driverCount = 0;
    result = zeDriverGet(&driverCount, nullptr);
    CHECK_RESULT(result);

    ze_driver_handle_t driver = nullptr;
    result = zeDriverGet(&driverCount, &driver);
    CHECK_RESULT(result);
    std::cout << "Getting devices..." << std::endl;
    uint32_t deviceCount = 0;
    result = zeDeviceGet(driver, &deviceCount, nullptr);
    CHECK_RESULT(result);

    if (deviceCount == 0) {
        std::cerr << "No devices found." << std::endl;
        exit(1);
    }

    ze_device_handle_t* devices = new ze_device_handle_t[deviceCount];
    result = zeDeviceGet(driver, &deviceCount, devices);
    CHECK_RESULT(result);
    ze_device_handle_t device = devices[0];

    std::cout << "Creating context for the first device..." << std::endl;
    ze_context_handle_t context;
    ze_context_desc_t contextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    result = zeContextCreate(driver, &contextDesc, &context);
    CHECK_RESULT(result);

    std::cout << "Loading kernel..." << std::endl;
    char kernelSource[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", kernelSource, PATH_MAX);
    if (count == -1) {
        std::cerr << "Failed to get executable path" << std::endl;
        exit(1);
    }
    std::string exePath = std::string(kernelSource, count);
    std::string exeDir = exePath.substr(0, exePath.find_last_of('/'));
    std::string kernelPath = exeDir + "/inputs/simple_kernel.spv";
    strncpy(kernelSource, kernelPath.c_str(), PATH_MAX);
    size_t kernelSize;
    uint8_t* pKernelBinary;

    FILE* fp = fopen(kernelSource, "rb");
    if (!fp) {
        std::cerr << "Failed to load kernel at " << kernelSource << std::endl;
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    kernelSize = ftell(fp);
    rewind(fp);

    pKernelBinary = new uint8_t[kernelSize];
    if (fread(pKernelBinary, 1, kernelSize, fp) != kernelSize) {
        std::cerr << "Failed to read kernel binary" << std::endl;
        exit(1);
    }
    fclose(fp);

    std::cout << "Creating module..." << std::endl;
    ze_module_handle_t module;
    ze_module_desc_t moduleDesc = {
        ZE_STRUCTURE_TYPE_MODULE_DESC,
        nullptr,
        ZE_MODULE_FORMAT_IL_SPIRV,
        kernelSize,
        pKernelBinary,
        nullptr,
        nullptr
    };
    result = zeModuleCreate(context, device, &moduleDesc, &module, nullptr);
    CHECK_RESULT(result);

    std::cout << "Creating kernel..." << std::endl;
    ze_kernel_handle_t kernel;
    ze_kernel_desc_t kernelDesc = {
        ZE_STRUCTURE_TYPE_KERNEL_DESC,
        nullptr,
        0,
        "simple_kernel"
    };
    result = zeKernelCreate(module, &kernelDesc, &kernel);
    CHECK_RESULT(result);

    std::cout << "Setting up command lists and queues..." << std::endl;
    ze_command_queue_desc_t cmdQueueDesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, ZE_COMMAND_QUEUE_PRIORITY_NORMAL, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS};
    ze_command_list_handle_t immCmdList;
    result = zeCommandListCreateImmediate(context, device, &cmdQueueDesc, &immCmdList);
    CHECK_RESULT(result);

    ze_command_list_handle_t cmdList;
    ze_command_list_desc_t cmdListDesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0};
    result = zeCommandListCreate(context, device, &cmdListDesc, &cmdList);
    CHECK_RESULT(result);

    ze_command_queue_handle_t cmdQueue;
    result = zeCommandQueueCreate(context, device, &cmdQueueDesc, &cmdQueue);
    CHECK_RESULT(result);

    std::cout << "Setting up synchronization..." << std::endl;

    // Create event pool of size 1
    ze_event_pool_handle_t eventPool1;
    ze_event_pool_desc_t eventPoolDesc1 = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr, ZE_EVENT_POOL_FLAG_HOST_VISIBLE, 1};
    result = zeEventPoolCreate(context, &eventPoolDesc1, 1, &device, &eventPool1);
    CHECK_RESULT(result);

    // Create event pool of size 2
    ze_event_pool_handle_t eventPool2;
    ze_event_pool_desc_t eventPoolDesc2 = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr, ZE_EVENT_POOL_FLAG_HOST_VISIBLE, 2};
    result = zeEventPoolCreate(context, &eventPoolDesc2, 1, &device, &eventPool2);
    CHECK_RESULT(result);

    // Create event pool of size 4
    ze_event_pool_handle_t eventPool4;
    ze_event_pool_desc_t eventPoolDesc4 = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr, ZE_EVENT_POOL_FLAG_HOST_VISIBLE, 4};
    result = zeEventPoolCreate(context, &eventPoolDesc4, 1, &device, &eventPool4);
    CHECK_RESULT(result);

    // Create events
    ze_event_handle_t userEvent, GpuReadyEvent, GpuCompleteEvent, UnusedEvent;
    ze_event_desc_t eventDesc = {ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, 0, ZE_EVENT_SCOPE_FLAG_HOST, ZE_EVENT_SCOPE_FLAG_HOST};

    // Create userEvent in eventPool1
    result = zeEventCreate(eventPool1, &eventDesc, &userEvent);
    CHECK_RESULT(result);

    // Create GpuReadyEvent in eventPool2
    eventDesc.index = 0;
    result = zeEventCreate(eventPool2, &eventDesc, &GpuReadyEvent);
    CHECK_RESULT(result);

    // Create GpuCompleteEvent in eventPool2
    eventDesc.index = 1;
    result = zeEventCreate(eventPool2, &eventDesc, &GpuCompleteEvent);
    CHECK_RESULT(result);

    // Create UnusedEvent in eventPool4
    eventDesc.index = 0;
    result = zeEventCreate(eventPool4, &eventDesc, &UnusedEvent);
    CHECK_RESULT(result);

    // Level Zero API calls based on the trace
    bool gpuReadyValue = true;
    zeCommandListAppendMemoryCopy(cmdList,  &GpuReady1, &gpuReadyValue, sizeof(GpuReady1), GpuReadyEvent,0, nullptr);

    zeCommandListAppendBarrier(cmdList, UnusedEvent, 1, &GpuReadyEvent);
    zeCommandListAppendBarrier(cmdList, GpuCompleteEvent, 1, &userEvent);

    result = zeCommandListClose(cmdList);
    CHECK_RESULT(result);
    result = zeCommandQueueExecuteCommandLists(cmdQueue, 1, &cmdList, nullptr);
    CHECK_RESULT(result);

    std::thread eventSyncThread([&]() {
        std::cout << "Waiting for user event..." << std::endl;
        ze_result_t res = ZE_RESULT_NOT_READY;
        while (res != ZE_RESULT_SUCCESS)
            res = zeEventHostSynchronize(userEvent, 1);
        std::cout << "GPU READY: Executing host callback!!!" << std::endl;
    });

    std::cout << "Signaling user event..." << std::endl;
    result = zeEventHostSignal(userEvent);
    CHECK_RESULT(result);

    std::cout << "Waiting for GPU to complete..." << std::endl;
    ze_result_t res = ZE_RESULT_NOT_READY;
    while (res != ZE_RESULT_SUCCESS)
        res = zeEventHostSynchronize(GpuCompleteEvent, 1);

    eventSyncThread.join();
    result = zeCommandListDestroy(immCmdList);
    CHECK_RESULT(result);
    result = zeCommandListDestroy(cmdList);
    CHECK_RESULT(result);
    result = zeContextDestroy(context);
    CHECK_RESULT(result);

    return 0;
}
