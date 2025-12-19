// Level Zero reproducer for callback-style host synchronization
// This simulates the pattern used by hipLaunchHostFunc where:
// 1. GPU work is submitted
// 2. A barrier waits on a host-signaled event
// 3. Host signals the event after some delay
// 4. Main thread tries to synchronize

#include <level_zero/ze_api.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <cstdlib>

#define CHECK_ZE(call) do { \
    ze_result_t result = (call); \
    if (result != ZE_RESULT_SUCCESS) { \
        std::cerr << "ZE Error: " << #call << " returned " << result << std::endl; \
        std::exit(1); \
    } \
} while(0)

int main() {
    std::cout << "Initializing Level Zero..." << std::endl;
    
    CHECK_ZE(zeInit(0));
    
    // Get driver and device
    uint32_t driverCount = 0;
    CHECK_ZE(zeDriverGet(&driverCount, nullptr));
    ze_driver_handle_t driver;
    CHECK_ZE(zeDriverGet(&driverCount, &driver));
    
    uint32_t deviceCount = 0;
    CHECK_ZE(zeDeviceGet(driver, &deviceCount, nullptr));
    ze_device_handle_t device;
    deviceCount = 1; // Just use first device
    CHECK_ZE(zeDeviceGet(driver, &deviceCount, &device));
    
    // Create context
    ze_context_desc_t contextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC};
    ze_context_handle_t context;
    CHECK_ZE(zeContextCreate(driver, &contextDesc, &context));
    
    // Create immediate command list (in-order)
    ze_command_queue_desc_t cmdQueueDesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
    cmdQueueDesc.ordinal = 0;
    cmdQueueDesc.index = 0;
    cmdQueueDesc.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
    cmdQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    cmdQueueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    
    ze_command_list_handle_t cmdList;
    CHECK_ZE(zeCommandListCreateImmediate(context, device, &cmdQueueDesc, &cmdList));
    
    // Create event pool with host-visible events
    ze_event_pool_desc_t eventPoolDesc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
    eventPoolDesc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    eventPoolDesc.count = 10;
    ze_event_pool_handle_t eventPool;
    CHECK_ZE(zeEventPoolCreate(context, &eventPoolDesc, 0, nullptr, &eventPool));
    
    // Create events
    ze_event_desc_t eventDesc = {ZE_STRUCTURE_TYPE_EVENT_DESC};
    eventDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    eventDesc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
    
    // Event that will be signaled by host (simulating CpuCallbackComplete)
    eventDesc.index = 0;
    ze_event_handle_t hostSignalEvent;
    CHECK_ZE(zeEventCreate(eventPool, &eventDesc, &hostSignalEvent));
    
    // Event that GPU signals after waiting on host event (simulating GpuAck)
    eventDesc.index = 1;
    ze_event_handle_t gpuAckEvent;
    CHECK_ZE(zeEventCreate(eventPool, &eventDesc, &gpuAckEvent));
    
    // Marker event for finish
    eventDesc.index = 2;
    ze_event_handle_t markerEvent;
    CHECK_ZE(zeEventCreate(eventPool, &eventDesc, &markerEvent));
    
    std::cout << "Appending barrier that waits on host-signaled event..." << std::endl;
    
    // Append barrier that waits on hostSignalEvent and signals gpuAckEvent
    // This simulates the callback's GpuAck barrier
    CHECK_ZE(zeCommandListAppendBarrier(cmdList, gpuAckEvent, 1, &hostSignalEvent));
    
    // Append another barrier with marker event (simulating finish marker)
    CHECK_ZE(zeCommandListAppendBarrier(cmdList, markerEvent, 0, nullptr));
    
    std::cout << "Starting host signal thread (simulating callback execution)..." << std::endl;
    
    // Start a thread that will signal the host event after a delay
    // This simulates the EventMonitor thread executing the callback
    std::thread signalThread([&]() {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "Host thread: Signaling event..." << std::endl;
        CHECK_ZE(zeEventHostSignal(hostSignalEvent));
        std::cout << "Host thread: Event signaled." << std::endl;
    });
    
    std::cout << "Main thread: Waiting on marker event (zeEventHostSynchronize)..." << std::endl;
    
    // Wait for the marker event - this should only complete after host signals
    ze_result_t syncResult = zeEventHostSynchronize(markerEvent, UINT64_MAX);
    if (syncResult != ZE_RESULT_SUCCESS) {
        std::cerr << "zeEventHostSynchronize failed: " << syncResult << std::endl;
    }
    
    std::cout << "Main thread: Marker event signaled." << std::endl;
    
    // Wait for command list to complete
    CHECK_ZE(zeCommandListHostSynchronize(cmdList, UINT64_MAX));
    
    std::cout << "Main thread: Command list synchronized." << std::endl;
    
    signalThread.join();
    
    // Cleanup
    zeEventDestroy(hostSignalEvent);
    zeEventDestroy(gpuAckEvent);
    zeEventDestroy(markerEvent);
    zeEventPoolDestroy(eventPool);
    zeCommandListDestroy(cmdList);
    zeContextDestroy(context);
    
    std::cout << "PASS" << std::endl;
    return 0;
}
