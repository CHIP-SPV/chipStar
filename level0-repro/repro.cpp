// Level Zero reproducer for callback-style host synchronization
// Mimics the exact pattern from TestHipLaunchHostFuncMultiStream:
// 1. Two streams (command lists)
// 2. kernel + memcpy + eventRecord on stream1
// 3. hipLaunchHostFunc on stream1 (barrier waiting on host signal)
// 4. hipStreamWaitEvent(stream2, event1)
// 5. More work on stream1
// 6. Another hipLaunchHostFunc on stream1
// 7. hipStreamSynchronize(stream1)

#include <level_zero/ze_api.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <cstdlib>
#include <vector>

#define CHECK_ZE(call)                                                         \
  do {                                                                         \
    ze_result_t result = (call);                                               \
    if (result != ZE_RESULT_SUCCESS) {                                         \
      std::cerr << "ZE Error: " << #call << " returned " << result             \
                << " at line " << __LINE__ << std::endl;                       \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

std::atomic<bool> callback1Ready{false};
std::atomic<bool> callback2Ready{false};
ze_event_handle_t hostSignalEvent1 = nullptr;
ze_event_handle_t hostSignalEvent2 = nullptr;
std::atomic<bool> done{false};

void callbackMonitorThread() {
  // Simulates the EventMonitor thread that executes callbacks
  while (!done.load()) {
    if (callback1Ready.load() && hostSignalEvent1) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      std::cout << "Monitor: Executing callback 1, signaling event..."
                << std::endl;
      CHECK_ZE(zeEventHostSignal(hostSignalEvent1));
      callback1Ready = false;
      hostSignalEvent1 = nullptr;
    }
    if (callback2Ready.load() && hostSignalEvent2) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      std::cout << "Monitor: Executing callback 2, signaling event..."
                << std::endl;
      CHECK_ZE(zeEventHostSignal(hostSignalEvent2));
      callback2Ready = false;
      hostSignalEvent2 = nullptr;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
}

int main() {
  std::cout << "Initializing Level Zero..." << std::endl;

  ze_result_t initResult = zeInit(0);
  std::cout << "zeInit returned: " << initResult << " (0x" << std::hex << initResult << std::dec << ")" << std::endl;
  if (initResult != ZE_RESULT_SUCCESS) {
    std::cerr << "zeInit failed!" << std::endl;
    return 1;
  }

  // Get driver and device
  uint32_t driverCount = 0;
  CHECK_ZE(zeDriverGet(&driverCount, nullptr));
  ze_driver_handle_t driver;
  CHECK_ZE(zeDriverGet(&driverCount, &driver));

  uint32_t deviceCount = 0;
  CHECK_ZE(zeDeviceGet(driver, &deviceCount, nullptr));
  std::cout << "Found " << deviceCount << " devices" << std::endl;
  
  std::vector<ze_device_handle_t> devices(deviceCount);
  CHECK_ZE(zeDeviceGet(driver, &deviceCount, devices.data()));
  
  // Use first device
  ze_device_handle_t device = devices[0];
  
  // Check for subdevices - use subdevice if available (mimics ZE_AFFINITY_MASK=0.0)
  uint32_t subDeviceCount = 0;
  CHECK_ZE(zeDeviceGetSubDevices(device, &subDeviceCount, nullptr));
  std::cout << "Device 0 has " << subDeviceCount << " subdevices" << std::endl;
  
  if (subDeviceCount > 0) {
    std::vector<ze_device_handle_t> subDevices(subDeviceCount);
    CHECK_ZE(zeDeviceGetSubDevices(device, &subDeviceCount, subDevices.data()));
    device = subDevices[0];  // Use subdevice 0, like ZE_AFFINITY_MASK=0.0
    std::cout << "Using subdevice 0" << std::endl;
  }

  // Create context
  ze_context_desc_t contextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC};
  ze_context_handle_t context;
  CHECK_ZE(zeContextCreate(driver, &contextDesc, &context));

  // Create two immediate command lists (simulating stream1 and stream2)
  ze_command_queue_desc_t cmdQueueDesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
  cmdQueueDesc.ordinal = 0;
  cmdQueueDesc.index = 0;
  cmdQueueDesc.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
  cmdQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  cmdQueueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

  ze_command_list_handle_t stream1;
  CHECK_ZE(zeCommandListCreateImmediate(context, device, &cmdQueueDesc, &stream1));

  ze_command_list_handle_t stream2;
  CHECK_ZE(zeCommandListCreateImmediate(context, device, &cmdQueueDesc, &stream2));

  // Create event pool
  ze_event_pool_desc_t eventPoolDesc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
  eventPoolDesc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
  eventPoolDesc.count = 20;
  ze_event_pool_handle_t eventPool;
  CHECK_ZE(zeEventPoolCreate(context, &eventPoolDesc, 0, nullptr, &eventPool));

  // Create events
  ze_event_desc_t eventDesc = {ZE_STRUCTURE_TYPE_EVENT_DESC};
  eventDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
  eventDesc.wait = ZE_EVENT_SCOPE_FLAG_HOST;

  int eventIndex = 0;
  auto createEvent = [&]() {
    ze_event_handle_t event;
    eventDesc.index = eventIndex++;
    CHECK_ZE(zeEventCreate(eventPool, &eventDesc, &event));
    return event;
  };

  // event1 - user event (for hipEventRecord)
  ze_event_handle_t event1 = createEvent();

  // Callback 1 events
  ze_event_handle_t cb1_hostSignal = createEvent();
  ze_event_handle_t cb1_gpuAck = createEvent();
  hostSignalEvent1 = cb1_hostSignal;

  // Callback 2 events
  ze_event_handle_t cb2_hostSignal = createEvent();
  ze_event_handle_t cb2_gpuAck = createEvent();
  hostSignalEvent2 = cb2_hostSignal;

  // Marker events for synchronization
  ze_event_handle_t markerEvent = createEvent();

  std::cout << "Starting monitor thread..." << std::endl;
  std::thread monitorThread(callbackMonitorThread);

  std::cout << "Simulating: kernel + memcpy on stream1..." << std::endl;
  // Signal event1 (simulating hipEventRecord after kernel+memcpy)
  CHECK_ZE(zeCommandListAppendSignalEvent(stream1, event1));

  std::cout << "Simulating: hipLaunchHostFunc on stream1 (callback 1)..."
            << std::endl;
  // Append barrier that waits on hostSignal and signals gpuAck
  CHECK_ZE(zeCommandListAppendBarrier(stream1, cb1_gpuAck, 1, &cb1_hostSignal));
  callback1Ready = true;

  std::cout << "Simulating: hipStreamWaitEvent(stream2, event1)..." << std::endl;
  // stream2 waits on event1
  CHECK_ZE(zeCommandListAppendBarrier(stream2, nullptr, 1, &event1));

  std::cout << "Simulating: more kernel + memcpy on stream1..." << std::endl;
  // Just another signal event
  ze_event_handle_t workEvent = createEvent();
  CHECK_ZE(zeCommandListAppendSignalEvent(stream1, workEvent));

  std::cout << "Simulating: hipLaunchHostFunc on stream1 (callback 2)..."
            << std::endl;
  // Append barrier for second callback
  CHECK_ZE(zeCommandListAppendBarrier(stream1, cb2_gpuAck, 1, &cb2_hostSignal));
  callback2Ready = true;

  std::cout << "Simulating: hipStreamSynchronize(stream1)..." << std::endl;
  // Append marker barrier
  CHECK_ZE(zeCommandListAppendBarrier(stream1, markerEvent, 0, nullptr));

  // Wait for marker event (this is what finish() does)
  std::cout << "Waiting on marker event..." << std::endl;
  ze_result_t syncResult = zeEventHostSynchronize(markerEvent, UINT64_MAX);
  if (syncResult != ZE_RESULT_SUCCESS) {
    std::cerr << "zeEventHostSynchronize failed: " << syncResult << std::endl;
    std::exit(1);
  }

  std::cout << "Marker event signaled, synchronizing command list..."
            << std::endl;
  CHECK_ZE(zeCommandListHostSynchronize(stream1, UINT64_MAX));

  std::cout << "Waiting for monitor thread..." << std::endl;
  done = true;
  monitorThread.join();

  // Cleanup
  zeEventDestroy(event1);
  zeEventDestroy(cb1_hostSignal);
  zeEventDestroy(cb1_gpuAck);
  zeEventDestroy(cb2_hostSignal);
  zeEventDestroy(cb2_gpuAck);
  zeEventDestroy(markerEvent);
  zeEventDestroy(workEvent);
  zeEventPoolDestroy(eventPool);
  zeCommandListDestroy(stream1);
  zeCommandListDestroy(stream2);
  zeContextDestroy(context);

  std::cout << "PASS" << std::endl;
  return 0;
}
