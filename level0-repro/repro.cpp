// Level Zero reproducer that exactly matches chipStar's hipLaunchHostFunc pattern
// Key patterns from l0.trace:
// 1. Separate event pool per callback event (each with count: 1)
// 2. Signal sync event on DEFAULT QUEUE before callback barriers
// 3. Stream1 waits for sync event, signals GpuReady
// 4. Stream1 waits for HostSignal, signals GpuAck
// 5. Stream1 signals FinalAck (no wait)
// 6. Monitor thread: polls GpuReady, signals HostSignal, waits for GpuAck

#include <level_zero/ze_api.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <cstdlib>
#include <vector>
#include <mutex>
#include <algorithm>

#define CHECK_ZE(call)                                                         \
  do {                                                                          \
    ze_result_t result = (call);                                               \
    if (result != ZE_RESULT_SUCCESS) {                                         \
      std::cerr << "ZE Error: " << #call << " returned " << result             \
                << " (0x" << std::hex << result << std::dec << ")"             \
                << " at line " << __LINE__ << std::endl;                       \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// Callback data structure - matches chipStar's CHIPCallbackDataLevel0
struct CallbackData {
  ze_event_handle_t GpuReady = nullptr;
  ze_event_handle_t HostSignal = nullptr;
  ze_event_handle_t GpuAck = nullptr;
  ze_event_handle_t GpuAckDone = nullptr;
  ze_event_pool_handle_t GpuReadyPool = nullptr;
  ze_event_pool_handle_t HostSignalPool = nullptr;
  ze_event_pool_handle_t GpuAckPool = nullptr;
  ze_event_pool_handle_t GpuAckDonePool = nullptr;
  std::atomic<bool> ready{false};
  int id = 0;
};

std::atomic<bool> done{false};
std::mutex callbackMtx;
std::vector<CallbackData*> pendingCallbacks;

static std::atomic<int> poolCounter{0};

ze_event_handle_t createEventWithPool(ze_context_handle_t context,
                                       ze_event_pool_handle_t& outPool) {
  int id = poolCounter++;
  std::cout << "[pool" << id << ":create..." << std::flush;
  ze_event_pool_desc_t poolDesc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
  poolDesc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
  poolDesc.count = 1;
  CHECK_ZE(zeEventPoolCreate(context, &poolDesc, 0, nullptr, &outPool));
  std::cout << "evt..." << std::flush;

  ze_event_desc_t eventDesc = {ZE_STRUCTURE_TYPE_EVENT_DESC};
  eventDesc.index = 0;
  eventDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
  eventDesc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
  
  ze_event_handle_t event;
  CHECK_ZE(zeEventCreate(outPool, &eventDesc, &event));
  std::cout << "ok]" << std::flush;
  return event;
}

void callbackMonitorThread() {
  std::cout << "Monitor thread started" << std::endl;
  
  while (!done.load()) {
    CallbackData* cbData = nullptr;
    
    {
      std::lock_guard<std::mutex> lock(callbackMtx);
      for (auto* cb : pendingCallbacks) {
        if (cb->ready.load() && cb->GpuReady) {
          ze_result_t status = zeEventQueryStatus(cb->GpuReady);
          if (status == ZE_RESULT_SUCCESS) {
            cbData = cb;
            cb->ready = false;
            break;
          }
        }
      }
    }
    
    if (cbData) {
      std::cout << "Monitor: Callback " << cbData->id << " - GpuReady signaled" << std::endl;
      
      std::cout << "Monitor: Signaling HostSignal for callback " << cbData->id << std::endl;
      CHECK_ZE(zeEventHostSignal(cbData->HostSignal));
      
      std::cout << "Monitor: Waiting for GpuAck for callback " << cbData->id << std::endl;
      CHECK_ZE(zeEventHostSynchronize(cbData->GpuAck, UINT64_MAX));
      std::cout << "Monitor: GpuAck received for callback " << cbData->id << std::endl;
      
      zeEventDestroy(cbData->GpuAck);
      zeEventDestroy(cbData->HostSignal);
      zeEventDestroy(cbData->GpuReady);
      zeEventDestroy(cbData->GpuAckDone);
      zeEventPoolDestroy(cbData->GpuAckPool);
      zeEventPoolDestroy(cbData->HostSignalPool);
      zeEventPoolDestroy(cbData->GpuReadyPool);
      zeEventPoolDestroy(cbData->GpuAckDonePool);
      
      {
        std::lock_guard<std::mutex> lock(callbackMtx);
        pendingCallbacks.erase(
          std::remove(pendingCallbacks.begin(), pendingCallbacks.end(), cbData),
          pendingCallbacks.end());
      }
      
      delete cbData;
    } else {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  }
  
  std::cout << "Monitor thread exiting" << std::endl;
}

int main() {
  std::cout << "Initializing Level Zero..." << std::endl;

  ze_result_t initResult = zeInit(0);
  std::cout << "zeInit returned: " << initResult << std::endl;
  if (initResult != ZE_RESULT_SUCCESS) {
    std::cerr << "zeInit failed!" << std::endl;
    return 1;
  }

  uint32_t driverCount = 0;
  CHECK_ZE(zeDriverGet(&driverCount, nullptr));
  ze_driver_handle_t driver;
  CHECK_ZE(zeDriverGet(&driverCount, &driver));

  uint32_t deviceCount = 0;
  CHECK_ZE(zeDeviceGet(driver, &deviceCount, nullptr));
  std::cout << "Found " << deviceCount << " devices" << std::endl;
  
  std::vector<ze_device_handle_t> devices(deviceCount);
  CHECK_ZE(zeDeviceGet(driver, &deviceCount, devices.data()));
  
  ze_device_handle_t device = devices[0];
  
  uint32_t subDeviceCount = 0;
  CHECK_ZE(zeDeviceGetSubDevices(device, &subDeviceCount, nullptr));
  std::cout << "Device 0 has " << subDeviceCount << " subdevices" << std::endl;
  
  if (subDeviceCount > 0) {
    std::vector<ze_device_handle_t> subDevices(subDeviceCount);
    CHECK_ZE(zeDeviceGetSubDevices(device, &subDeviceCount, subDevices.data()));
    device = subDevices[0];
    std::cout << "Using subdevice 0" << std::endl;
  }

  ze_context_desc_t contextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC};
  ze_context_handle_t context;
  CHECK_ZE(zeContextCreate(driver, &contextDesc, &context));

  ze_command_queue_desc_t cmdQueueDesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
  cmdQueueDesc.ordinal = 0;
  cmdQueueDesc.index = 0;
  cmdQueueDesc.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
  cmdQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  cmdQueueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

  ze_command_list_handle_t defaultQueue;
  CHECK_ZE(zeCommandListCreateImmediate(context, device, &cmdQueueDesc, &defaultQueue));
  
  ze_command_list_handle_t stream1;
  CHECK_ZE(zeCommandListCreateImmediate(context, device, &cmdQueueDesc, &stream1));

  ze_command_list_handle_t stream2;
  CHECK_ZE(zeCommandListCreateImmediate(context, device, &cmdQueueDesc, &stream2));
  
  std::cout << "Created 3 command lists" << std::endl;

  ze_event_pool_desc_t syncPoolDesc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
  syncPoolDesc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
  syncPoolDesc.count = 20;
  ze_event_pool_handle_t syncEventPool;
  CHECK_ZE(zeEventPoolCreate(context, &syncPoolDesc, 0, nullptr, &syncEventPool));

  int eventIndex = 0;
  auto createSyncEvent = [&]() {
    ze_event_desc_t eventDesc = {ZE_STRUCTURE_TYPE_EVENT_DESC};
    eventDesc.index = eventIndex++;
    eventDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    eventDesc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
    ze_event_handle_t event;
    CHECK_ZE(zeEventCreate(syncEventPool, &eventDesc, &event));
    return event;
  };

  ze_event_handle_t syncEvent1 = createSyncEvent();
  ze_event_handle_t syncEvent2 = createSyncEvent();
  ze_event_handle_t event1 = createSyncEvent();
  ze_event_handle_t workEvent = createSyncEvent();
  ze_event_handle_t markerEvent = createSyncEvent();

  std::cout << "Starting monitor thread..." << std::endl;
  std::thread monitorThread(callbackMonitorThread);

  std::cout << "Simulating: kernel + memcpy on stream1..." << std::endl;
  CHECK_ZE(zeCommandListAppendSignalEvent(stream1, event1));

  std::cout << "Simulating: hipLaunchHostFunc (callback 1)..." << std::endl;
  {
    CallbackData* cb1 = new CallbackData();
    cb1->id = 1;
    
    cb1->GpuReady = createEventWithPool(context, cb1->GpuReadyPool);
    cb1->HostSignal = createEventWithPool(context, cb1->HostSignalPool);
    cb1->GpuAck = createEventWithPool(context, cb1->GpuAckPool);
    cb1->GpuAckDone = createEventWithPool(context, cb1->GpuAckDonePool);
    
    CHECK_ZE(zeEventHostReset(syncEvent1));
    
    // FIX: Add barrier BEFORE signaling to avoid race
    CHECK_ZE(zeCommandListAppendBarrier(stream1, cb1->GpuReady, 1, &syncEvent1));
    CHECK_ZE(zeCommandListAppendSignalEvent(defaultQueue, syncEvent1));
    CHECK_ZE(zeCommandListAppendBarrier(stream1, cb1->GpuAck, 1, &cb1->HostSignal));
    CHECK_ZE(zeCommandListAppendBarrier(stream1, cb1->GpuAckDone, 0, nullptr));
    
    {
      std::lock_guard<std::mutex> lock(callbackMtx);
      pendingCallbacks.push_back(cb1);
    }
    cb1->ready = true;
  }

  std::cout << "Simulating: hipStreamWaitEvent(stream2, event1)..." << std::endl;
  CHECK_ZE(zeCommandListAppendBarrier(stream2, nullptr, 1, &event1));

  std::cout << "Simulating: more kernel + memcpy on stream1..." << std::endl;
  CHECK_ZE(zeCommandListAppendSignalEvent(stream1, workEvent));

  std::cout << "Simulating: hipLaunchHostFunc (callback 2)..." << std::flush;
  {
    CallbackData* cb2 = new CallbackData();
    cb2->id = 2;
    
    std::cout << " creating events..." << std::flush;
    cb2->GpuReady = createEventWithPool(context, cb2->GpuReadyPool);
    cb2->HostSignal = createEventWithPool(context, cb2->HostSignalPool);
    cb2->GpuAck = createEventWithPool(context, cb2->GpuAckPool);
    cb2->GpuAckDone = createEventWithPool(context, cb2->GpuAckDonePool);
    
    std::cout << " reset..." << std::flush;
    CHECK_ZE(zeEventHostReset(syncEvent2));
    
    // FIX: Add barrier BEFORE signaling to avoid race where event is signaled
    // before barrier is added (immediate command lists execute immediately)
    std::cout << " barrier1(wait sync)..." << std::flush;
    CHECK_ZE(zeCommandListAppendBarrier(stream1, cb2->GpuReady, 1, &syncEvent2));
    
    std::cout << " signal on defaultQueue..." << std::flush;
    CHECK_ZE(zeCommandListAppendSignalEvent(defaultQueue, syncEvent2));
    std::cout << " barrier2(wait host)..." << std::flush;
    CHECK_ZE(zeCommandListAppendBarrier(stream1, cb2->GpuAck, 1, &cb2->HostSignal));
    std::cout << " barrier3..." << std::flush;
    CHECK_ZE(zeCommandListAppendBarrier(stream1, cb2->GpuAckDone, 0, nullptr));
    
    std::cout << " done" << std::endl;
    {
      std::lock_guard<std::mutex> lock(callbackMtx);
      pendingCallbacks.push_back(cb2);
    }
    cb2->ready = true;
  }

  std::cout << "Simulating: hipStreamSynchronize(stream1)..." << std::endl;
  CHECK_ZE(zeCommandListAppendBarrier(stream1, markerEvent, 0, nullptr));

  std::cout << "Waiting on marker event..." << std::endl;
  ze_result_t syncResult = zeEventHostSynchronize(markerEvent, UINT64_MAX);
  if (syncResult != ZE_RESULT_SUCCESS) {
    std::cerr << "zeEventHostSynchronize failed: " << syncResult << std::endl;
    std::exit(1);
  }

  std::cout << "Marker signaled, synchronizing..." << std::endl;
  CHECK_ZE(zeCommandListHostSynchronize(stream1, UINT64_MAX));

  std::cout << "Waiting for monitor thread..." << std::endl;
  done = true;
  monitorThread.join();

  zeEventDestroy(syncEvent1);
  zeEventDestroy(syncEvent2);
  zeEventDestroy(event1);
  zeEventDestroy(workEvent);
  zeEventDestroy(markerEvent);
  zeEventPoolDestroy(syncEventPool);
  zeCommandListDestroy(defaultQueue);
  zeCommandListDestroy(stream1);
  zeCommandListDestroy(stream2);
  zeContextDestroy(context);

  std::cout << "PASS" << std::endl;
  return 0;
}
