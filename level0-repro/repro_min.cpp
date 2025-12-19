// L0 bug: zeEventHostSynchronize hangs with concurrent event pool creation (Aurora)
#include <level_zero/ze_api.h>
#include <thread>
#include <atomic>
#include <iostream>
#include <vector>
#define C(x) do{ze_result_t r=(x);if(r!=ZE_RESULT_SUCCESS){std::cerr<<"FAIL:"<<#x<<" = "<<r<<"\n";return 1;}}while(0)

std::atomic<bool> stop{false};
ze_event_handle_t evt = nullptr;

void syncThread() {
  while (!stop.load()) {
    if (evt != nullptr) {
      zeEventHostSynchronize(evt, 1000);
    }
  }
}

int main() {
  ze_result_t initResult = zeInit(0);
  if (initResult != ZE_RESULT_SUCCESS) {
    std::cerr << "zeInit failed: " << initResult << std::endl;
    return 1;
  }

  uint32_t driverCount = 0;
  C(zeDriverGet(&driverCount, nullptr));
  if(driverCount == 0) {
    std::cerr << "No drivers found\n";
    return 1;
  }
  ze_driver_handle_t driver;
  C(zeDriverGet(&driverCount, &driver));

  uint32_t deviceCount = 0;
  C(zeDeviceGet(driver, &deviceCount, nullptr));
  if(deviceCount == 0) {
    std::cerr << "No devices found\n";
    return 1;
  }
  std::vector<ze_device_handle_t> devices(deviceCount);
  C(zeDeviceGet(driver, &deviceCount, devices.data()));
  ze_device_handle_t device = devices[0];

  uint32_t subDeviceCount = 0;
  zeDeviceGetSubDevices(device, &subDeviceCount, nullptr);
  if (subDeviceCount > 0) {
    std::vector<ze_device_handle_t> subDevices(subDeviceCount);
    uint32_t count = subDeviceCount;
    C(zeDeviceGetSubDevices(device, &count, subDevices.data()));
    device = subDevices[0];
  }

  ze_context_desc_t contextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC};
  ze_context_handle_t context;
  C(zeContextCreate(driver, &contextDesc, &context));

  ze_command_queue_desc_t cmdQueueDesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
  cmdQueueDesc.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
  ze_command_list_handle_t commandList;
  C(zeCommandListCreateImmediate(context, device, &cmdQueueDesc, &commandList));

  ze_event_pool_desc_t poolDesc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
  poolDesc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
  poolDesc.count = 1;
  ze_event_pool_handle_t pool;
  C(zeEventPoolCreate(context, &poolDesc, 0, nullptr, &pool));

  ze_event_desc_t eventDesc = {ZE_STRUCTURE_TYPE_EVENT_DESC};
  eventDesc.index = 0;
  eventDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
  eventDesc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
  C(zeEventCreate(pool, &eventDesc, &evt));

  std::thread t(syncThread);

  for(int i=0; i<10; i++) {
    ze_event_pool_handle_t p;
    C(zeEventPoolCreate(context, &poolDesc, 0, nullptr, &p));
    zeEventPoolDestroy(p);
  }

  stop = true;
  zeEventHostSignal(evt);
  t.join();
  zeEventDestroy(evt);
  zeEventPoolDestroy(pool);
  zeCommandListDestroy(commandList);
  zeContextDestroy(context);
  std::cout << "PASS\n";
  return 0;
}
