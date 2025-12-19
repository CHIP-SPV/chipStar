// Minimal Level Zero reproducer - zeEventPoolCreate hangs during concurrent zeEventHostSynchronize
#include <level_zero/ze_api.h>
#include <thread>
#include <atomic>
#include <iostream>

#define CHECK(x) if((x) != ZE_RESULT_SUCCESS) { std::cerr << "FAIL: " << #x << std::endl; return 1; }

std::atomic<bool> stop{false};
ze_context_handle_t ctx;
ze_event_handle_t waitEvent;

void syncThread() {
  while (!stop) {
    // This blocks zeEventPoolCreate in main thread
    zeEventHostSynchronize(waitEvent, 1000);
  }
}

int main() {
  CHECK(zeInit(0));
  
  uint32_t drvCnt = 0;
  CHECK(zeDriverGet(&drvCnt, nullptr));
  ze_driver_handle_t drv;
  CHECK(zeDriverGet(&drvCnt, &drv));
  
  uint32_t devCnt = 0;
  CHECK(zeDeviceGet(drv, &devCnt, nullptr));
  ze_device_handle_t dev;
  CHECK(zeDeviceGet(drv, &devCnt, &dev));
  
  ze_context_desc_t ctxDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC};
  CHECK(zeContextCreate(drv, &ctxDesc, &ctx));
  
  // Create event for sync thread to wait on
  ze_event_pool_desc_t poolDesc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
  poolDesc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
  poolDesc.count = 1;
  ze_event_pool_handle_t pool;
  CHECK(zeEventPoolCreate(ctx, &poolDesc, 0, nullptr, &pool));
  
  ze_event_desc_t evtDesc = {ZE_STRUCTURE_TYPE_EVENT_DESC};
  evtDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
  evtDesc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
  CHECK(zeEventCreate(pool, &evtDesc, &waitEvent));
  
  std::thread t(syncThread);
  
  // Create pools while sync thread is in zeEventHostSynchronize - this hangs
  for (int i = 0; i < 10; i++) {
    std::cout << i << std::flush;
    ze_event_pool_handle_t p;
    CHECK(zeEventPoolCreate(ctx, &poolDesc, 0, nullptr, &p));  // HANGS HERE
    zeEventPoolDestroy(p);
  }
  std::cout << " PASS" << std::endl;
  
  stop = true;
  zeEventHostSignal(waitEvent);
  t.join();
  
  zeEventDestroy(waitEvent);
  zeEventPoolDestroy(pool);
  zeContextDestroy(ctx);
  return 0;
}
