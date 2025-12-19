// L0 bug: Cross-queue event sync hangs (Aurora) - MINIMAL REPRODUCER
// Removes workaround to demonstrate the bug
#include <level_zero/ze_api.h>
#include <thread>
#include <atomic>
#include <iostream>
#include <vector>
#include <chrono>
#define C(x) do{ze_result_t r=(x);if(r!=ZE_RESULT_SUCCESS){std::cerr<<"FAIL:"<<#x<<" = "<<r<<"\n";return 1;}}while(0)

std::atomic<bool> stop{false};
ze_event_handle_t gpuAck1 = nullptr;

void monitorThread() {
  // Simulate monitor processing callback 1 - blocks on zeEventHostSynchronize
  if (gpuAck1 != nullptr) {
    zeEventHostSynchronize(gpuAck1, UINT64_MAX);
  }
}

int main() {
  C(zeInit(0));
  uint32_t dc=0; ze_driver_handle_t drv; C(zeDriverGet(&dc,nullptr)); C(zeDriverGet(&dc,&drv));
  uint32_t nc=0; std::vector<ze_device_handle_t> devs(16); C(zeDeviceGet(drv,&nc,nullptr)); C(zeDeviceGet(drv,&nc,devs.data()));
  ze_device_handle_t dev=devs[0];
  uint32_t sc=0; zeDeviceGetSubDevices(dev,&sc,nullptr);
  if(sc>0){std::vector<ze_device_handle_t> sds(16);uint32_t c=sc;C(zeDeviceGetSubDevices(dev,&c,sds.data()));dev=sds[0];}
  
  ze_context_desc_t cd={ZE_STRUCTURE_TYPE_CONTEXT_DESC};
  ze_context_handle_t ctx; C(zeContextCreate(drv,&cd,&ctx));
  
  ze_command_queue_desc_t qd={ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
  qd.flags=ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
  qd.mode=ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  ze_command_list_handle_t defaultQueue,stream1;
  C(zeCommandListCreateImmediate(ctx,dev,&qd,&defaultQueue));
  C(zeCommandListCreateImmediate(ctx,dev,&qd,&stream1));
  
  ze_event_pool_desc_t pd={ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
  pd.flags=ZE_EVENT_POOL_FLAG_HOST_VISIBLE; pd.count=5;
  ze_event_pool_handle_t pool; C(zeEventPoolCreate(ctx,&pd,0,nullptr,&pool));
  
  ze_event_desc_t ed={ZE_STRUCTURE_TYPE_EVENT_DESC};
  ed.signal=ed.wait=ZE_EVENT_SCOPE_FLAG_HOST;
  ed.index=0; ze_event_handle_t syncEvent1; C(zeEventCreate(pool,&ed,&syncEvent1));
  ed.index=1; ze_event_handle_t gpuReady1; C(zeEventCreate(pool,&ed,&gpuReady1));
  ed.index=2; C(zeEventCreate(pool,&ed,&gpuAck1));
  ed.index=3; ze_event_handle_t syncEvent2; C(zeEventCreate(pool,&ed,&syncEvent2));
  ed.index=4; ze_event_handle_t marker; C(zeEventCreate(pool,&ed,&marker));
  
  // Callback 1 setup
  C(zeEventHostReset(syncEvent1));
  C(zeEventHostReset(gpuReady1));
  C(zeEventHostReset(gpuAck1));
  C(zeCommandListAppendSignalEvent(defaultQueue,syncEvent1));
  // BUG: Missing zeEventHostSynchronize workaround
  C(zeCommandListAppendBarrier(stream1,gpuReady1,1,&syncEvent1));
  C(zeCommandListAppendBarrier(stream1,gpuAck1,0,nullptr));
  
  // Start monitor thread - will block on zeEventHostSynchronize(gpuAck1)
  std::thread t(monitorThread);
  
  // Signal gpuReady1 from host (simulating GPU completion)
  C(zeEventHostSignal(gpuReady1));
  
  // Signal gpuAck1 from GPU side so monitor thread blocks
  C(zeCommandListAppendBarrier(stream1,nullptr,0,nullptr));
  C(zeCommandListAppendSignalEvent(stream1,gpuAck1));
  
  // Small delay to let monitor thread start blocking
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  
  // Callback 2 setup - while monitor is BLOCKED on zeEventHostSynchronize(gpuAck1)
  C(zeEventHostReset(syncEvent2));
  C(zeCommandListAppendSignalEvent(defaultQueue,syncEvent2));
  // BUG: Missing zeEventHostSynchronize workaround - THIS CAUSES HANG
  C(zeCommandListAppendBarrier(stream1,marker,1,&syncEvent2));
  
  // Wait for marker - HANGS
  C(zeEventHostSynchronize(marker,UINT64_MAX));
  
  stop=true; C(zeEventHostSignal(gpuAck1)); t.join();
  zeEventDestroy(syncEvent1); zeEventDestroy(gpuReady1); zeEventDestroy(gpuAck1);
  zeEventDestroy(syncEvent2); zeEventDestroy(marker);
  zeEventPoolDestroy(pool);
  zeCommandListDestroy(defaultQueue); zeCommandListDestroy(stream1);
  zeContextDestroy(ctx);
  std::cout<<"PASS\n"; return 0;
}
