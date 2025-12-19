// L0 bug: Cross-queue event sync hangs (Aurora) - MINIMAL REPRODUCER
#include <level_zero/ze_api.h>
#include <thread>
#include <atomic>
#include <iostream>
#include <vector>
#define C(x) do{ze_result_t r=(x);if(r!=ZE_RESULT_SUCCESS){std::cerr<<"FAIL:"<<#x<<" = "<<r<<"\n";return 1;}}while(0)

std::atomic<bool> stop{false};
ze_event_handle_t monitorEvt = nullptr;

void monitorThread() {
  while (!stop.load() && monitorEvt != nullptr) {
    zeEventHostSynchronize(monitorEvt, 1000);
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
  ze_command_list_handle_t q1,q2;
  C(zeCommandListCreateImmediate(ctx,dev,&qd,&q1));
  C(zeCommandListCreateImmediate(ctx,dev,&qd,&q2));
  
  ze_event_pool_desc_t pd={ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
  pd.flags=ZE_EVENT_POOL_FLAG_HOST_VISIBLE; pd.count=3;
  ze_event_pool_handle_t pool; C(zeEventPoolCreate(ctx,&pd,0,nullptr,&pool));
  
  ze_event_desc_t ed={ZE_STRUCTURE_TYPE_EVENT_DESC};
  ed.signal=ed.wait=ZE_EVENT_SCOPE_FLAG_HOST;
  ed.index=0; ze_event_handle_t syncEvt; C(zeEventCreate(pool,&ed,&syncEvt));
  ed.index=1; ze_event_handle_t marker; C(zeEventCreate(pool,&ed,&marker));
  
  ze_event_pool_desc_t mpd={ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
  mpd.flags=ZE_EVENT_POOL_FLAG_HOST_VISIBLE; mpd.count=1;
  ze_event_pool_handle_t mpool; C(zeEventPoolCreate(ctx,&mpd,0,nullptr,&mpool));
  ze_event_desc_t med={ZE_STRUCTURE_TYPE_EVENT_DESC};
  med.index=0; med.signal=med.wait=ZE_EVENT_SCOPE_FLAG_HOST;
  C(zeEventCreate(mpool,&med,&monitorEvt));
  C(zeEventHostSignal(monitorEvt));
  
  std::thread t(monitorThread);
  
  // BUG PATTERN: Signal on q1, barrier on q2 waits - WITHOUT host sync
  C(zeEventHostReset(syncEvt));
  C(zeCommandListAppendSignalEvent(q1,syncEvt));
  // MISSING: zeEventHostSynchronize(syncEvt) - this is the bug!
  C(zeCommandListAppendBarrier(q2,marker,1,&syncEvt));
  
  // Wait for marker - HANGS because q2 barrier never completes
  C(zeEventHostSynchronize(marker,UINT64_MAX));
  
  stop=true; zeEventHostSignal(monitorEvt); t.join();
  zeEventDestroy(syncEvt); zeEventDestroy(marker); zeEventDestroy(monitorEvt);
  zeEventPoolDestroy(pool); zeEventPoolDestroy(mpool);
  zeCommandListDestroy(q1); zeCommandListDestroy(q2); zeContextDestroy(ctx);
  std::cout<<"PASS\n"; return 0;
}
