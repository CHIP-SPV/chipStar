// L0 bug: Host sync hangs when monitor thread also syncing (Aurora)
#include <level_zero/ze_api.h>
#include <thread>
#include <atomic>
#include <iostream>
#define C(x) if((x)!=ZE_RESULT_SUCCESS){std::cerr<<"FAIL:"<<#x<<"\n";return 1;}

std::atomic<bool> stop{false};
ze_event_handle_t monitorEvt;

void monitorThread() {
  while (!stop.load()) {
    zeEventHostSynchronize(monitorEvt, 1000);
  }
}

int main() {
  C(zeInit(0));
  uint32_t dc=0;
  ze_driver_handle_t drv; C(zeDriverGet(&dc,nullptr)); C(zeDriverGet(&dc,&drv));
  uint32_t nc=0;
  ze_device_handle_t dev; C(zeDeviceGet(drv,&nc,nullptr)); C(zeDeviceGet(drv,&nc,&dev));
  uint32_t sc=0; zeDeviceGetSubDevices(dev,&sc,nullptr);
  if(sc>0){ze_device_handle_t sds[16];uint32_t c=sc;zeDeviceGetSubDevices(dev,&c,sds);dev=sds[0];}
  
  ze_context_desc_t cd={ZE_STRUCTURE_TYPE_CONTEXT_DESC};
  ze_context_handle_t ctx; C(zeContextCreate(drv,&cd,&ctx));
  
  ze_command_queue_desc_t qd={ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
  qd.flags=ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
  ze_command_list_handle_t cl1,cl2;
  C(zeCommandListCreateImmediate(ctx,dev,&qd,&cl1));
  C(zeCommandListCreateImmediate(ctx,dev,&qd,&cl2));
  
  ze_event_pool_desc_t pd={ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
  pd.flags=ZE_EVENT_POOL_FLAG_HOST_VISIBLE; pd.count=4;
  ze_event_pool_handle_t pool; C(zeEventPoolCreate(ctx,&pd,0,nullptr,&pool));
  
  ze_event_desc_t ed={ZE_STRUCTURE_TYPE_EVENT_DESC};
  ed.signal=ed.wait=ZE_EVENT_SCOPE_FLAG_HOST;
  ed.index=0; C(zeEventCreate(pool,&ed,&monitorEvt));
  ed.index=1; ze_event_handle_t gpuReady; C(zeEventCreate(pool,&ed,&gpuReady));
  ed.index=2; ze_event_handle_t hostSignal; C(zeEventCreate(pool,&ed,&hostSignal));
  ed.index=3; ze_event_handle_t marker; C(zeEventCreate(pool,&ed,&marker));
  
  std::thread t(monitorThread);
  
  // cl1 signals gpuReady, waits for hostSignal, then signals marker
  C(zeCommandListAppendBarrier(cl1,gpuReady,0,nullptr));
  C(zeCommandListAppendBarrier(cl1,nullptr,1,&hostSignal));
  C(zeCommandListAppendBarrier(cl1,marker,0,nullptr));
  
  // Poll for gpuReady then signal hostSignal
  while(zeEventQueryStatus(gpuReady)!=ZE_RESULT_SUCCESS){}
  C(zeEventHostSignal(hostSignal));
  
  // Main thread waits for marker - THIS HANGS
  C(zeEventHostSynchronize(marker,UINT64_MAX));
  
  stop=true; zeEventHostSignal(monitorEvt); t.join();
  zeEventDestroy(monitorEvt); zeEventDestroy(gpuReady);
  zeEventDestroy(hostSignal); zeEventDestroy(marker);
  zeEventPoolDestroy(pool); zeCommandListDestroy(cl1);
  zeCommandListDestroy(cl2); zeContextDestroy(ctx);
  std::cout<<"PASS\n"; return 0;
}
