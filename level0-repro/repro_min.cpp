// L0 bug: zeEventHostSynchronize hangs with concurrent event pool creation (Aurora)
#include <level_zero/ze_api.h>
#include <thread>
#include <atomic>
#include <iostream>
#define C(x) if((x)!=ZE_RESULT_SUCCESS){std::cerr<<"FAIL:"<<#x<<"\n";return 1;}

std::atomic<bool> stop{false};
ze_event_handle_t evt;

void syncThread() {
  while (!stop.load()) {
    zeEventHostSynchronize(evt, 1000);
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
  ze_command_list_handle_t cl; C(zeCommandListCreateImmediate(ctx,dev,&qd,&cl));
  
  ze_event_pool_desc_t pd={ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
  pd.flags=ZE_EVENT_POOL_FLAG_HOST_VISIBLE; pd.count=1;
  ze_event_pool_handle_t pool; C(zeEventPoolCreate(ctx,&pd,0,nullptr,&pool));
  
  ze_event_desc_t ed={ZE_STRUCTURE_TYPE_EVENT_DESC};
  ed.signal=ed.wait=ZE_EVENT_SCOPE_FLAG_HOST;
  C(zeEventCreate(pool,&ed,&evt));
  
  std::thread t(syncThread);
  
  for(int i=0;i<10;i++){
    ze_event_pool_handle_t p;
    C(zeEventPoolCreate(ctx,&pd,0,nullptr,&p));
    zeEventPoolDestroy(p);
  }
  
  stop=true; zeEventHostSignal(evt); t.join();
  zeEventDestroy(evt); zeEventPoolDestroy(pool);
  zeCommandListDestroy(cl); zeContextDestroy(ctx);
  std::cout<<"PASS\n"; return 0;
}
