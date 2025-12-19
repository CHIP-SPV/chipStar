// L0 bug: zeEventHostSynchronize hangs with concurrent event pool creation (Aurora)
#include <level_zero/ze_api.h>
#include <thread>
#include <atomic>
#include <iostream>
#include <chrono>
#define C(x) if((x)!=ZE_RESULT_SUCCESS){std::cerr<<"FAIL:"<<#x<<"\n";return 1;}

std::atomic<bool> stop{false};
ze_event_handle_t evt = nullptr;

void syncThread() {
  while (!stop.load()) {
    if (evt != nullptr) {
      zeEventHostSynchronize(evt, 1000);
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
}

int main() {
  std::cerr<<"1\n"; C(zeInit(0));
  std::cerr<<"2\n"; uint32_t dc=0;
  std::cerr<<"3\n"; ze_driver_handle_t drv; C(zeDriverGet(&dc,nullptr)); C(zeDriverGet(&dc,&drv));
  if(dc==0){std::cerr<<"No drivers\n";return 1;}
  std::cerr<<"4\n"; uint32_t nc=0;
  std::cerr<<"5\n"; ze_device_handle_t dev; C(zeDeviceGet(drv,&nc,nullptr));
  if(nc==0){std::cerr<<"No devices\n";return 1;}
  std::cerr<<"6\n"; C(zeDeviceGet(drv,&nc,&dev));
  std::cerr<<"7\n"; uint32_t sc=0; zeDeviceGetSubDevices(dev,&sc,nullptr);
  if(sc>0){ze_device_handle_t sds[16];uint32_t c=sc;zeDeviceGetSubDevices(dev,&c,sds);dev=sds[0];}
  std::cerr<<"8\n";
  ze_context_desc_t cd={ZE_STRUCTURE_TYPE_CONTEXT_DESC};
  std::cerr<<"9\n"; ze_context_handle_t ctx; C(zeContextCreate(drv,&cd,&ctx));
  std::cerr<<"10\n";
  
  ze_command_queue_desc_t qd={ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
  qd.flags=ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
  ze_command_list_handle_t cl; C(zeCommandListCreateImmediate(ctx,dev,&qd,&cl));
  
  ze_event_pool_desc_t pd={ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
  pd.flags=ZE_EVENT_POOL_FLAG_HOST_VISIBLE; pd.count=1;
  ze_event_pool_handle_t pool; C(zeEventPoolCreate(ctx,&pd,0,nullptr,&pool));
  
  ze_event_desc_t ed={ZE_STRUCTURE_TYPE_EVENT_DESC};
  ed.index=0;
  ed.signal=ed.wait=ZE_EVENT_SCOPE_FLAG_HOST;
  C(zeEventCreate(pool,&ed,&evt));
  if(evt==nullptr){std::cerr<<"Event creation failed\n";return 1;}
  
  // Signal once to ensure event is valid
  C(zeEventHostReset(evt));
  C(zeEventHostSignal(evt));
  
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
