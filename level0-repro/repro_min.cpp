// Minimal L0 reproducer: zeEventPoolCreate hangs with concurrent zeEventHostSynchronize
#include <level_zero/ze_api.h>
#include <thread>
#include <atomic>
#include <iostream>

#define C(x) if((x)!=ZE_RESULT_SUCCESS){std::cerr<<"FAIL:"<<#x<<"\n";return 1;}

std::atomic<bool> stop{false};
ze_event_handle_t syncEvt;

void monitorThread() {
  while(!stop) zeEventHostSynchronize(syncEvt, 1000);
}

int main() {
  std::cout<<"1"<<std::flush; C(zeInit(0));
  
  uint32_t n=0; C(zeDriverGet(&n,nullptr));
  ze_driver_handle_t drv; C(zeDriverGet(&n,&drv));
  std::cout<<"2"<<std::flush;
  
  C(zeDeviceGet(drv,&n,nullptr));
  ze_device_handle_t dev; C(zeDeviceGet(drv,&n,&dev));
  std::cout<<"3"<<std::flush;
  
  ze_context_desc_t cd={ZE_STRUCTURE_TYPE_CONTEXT_DESC};
  ze_context_handle_t ctx; C(zeContextCreate(drv,&cd,&ctx));
  std::cout<<"4"<<std::flush;
  
  // Create immediate command list (required to trigger bug)
  ze_command_queue_desc_t qd={ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
  qd.flags=ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
  qd.mode=ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  ze_command_list_handle_t cl;
  C(zeCommandListCreateImmediate(ctx,dev,&qd,&cl));
  std::cout<<"5"<<std::flush;
  
  ze_event_pool_desc_t pd={ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
  pd.flags=ZE_EVENT_POOL_FLAG_HOST_VISIBLE; pd.count=1;
  ze_event_pool_handle_t pool; C(zeEventPoolCreate(ctx,&pd,0,nullptr,&pool));
  
  ze_event_desc_t ed={ZE_STRUCTURE_TYPE_EVENT_DESC};
  ed.signal=ZE_EVENT_SCOPE_FLAG_HOST; ed.wait=ZE_EVENT_SCOPE_FLAG_HOST;
  C(zeEventCreate(pool,&ed,&syncEvt));
  std::cout<<"6"<<std::flush;
  
  std::thread t(monitorThread);
  
  // This loop hangs in zeEventPoolCreate
  for(int i=0;i<10;i++){
    std::cout<<(char)('a'+i)<<std::flush;
    ze_event_pool_handle_t p;
    C(zeEventPoolCreate(ctx,&pd,0,nullptr,&p));
    zeEventPoolDestroy(p);
  }
  std::cout<<" PASS\n";
  
  stop=true; zeEventHostSignal(syncEvt); t.join();
  zeEventDestroy(syncEvt); zeEventPoolDestroy(pool);
  zeCommandListDestroy(cl); zeContextDestroy(ctx);
  return 0;
}
