// Example for dispatching a SPIR-V Kernel using Level Zero on the Intel HD
// Graphics Sample based on the test-suite exanples from Level-Zero:
//      https://github.com/intel/compute-runtime/blob/master/level_zero/core/test/black_box_tests/zello_world_gpu.cpp

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

#define IMMEDIATE
#include "common.hpp"
#include "ze_api.h"

int main(int argc, char **argv) {
  setupLevelZero();
  compileKernel("SlowKernel.spv", "myKernel");
  ze_result_t Status;

  ze_event_pool_handle_t EventPool_;
  ze_event_pool_handle_t EventPoolUser_;
  unsigned int PoolFlags =
      ZE_EVENT_POOL_FLAG_HOST_VISIBLE;

  ze_event_pool_desc_t EventPoolDesc = {
      ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, // stype
      nullptr,                           // pNext
      PoolFlags,                         // Flags
      1                                 // count
  };

  ZE_CHECK(zeEventPoolCreate(context, &EventPoolDesc, 0, nullptr, &EventPoolUser_));
  EventPoolDesc.count = 10; 
  ZE_CHECK(zeEventPoolCreate(context, &EventPoolDesc, 0, nullptr, &EventPool_));

  ze_event_desc_t EventDesc = {
      ZE_STRUCTURE_TYPE_EVENT_DESC, // stype
      nullptr,                      // pNext
      0,                            // index
      ZE_EVENT_SCOPE_FLAG_HOST,     // ensure memory/cache coherency required on
                                    // signal
      ZE_EVENT_SCOPE_FLAG_HOST      // ensure memory coherency across device and
                                    // Host after Event completes
  };

  ze_event_handle_t StartEvent, EndEvent, timestampRecordEventStart,
      timestampMemcopyEventStart, myEvent;
  ZE_CHECK(zeEventCreate(EventPoolUser_, &EventDesc, &timestampMemcopyEventStart));

  ZE_CHECK(zeEventCreate(EventPool_, &EventDesc, &StartEvent));
  EventDesc.index++;
  ZE_CHECK(zeEventCreate(EventPool_, &EventDesc, &EndEvent));
  EventDesc.index++;
  ZE_CHECK(zeEventCreate(EventPool_, &EventDesc, &timestampRecordEventStart));
  EventDesc.index++;
  ZE_CHECK(zeEventCreate(EventPool_, &EventDesc, &myEvent));

  ZE_CHECK(zeEventHostReset(timestampMemcopyEventStart));
  ZE_CHECK(zeEventHostReset(timestampMemcopyEventStart));

  ze_device_mem_alloc_desc_t deviceMemDesc = {
      ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
  deviceMemDesc.ordinal = 0;

  ze_host_mem_alloc_desc_t hostMemDesc = {
      ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
  hostMemDesc.pNext = nullptr;

  void *startTime = nullptr;
  void *endTime = nullptr;
  uint64_t startTimestamp, endTimestamp;
  zeMemAllocDevice(context, &deviceMemDesc, sizeof(uint64_t), 1, device,
                   &startTime);
  zeMemAllocDevice(context, &deviceMemDesc, sizeof(uint64_t), 1, device,
                   &endTime);


  // get the start time for the host
  auto start = std::chrono::steady_clock::now();

  zeCommandListAppendWriteGlobalTimestamp(
      cmdList, (uint64_t *)startTime, timestampRecordEventStart, 0, nullptr);
    zeCommandListAppendBarrier(cmdList, nullptr, 0, nullptr);
    zeCommandListAppendMemoryCopy(cmdList, &startTimestamp, startTime, sizeof(uint64_t),
                                  timestampMemcopyEventStart, 0, nullptr);
    zeCommandListAppendBarrier(cmdList, myEvent, 0, nullptr);
    Status = zeEventQueryStatus(timestampMemcopyEventStart);
    std::cout << "StartEvent Query: " << resultToString(Status) << std::endl;
  std::cout << "Launching Kernel" << std::endl;
  // Launch kernel on the GPU
  ze_group_count_t dispatch;
  dispatch.groupCountX = 1;
  dispatch.groupCountY = 1;
  dispatch.groupCountZ = 1;
  ze_kernel_indirect_access_flags_t flags = ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST | ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE;
  zeKernelSetIndirectAccess(kernel, flags);
  ZE_CHECK(zeCommandListAppendLaunchKernel(cmdList, kernel, &dispatch, EndEvent,
                                           1, &timestampRecordEventStart));
  execCmdList(cmdList);
  std::cout << "Kernel Launched" << std::endl;
  // query StartEvent, then Event, then StartEvent, then Event
  //   ZE_CHECK(
  //       zeEventHostSynchronize(StartEvent,
  //       std::numeric_limits<uint64_t>::max()));
    Status = zeEventQueryStatus(timestampMemcopyEventStart);
    std::cout << "StartEvent Query: " << resultToString(Status) << std::endl;
    Status = zeEventQueryStatus(timestampMemcopyEventStart);
    std::cout << "StartEvent Query: " << resultToString(Status) << std::endl;
  //   Status = zeEventQueryStatus(timestampRecordEventStop);
  //   std::cout << "EndEvent Query: " << resultToString(Status) << std::endl;
  //   Status = zeEventQueryStatus(StartEvent);
  //   std::cout << "StartEvent Query: " << resultToString(Status) << std::endl;
  //   Status = zeEventQueryStatus(timestampRecordEventStop);
  //   std::cout << "EndEvent Query: " << resultToString(Status) << std::endl;

 
  cleanupLevelZero();
  return 0;
}