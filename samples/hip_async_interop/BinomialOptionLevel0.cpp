
#include "kernel.h"

#include <ze_api.h>

#include <cassert>
#include <iostream>

#define MAX_EVENTS 2048

extern "C" {
  void* runLevel0Kernel(void *NativeEventDep, uintptr_t *NativeHandles, int NumHandles, unsigned Blocks, unsigned Threads, unsigned Arg1, void *Arg2, void *Arg3);
}

static ze_kernel_handle_t Kernel = 0;
static ze_module_handle_t Program = 0;
static ze_event_pool_handle_t EventPool = 0;
static uint32_t LastEventIndex = 0;

static ze_command_list_handle_t getCmdListForCmdQueue(ze_context_handle_t Context, ze_device_handle_t Device) {
  // Discover all command queue groups
  ze_result_t Err = ZE_RESULT_SUCCESS;
  uint32_t CmdQueueGroupCount = 0;
  Err = zeDeviceGetCommandQueueGroupProperties(Device, &CmdQueueGroupCount, nullptr);
  assert(Err == ZE_RESULT_SUCCESS);

  ze_command_queue_group_properties_t* CmdQueueGroupProperties = new ze_command_queue_group_properties_t [CmdQueueGroupCount];
  Err = zeDeviceGetCommandQueueGroupProperties(Device, &CmdQueueGroupCount, CmdQueueGroupProperties);
  assert(Err == ZE_RESULT_SUCCESS);

  // Find a command queue type that support compute
  uint32_t ComputeQueueGroupOrdinal = CmdQueueGroupCount;
  for(uint32_t i = 0; i < CmdQueueGroupCount; ++i ) {
    if(CmdQueueGroupProperties[ i ].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE ) {
      ComputeQueueGroupOrdinal = i;
      break;
    }
  }
  delete [] CmdQueueGroupProperties;

  // Create a command list
  ze_command_list_desc_t CommandListDesc = {
    ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
    nullptr,
    ComputeQueueGroupOrdinal,
    0 // flags
  };
  ze_command_list_handle_t CommandList;
  Err = zeCommandListCreate(Context, Device, &CommandListDesc, &CommandList);
  assert(Err == ZE_RESULT_SUCCESS);

  return CommandList;
}

void* runLevel0Kernel(void *NativeEventDep, uintptr_t *NativeHandles, int NumHandles, unsigned Blocks, unsigned Threads, unsigned Arg1, void *Arg2, void *Arg3) {
  assert (NumHandles == 4);
  ze_result_t Err = ZE_RESULT_SUCCESS;
  //ze_driver_handle_t Driv = (ze_driver_handle_t)NativeHandles[0];
  ze_device_handle_t Dev = (ze_device_handle_t)NativeHandles[1];
  ze_context_handle_t Ctx = (ze_context_handle_t)NativeHandles[2];
  ze_command_queue_handle_t CQ = (ze_command_queue_handle_t)NativeHandles[3];

  ze_command_list_handle_t CommandList = getCmdListForCmdQueue(Ctx, Dev);

  ze_event_handle_t DepEv = (ze_event_handle_t)NativeEventDep;


  if (Program == 0) {
    std::cout << "Level0: Building program\n";
    ze_module_desc_t ModuleDesc = {
      ZE_STRUCTURE_TYPE_MODULE_DESC,
      nullptr,
      ZE_MODULE_FORMAT_IL_SPIRV,
      KernelSpirVLength,
      KernelSpirV,
      nullptr,
      nullptr
    };
    ze_module_build_log_handle_t Buildlog = nullptr;
    Err = zeModuleCreate(Ctx, Dev, &ModuleDesc, &Program, &Buildlog);

    // Only save build logs for module creation errors.
    if (Err != ZE_RESULT_SUCCESS) {
      std::cout << "Level0: Build failed.\n";
      size_t LogSize = 0;
      zeModuleBuildLogGetString(Buildlog, &LogSize, nullptr);
      char* BuildLog = new char[LogSize];
      zeModuleBuildLogGetString(Buildlog, &LogSize, BuildLog);
      std::cout << "Level0: Build log:\n" << BuildLog << "\n";
      zeModuleBuildLogDestroy(Buildlog);
      delete [] BuildLog;
      return NULL;
    }

    assert(Program);

    uint32_t KernelCount = 0;
    Err = zeModuleGetKernelNames(Program, &KernelCount, nullptr);
    assert(Err == ZE_RESULT_SUCCESS);
    std::cout << "Level0: Found " << KernelCount << " kernels in this module." << std::endl;

    const char *KernelNames[KernelCount];
    Err = zeModuleGetKernelNames(Program, &KernelCount, KernelNames);
    assert(Err == ZE_RESULT_SUCCESS);
    for (uint32_t i = 0; i < KernelCount; ++i) {
      std::cout << "Level0: kernel " << i << " : " << KernelNames[i] << std::endl;
    }

    ze_kernel_desc_t KernelDesc = {
      ZE_STRUCTURE_TYPE_KERNEL_DESC,
      nullptr,
      0, // flags
     "binomial_options_level0.1"
    };
    Err = zeKernelCreate(Program, &KernelDesc, &Kernel);
    assert(Err == ZE_RESULT_SUCCESS);
    assert(Kernel);

    zeKernelSetGroupSize(Kernel, Threads, 1, 1);

    Err = zeKernelSetArgumentValue(Kernel, 0, sizeof(int), &Arg1);
    assert(Err == ZE_RESULT_SUCCESS);
    Err = zeKernelSetArgumentValue(Kernel, 1, sizeof(void*), &Arg2);
    assert(Err == ZE_RESULT_SUCCESS);
    Err = zeKernelSetArgumentValue(Kernel, 2, sizeof(void*), &Arg3);
    assert(Err == ZE_RESULT_SUCCESS);
  }

  if (EventPool == 0) {
    // Create event pool TODO this is never released
    ze_event_pool_desc_t EventPoolDesc = {
      ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
      nullptr,
      ZE_EVENT_POOL_FLAG_HOST_VISIBLE, // all events in pool are visible to Host
      MAX_EVENTS // count
    };
    Err = zeEventPoolCreate(Ctx, &EventPoolDesc, 0, nullptr, &EventPool);
    assert(Err == ZE_RESULT_SUCCESS);
  }

  // create event to return
  ze_event_handle_t RetEvent = 0;
  ze_event_desc_t EventDesc = {
    ZE_STRUCTURE_TYPE_EVENT_DESC,
    nullptr,
    LastEventIndex++, // index
    ZE_EVENT_SCOPE_FLAG_HOST, // memory/cache coherency required on signal
    ZE_EVENT_SCOPE_FLAG_HOST  // ensure memory coherency across device and Host after event completes
  };
  Err = zeEventCreate(EventPool, &EventDesc, &RetEvent);
  assert(Err == ZE_RESULT_SUCCESS);

  ze_group_count_t LaunchArgs = { Blocks*Threads, 1, 1 };
  // Append launch kernel
  Err = zeCommandListAppendLaunchKernel(CommandList, Kernel, &LaunchArgs, RetEvent, 1, &DepEv);
  assert(Err == ZE_RESULT_SUCCESS);

  Err = zeCommandListClose(CommandList);
  assert(Err == ZE_RESULT_SUCCESS);

  Err = zeCommandQueueExecuteCommandLists(CQ, 1, &CommandList, nullptr);
  assert(Err == ZE_RESULT_SUCCESS);

  assert(RetEvent != NULL);
  return (void*)RetEvent;
}
