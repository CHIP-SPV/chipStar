#include <stdio.h>
#include <stdlib.h>
#include <level_zero/ze_api.h>
#include <iostream>

#define CHECK_ERROR(err) \
  if (err != ZE_RESULT_SUCCESS) { \
    fprintf(stderr, "Error: %d at line %d\n", err, __LINE__); \
    abort(); \
  }

void callbackFunction(ze_event_handle_t hEvent, void *user_data) {
  printf("callback complete\n");
}

int main() {
  ze_result_t err;
  ze_context_handle_t context = NULL;
  ze_driver_handle_t driver = NULL;
  ze_device_handle_t device = NULL;
  ze_command_queue_handle_t command_queue = NULL;
  ze_command_list_handle_t command_list = NULL;
  ze_module_handle_t module = NULL;
  ze_kernel_handle_t kernel = NULL;
  ze_event_handle_t user_event1 = NULL;
  ze_event_handle_t user_event2 = NULL;
  ze_event_handle_t kernel_event = NULL;
  ze_event_handle_t barrier_event1 = NULL;
  ze_event_handle_t barrier_event2 = NULL;

  // Initialize Level Zero
  err = zeInit(ZE_INIT_FLAG_GPU_ONLY);
  CHECK_ERROR(err);

  // Discover all the driver instances
  uint32_t driver_count = 0;
  err = zeDriverGet(&driver_count, NULL);
  CHECK_ERROR(err);

  ze_driver_handle_t *drivers = (ze_driver_handle_t *)malloc(driver_count * sizeof(ze_driver_handle_t));
  err = zeDriverGet(&driver_count, drivers);
  CHECK_ERROR(err);

  // Find the first driver
  driver = drivers[0];
  free(drivers);

  // Get all devices
  uint32_t device_count = 0;
  err = zeDeviceGet(driver, &device_count, NULL);
  CHECK_ERROR(err);

  ze_device_handle_t *devices = (ze_device_handle_t *)malloc(device_count * sizeof(ze_device_handle_t));
  err = zeDeviceGet(driver, &device_count, devices);
  CHECK_ERROR(err);

  // Find the first device
  device = devices[0];
  free(devices);

  // Create context
  ze_context_desc_t context_desc = { ZE_STRUCTURE_TYPE_CONTEXT_DESC, NULL, 0 };
  err = zeContextCreate(driver, &context_desc, &context);
  CHECK_ERROR(err);

  // Create command queue
  ze_command_queue_desc_t command_queue_desc = {
    ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
    NULL,
    0,
    0,
    0,
    ZE_COMMAND_QUEUE_MODE_DEFAULT,
    ZE_COMMAND_QUEUE_PRIORITY_NORMAL
  };
  err = zeCommandQueueCreate(context, device, &command_queue_desc, &command_queue);
  CHECK_ERROR(err);

  // Create command list
  ze_command_list_desc_t command_list_desc = {
    ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
    NULL,
    0,
    0
  };
  err = zeCommandListCreate(context, device, &command_list_desc, &command_list);
  CHECK_ERROR(err);

  // Allocate device memory
  ze_device_mem_alloc_desc_t device_mem_alloc_desc = {
    ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
    NULL,
    0,
    0
  };

  void *ptr1;
  err = zeMemAllocDevice(context, &device_mem_alloc_desc, 10, 1, device, &ptr1);
  CHECK_ERROR(err);

  void *ptr2;
  err = zeMemAllocDevice(context, &device_mem_alloc_desc, 10, 1, device, &ptr2);
  CHECK_ERROR(err);

  // Load SPIR-V binary
  std::cout << "Loading SPIR-V binary" << std::endl;
  FILE *file = fopen("simple_kernel.spv", "rb");
  if (!file) {
    fprintf(stderr, "Failed to open SPIR-V binary\n");
    return -1;
  }
  fseek(file, 0, SEEK_END);
  size_t size = ftell(file);
  fseek(file, 0, SEEK_SET);
  uint8_t *spirv = (uint8_t *)malloc(size);
  fread(spirv, 1, size, file);
  fclose(file);

  // Create and compile module
  ze_module_desc_t module_desc = {
    ZE_STRUCTURE_TYPE_MODULE_DESC,
    NULL,
    ZE_MODULE_FORMAT_IL_SPIRV,
    size,
    spirv,
    NULL,
    NULL
  };

  err = zeModuleCreate(context, device, &module_desc, &module, NULL);
  CHECK_ERROR(err);
  free(spirv);

  // Create kernel
  ze_kernel_desc_t kernel_desc = {
    ZE_STRUCTURE_TYPE_KERNEL_DESC,
    NULL,
    0,
    "simple_kernel"
  };
  err = zeKernelCreate(module, &kernel_desc, &kernel);
  CHECK_ERROR(err);

  // Set kernel args
  err = zeKernelSetArgumentValue(kernel, 0, sizeof(ptr1), &ptr1);
  CHECK_ERROR(err);
  err = zeKernelSetArgumentValue(kernel, 1, sizeof(ptr2), &ptr2);
  CHECK_ERROR(err);
  size_t n = 10;
  err = zeKernelSetArgumentValue(kernel, 2, sizeof(size_t), &n);
  CHECK_ERROR(err);

  // Create events
  ze_event_pool_desc_t event_pool_desc = {
    ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
    NULL,
    ZE_EVENT_POOL_FLAG_HOST_VISIBLE,
    4
  };

  ze_event_pool_handle_t event_pool;
  err = zeEventPoolCreate(context, &event_pool_desc, 1, &device, &event_pool);
  CHECK_ERROR(err);

  ze_event_desc_t event_desc = {
    ZE_STRUCTURE_TYPE_EVENT_DESC,
    NULL,
    0,
    ZE_EVENT_SCOPE_FLAG_HOST,
    ZE_EVENT_SCOPE_FLAG_HOST
  };

  err = zeEventCreate(event_pool, &event_desc, &user_event1);
  CHECK_ERROR(err);
  event_desc.index = 1;
  err = zeEventCreate(event_pool, &event_desc, &user_event2);
  CHECK_ERROR(err);
  event_desc.index = 2;
  err = zeEventCreate(event_pool, &event_desc, &kernel_event);
  CHECK_ERROR(err);
  event_desc.index = 3;
  err = zeEventCreate(event_pool, &event_desc, &barrier_event1);
  CHECK_ERROR(err);
  err = zeEventCreate(event_pool, &event_desc, &barrier_event2);
  CHECK_ERROR(err);

  // Enqueue barriers
  err = zeCommandListAppendBarrier(command_list, barrier_event1, 1, &user_event1);
  CHECK_ERROR(err);

  // Enqueue kernel
  ze_group_count_t dispatch = { 10, 1, 1 };
  err = zeCommandListAppendLaunchKernel(command_list, kernel, &dispatch, kernel_event, 1, &barrier_event1);
  CHECK_ERROR(err)

//   err = zeCommandListAppendBarrier(command_list, barrier_event2, 1, &user_event2);
//   CHECK_ERROR(err);

  // Close and execute command list
  err = zeCommandListClose(command_list);
  CHECK_ERROR(err);
  err = zeCommandQueueExecuteCommandLists(command_queue, 1, &command_list, NULL);
  CHECK_ERROR(err);

  
  std::cout << "Signaling user event 1" << std::endl;
  err = zeEventHostSignal(user_event1);
  CHECK_ERROR(err);

  err = zeCommandQueueSynchronize(command_queue, UINT32_MAX);
  CHECK_ERROR(err);

  // Wait for events
//   err = zeEventHostSynchronize(kernel_event, UINT32_MAX);
//   CHECK_ERROR(err);
//   err = zeEventHostSynchronize(barrier_event1, UINT64_MAX);
//   if(err == ZE_RESULT_NOT_READY) {
//     std::cout << "Error: ZE_RESULT_NOT_READY" << err << std::endl;
//   }
//   CHECK_ERROR(err);
//   err = zeEventHostSynchronize(barrier_event2, UINT32_MAX);
//   CHECK_ERROR(err);

  // Free device memory
  err = zeMemFree(context, ptr2);
  CHECK_ERROR(err);
  err = zeMemFree(context, ptr1);
  CHECK_ERROR(err);

  // Cleanup
  zeEventDestroy(barrier_event2);
  zeEventDestroy(barrier_event1);
  zeEventDestroy(kernel_event);
  zeEventDestroy(user_event2);
  zeEventDestroy(user_event1);
  zeEventPoolDestroy(event_pool);
  zeKernelDestroy(kernel);
  zeModuleDestroy(module);
  zeCommandListDestroy(command_list);
  zeCommandQueueDestroy(command_queue);
  zeContextDestroy(context);

  return 0;
}
