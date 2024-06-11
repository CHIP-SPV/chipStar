#include <stdio.h>
#include <stdlib.h>
#include <level_zero/ze_api.h>
#include <iostream>
#include <string.h>

#define CHECK_ERROR(err) \
  if (err != ZE_RESULT_SUCCESS) { \
    fprintf(stderr, "Error: %d at line %d\n", err, __LINE__); \
    abort(); \
  }

void callbackFunction(ze_event_handle_t hEvent, void *user_data) {
  printf("callback complete\n");
}

int main(int argc, char *argv[]) {
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

  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " --immediate | --regular" << std::endl;
    return -1;
  }

  bool useImmediate = false;
  if (strcmp(argv[1], "--immediate") == 0) {
    useImmediate = true;
    std::cout << "Using immediate command list." << std::endl;
  } else if (strcmp(argv[1], "--regular") == 0) {
    useImmediate = false;
    std::cout << "Using regular command list." << std::endl;
  } else {
    std::cerr << "Invalid argument. Use --immediate or --regular." << std::endl;
    return -1;
  }

  std::cout << "Initializing Level Zero." << std::endl;
  err = zeInit(ZE_INIT_FLAG_GPU_ONLY);
  CHECK_ERROR(err);

  std::cout << "Discovering all driver instances." << std::endl;
  uint32_t driver_count = 0;
  err = zeDriverGet(&driver_count, NULL);
  CHECK_ERROR(err);

  ze_driver_handle_t *drivers = (ze_driver_handle_t *)malloc(driver_count * sizeof(ze_driver_handle_t));
  err = zeDriverGet(&driver_count, drivers);
  CHECK_ERROR(err);

  driver = drivers[0];
  free(drivers);
  std::cout << "First driver selected." << std::endl;

  std::cout << "Getting all devices." << std::endl;
  uint32_t device_count = 0;
  err = zeDeviceGet(driver, &device_count, NULL);
  CHECK_ERROR(err);

  ze_device_handle_t *devices = (ze_device_handle_t *)malloc(device_count * sizeof(ze_device_handle_t));
  err = zeDeviceGet(driver, &device_count, devices);
  CHECK_ERROR(err);

  device = devices[0];
  free(devices);
  std::cout << "First device selected." << std::endl;

  std::cout << "Creating context." << std::endl;
  ze_context_desc_t context_desc = { ZE_STRUCTURE_TYPE_CONTEXT_DESC, NULL, 0 };
  err = zeContextCreate(driver, &context_desc, &context);
  CHECK_ERROR(err);

  std::cout << "Creating command queue." << std::endl;
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

  std::cout << "Creating command list." << std::endl;
  ze_command_list_desc_t command_list_desc = {
    ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
    NULL,
    0,
    0
  };

  if (useImmediate) {
    ze_command_queue_desc_t immediate_queue_desc = {
      ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
      NULL,
      0,
      0,
      0,
      ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
      ZE_COMMAND_QUEUE_PRIORITY_NORMAL
    };
    err = zeCommandListCreateImmediate(context, device, &immediate_queue_desc, &command_list);
  } else {
    err = zeCommandListCreate(context, device, &command_list_desc, &command_list);
  }
  CHECK_ERROR(err);

  std::cout << "Allocating device memory." << std::endl;
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

  std::cout << "Loading SPIR-V binary." << std::endl;
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

  std::cout << "Creating and compiling module." << std::endl;
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

  std::cout << "Creating kernel." << std::endl;
  ze_kernel_desc_t kernel_desc = {
    ZE_STRUCTURE_TYPE_KERNEL_DESC,
    NULL,
    0,
    "simple_kernel"
  };
  err = zeKernelCreate(module, &kernel_desc, &kernel);
  CHECK_ERROR(err);

  std::cout << "Setting kernel arguments." << std::endl;
  err = zeKernelSetArgumentValue(kernel, 0, sizeof(ptr1), &ptr1);
  CHECK_ERROR(err);
  err = zeKernelSetArgumentValue(kernel, 1, sizeof(ptr2), &ptr2);
  CHECK_ERROR(err);
  size_t n = 10;
  err = zeKernelSetArgumentValue(kernel, 2, sizeof(size_t), &n);
  CHECK_ERROR(err);

  std::cout << "Creating events." << std::endl;
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

  std::cout << "Enqueuing barriers and kernel launch." << std::endl;
  err = zeCommandListAppendBarrier(command_list, barrier_event1, 1, &user_event1);
  CHECK_ERROR(err);

  ze_group_count_t dispatch = { 10, 1, 1 };
  err = zeCommandListAppendLaunchKernel(command_list, kernel, &dispatch, kernel_event, 1, &barrier_event1);
  CHECK_ERROR(err);

  if (!useImmediate) {
    std::cout << "Closing and executing command list." << std::endl;
    err = zeCommandListClose(command_list);
    CHECK_ERROR(err);
    err = zeCommandQueueExecuteCommandLists(command_queue, 1, &command_list, NULL);
    CHECK_ERROR(err);
  }

  std::cout << "Signaling user event 1." << std::endl;
  err = zeEventHostSignal(user_event1);
  CHECK_ERROR(err);

  std::cout << "Waiting for kernel event synchronization." << std::endl;
  err = zeEventHostSynchronize(kernel_event, UINT32_MAX);
  CHECK_ERROR(err);

  std::cout << "Freeing device memory." << std::endl;
  err = zeMemFree(context, ptr2);
  CHECK_ERROR(err);
  err = zeMemFree(context, ptr1);
  CHECK_ERROR(err);

  std::cout << "Cleaning up resources." << std::endl;
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
