
#include "ze_api.h"

ze_context_handle_t context;
ze_driver_handle_t driverHandle;
ze_device_handle_t device;
ze_command_queue_handle_t cmdQueue;
ze_command_list_handle_t cmdList;
ze_module_build_log_handle_t buildLog;
ze_module_handle_t module = nullptr;
ze_kernel_handle_t kernel = nullptr;

std::string resultToString(ze_result_t Status) {
  switch (Status) {
  case ZE_RESULT_SUCCESS:
    return "ZE_RESULT_SUCCESS";
  case ZE_RESULT_NOT_READY:
    return "ZE_RESULT_NOT_READY";
  case ZE_RESULT_ERROR_DEVICE_LOST:
    return "ZE_RESULT_ERROR_DEVICE_LOST";
  case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
    return "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
  case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
    return "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
  case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:
    return "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE";
  case ZE_RESULT_ERROR_MODULE_LINK_FAILURE:
    return "ZE_RESULT_ERROR_MODULE_LINK_FAILURE";
  case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS:
    return "ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS";
  case ZE_RESULT_ERROR_NOT_AVAILABLE:
    return "ZE_RESULT_ERROR_NOT_AVAILABLE";
  case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE:
    return "ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE";
  case ZE_RESULT_ERROR_UNINITIALIZED:
    return "ZE_RESULT_ERROR_UNINITIALIZED";
  case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
    return "ZE_RESULT_ERROR_UNSUPPORTED_VERSION";
  case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
    return "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
  case ZE_RESULT_ERROR_INVALID_ARGUMENT:
    return "ZE_RESULT_ERROR_INVALID_ARGUMENT";
  case ZE_RESULT_ERROR_INVALID_NULL_HANDLE:
    return "ZE_RESULT_ERROR_INVALID_NULL_HANDLE";
  case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
    return "ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE";
  case ZE_RESULT_ERROR_INVALID_NULL_POINTER:
    return "ZE_RESULT_ERROR_INVALID_NULL_POINTER";
  case ZE_RESULT_ERROR_INVALID_SIZE:
    return "ZE_RESULT_ERROR_INVALID_SIZE";
  case ZE_RESULT_ERROR_UNSUPPORTED_SIZE:
    return "ZE_RESULT_ERROR_UNSUPPORTED_SIZE";
  case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
    return "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT";
  case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
    return "ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT";
  case ZE_RESULT_ERROR_INVALID_ENUMERATION:
    return "ZE_RESULT_ERROR_INVALID_ENUMERATION";
  case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
    return "ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION";
  case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
    return "ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT";
  case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY:
    return "ZE_RESULT_ERROR_INVALID_NATIVE_BINARY";
  case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME:
    return "ZE_RESULT_ERROR_INVALID_GLOBAL_NAME";
  case ZE_RESULT_ERROR_INVALID_KERNEL_NAME:
    return "ZE_RESULT_ERROR_INVALID_KERNEL_NAME";
  case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME:
    return "ZE_RESULT_ERROR_INVALID_FUNCTION_NAME";
  case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
    return "ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION";
  case ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
    return "ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION";
  case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
    return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX";
  case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
    return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE";
  case ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
    return "ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE";
  case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED:
    return "ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED";
  case ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE:
    return "ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE";
  case ZE_RESULT_ERROR_OVERLAPPING_REGIONS:
    return "ZE_RESULT_ERROR_OVERLAPPING_REGIONS";
  case ZE_RESULT_ERROR_UNKNOWN:
    return "ZE_RESULT_ERROR_UNKNOWN";
  default:
    return "Unknown Error Code";
  }
}

#define ZE_CHECK(myZeCall)                                                     \
  if (myZeCall != ZE_RESULT_SUCCESS) {                                         \
    std::cout << "Error at " << #myZeCall << ": " << __FUNCTION__ << ": "      \
              << __LINE__ << std::endl;                                        \
    std::cout << "Exit with Error Code: "                                      \
              << "0x" << std::hex << myZeCall << std::dec << std::endl;        \
    std::cout << "Error Description: " << resultToString(myZeCall)             \
              << std::endl;                                                    \
    std::terminate();                                                          \
  }

void setupLevelZero() {
#ifdef IMMEDIATE
  std::cout << "Using immediate command list\n";
#else
  std::cout << "Using regular command list\n";
#endif
  // Initialization
  ZE_CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));

  // Get the driver
  uint32_t driverCount = 0;
  ZE_CHECK(zeDriverGet(&driverCount, nullptr));
  ZE_CHECK(zeDriverGet(&driverCount, &driverHandle));

  // Create the context
  ze_context_desc_t contextDescription = {};
  contextDescription.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  ZE_CHECK(zeContextCreate(driverHandle, &contextDescription, &context));

  // Get the device
  uint32_t deviceCount = 0;
  ZE_CHECK(zeDeviceGet(driverHandle, &deviceCount, nullptr));
  ZE_CHECK(zeDeviceGet(driverHandle, &deviceCount, &device));

  // Print basic properties of the device
  ze_device_properties_t deviceProperties = {};
  ZE_CHECK(zeDeviceGetProperties(device, &deviceProperties));
  std::cout << "Device   : " << deviceProperties.name << "\n"
            << "Type     : "
            << ((deviceProperties.type == ZE_DEVICE_TYPE_GPU) ? "GPU" : "FPGA")
            << "\n"
            << "Vendor ID: " << std::hex << deviceProperties.vendorId
            << std::dec << "\n";

  // Create a command queue
  uint32_t numQueueGroups = 0;
  ZE_CHECK(
      zeDeviceGetCommandQueueGroupProperties(device, &numQueueGroups, nullptr));
  if (numQueueGroups == 0) {
    std::cout << "No queue groups found\n";
    std::terminate();
  } else {
    std::cout << "#Queue Groups: " << numQueueGroups << std::endl;
  }
  std::vector<ze_command_queue_group_properties_t> queueProperties(
      numQueueGroups);
  ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(device, &numQueueGroups,
                                                  queueProperties.data()));

  ze_command_queue_desc_t cmdQueueDesc = {};
  for (uint32_t i = 0; i < numQueueGroups; i++) {
    if (queueProperties[i].flags &
        ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
      cmdQueueDesc.ordinal = i;
    }
  }

  cmdQueueDesc.index = 0;
  cmdQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  ZE_CHECK(zeCommandQueueCreate(context, device, &cmdQueueDesc, &cmdQueue));

  // Create a command list
  ze_command_list_desc_t cmdListDesc = {};
  cmdListDesc.commandQueueGroupOrdinal = cmdQueueDesc.ordinal;
#ifdef IMMEDIATE
  ZE_CHECK(
      zeCommandListCreateImmediate(context, device, &cmdQueueDesc, &cmdList));
#else
  ZE_CHECK(zeCommandListCreate(context, device, &cmdListDesc, &cmdList));
#endif
}

void cleanupLevelZero() {
    ZE_CHECK(zeCommandListDestroy(cmdList));
    ZE_CHECK(zeCommandQueueDestroy(cmdQueue));
    ZE_CHECK(zeContextDestroy(context));
}

void execCmdList(ze_command_list_handle_t cmdList) {
#ifndef IMMEDIATE
  // Close list abd submit for execution
  std::cout << "Closing Command List ...";
  ZE_CHECK(zeCommandListClose(cmdList));
  std::cout << " complete" << std::endl;
  ZE_CHECK(zeCommandQueueExecuteCommandLists(cmdQueue, 1, &cmdList, nullptr));
#endif
}

void compileKernel(std::string kernelFile, std::string kernelName) {
  // Module Initialization
  std::ifstream file("SlowKernel.spv", std::ios::binary);
  if (!file.is_open()) {
    std::cout << "binary file not found\n";
    std::terminate();
  }

  file.seekg(0, file.end);
  auto length = file.tellg();
  file.seekg(0, file.beg);

  std::unique_ptr<char[]> spirvInput(new char[length]);
  file.read(spirvInput.get(), length);
  file.close();

  ze_module_desc_t moduleDesc = {};

  moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  moduleDesc.pInputModule = reinterpret_cast<const uint8_t *>(spirvInput.get());
  moduleDesc.inputSize = length;
  moduleDesc.pBuildFlags = "";

  auto status =
      zeModuleCreate(context, device, &moduleDesc, &module, &buildLog);
  if (status != ZE_RESULT_SUCCESS) {
    // print log
    size_t szLog = 0;
    zeModuleBuildLogGetString(buildLog, &szLog, nullptr);

    char *stringLog = (char *)malloc(szLog);
    zeModuleBuildLogGetString(buildLog, &szLog, stringLog);
    std::cout << "zeModuleCreate failed: Build log: " << stringLog << std::endl;
    std::abort();
  }
  ZE_CHECK(zeModuleBuildLogDestroy(buildLog));

  ze_kernel_desc_t kernelDesc = {};
  kernelDesc.flags = ZE_KERNEL_FLAG_FORCE_RESIDENCY;
  kernelDesc.pKernelName = "myKernel";
  ZE_CHECK(zeKernelCreate(module, &kernelDesc, &kernel));
}

float timestampToMsKernel(uint64_t start, uint64_t stop) {
  // query device properties to get timer resolution
  ze_device_properties_t Props;
  ZE_CHECK(zeDeviceGetProperties(device, &Props));
  uint64_t TimerResolution = Props.timerResolution;
  uint32_t TimestampValidBits = Props.kernelTimestampValidBits;

  uint64_t T = ((stop - start) & (((uint64_t)1 << TimestampValidBits) - 1));
  T = T * TimerResolution;
  return T / 1000000.0;
}

float timestampToMs(uint64_t start, uint64_t stop) {
  // query device properties to get timer resolution
  ze_device_properties_t Props;
  ZE_CHECK(zeDeviceGetProperties(device, &Props));
  uint64_t TimerResolution = Props.timerResolution;
  uint32_t TimestampValidBits = Props.timestampValidBits;

  uint64_t T = ((stop - start) & (((uint64_t)1 << TimestampValidBits) - 1));
  T = T * TimerResolution;
  return T / 1000000.0;
}
