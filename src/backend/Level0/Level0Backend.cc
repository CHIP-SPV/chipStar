#include "Level0Backend.hh"

// CHIPBackendLevelZero
// ***********************************************************************
void CHIPBackendLevel0::initialize_(std::string CHIPPlatformStr,
                                    std::string CHIPDeviceTypeStr,
                                    std::string CHIPDeviceStr) {
  logDebug("CHIPBackendLevel0 Initialize");
  ze_result_t status;
  status = zeInit(0);
  logDebug("INITIALIZE LEVEL-0 (via calling zeInit) {}\n", status);

  ze_device_type_t ze_device_type;
  if (!CHIPDeviceTypeStr.compare("GPU")) {
    ze_device_type = ZE_DEVICE_TYPE_GPU;
  } else if (!CHIPDeviceTypeStr.compare("FPGA")) {
    ze_device_type = ZE_DEVICE_TYPE_FPGA;
  } else {
    logCritical("CHIP_DEVICE_TYPE must be either GPU or FPGA");
  }
  int platform_idx = std::atoi(CHIPPlatformStr.c_str());
  std::vector<ze_driver_handle_t> ze_drivers;
  std::vector<ze_device_handle_t> ze_devices;

  // Get number of drivers
  uint32_t driverCount = 0, deviceCount = 0;
  status = zeDriverGet(&driverCount, nullptr);
  logDebug("Found Level0 Drivers: {}", driverCount);
  // Resize and fill ze_driver vector with drivers
  ze_drivers.resize(driverCount);
  status = zeDriverGet(&driverCount, ze_drivers.data());

  // TODO Allow for multilpe platforms(drivers)
  // TODO Check platform ID is not the same as OpenCL. You can have
  // two OCL platforms but only one level0 driver
  ze_driver_handle_t ze_driver = ze_drivers[platform_idx];

  assert(ze_driver != nullptr);
  // Load devices to device vector
  zeDeviceGet(ze_driver, &deviceCount, nullptr);
  ze_devices.resize(deviceCount);
  zeDeviceGet(ze_driver, &deviceCount, ze_devices.data());

  const ze_context_desc_t ctxDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr,
                                     0};

  ze_context_handle_t ze_ctx;
  zeContextCreateEx(ze_driver, &ctxDesc, deviceCount, ze_devices.data(),
                    &ze_ctx);
  CHIPContextLevel0* chip_l0_ctx = new CHIPContextLevel0(ze_ctx);
  Backend->addContext(chip_l0_ctx);

  // Filter in only devices of selected type and add them to the
  // backend as derivates of CHIPDevice
  for (int i = 0; i < deviceCount; i++) {
    auto dev = ze_devices[i];
    ze_device_properties_t device_properties;
    zeDeviceGetProperties(dev, &device_properties);
    if (ze_device_type == device_properties.type) {
      CHIPDeviceLevel0* chip_l0_dev =
          new CHIPDeviceLevel0(std::move(dev), chip_l0_ctx);
      chip_l0_ctx->addDevice(chip_l0_dev);

      CHIPQueueLevel0* q = new CHIPQueueLevel0(chip_l0_dev);
      chip_l0_dev->addQueue(q);
      Backend->addDevice(chip_l0_dev);
      break;  // For now don't add more than one device
    }
  }  // End adding CHIPDevices
}

// CHIPContextLevelZero
// ***********************************************************************
hipError_t CHIPContextLevel0::memCopy(void* dst, const void* src, size_t size,
                                      hipStream_t stream) {
  logTrace("CHIPContextLevel0.memCopy");
  // Stream halding done in Bindings.
  // if (stream == nullptr) {
  //   return getDefaultQueue()->memCopy(dst, src, size);
  // } else {
  //   logCritical("Queue lookup not yet implemented");
  //   std::abort();
  // }

  CHIPQueueLevel0* chip_q = (CHIPQueueLevel0*)stream;
  ze_result_t status = zeCommandQueueSynchronize(chip_q->get(), UINT64_MAX);
  if (status != ZE_RESULT_SUCCESS) {
    logCritical("Failed to memcopy");
    std::abort();
  }

  return hipSuccess;
}
void* CHIPContextLevel0::allocate_(size_t size, size_t alignment,
                                   CHIPMemoryType memTy) {
  alignment = 0x1000;  // TODO Where/why
  void* ptr = 0;
  if (memTy == CHIPMemoryType::Shared) {
    ze_device_mem_alloc_desc_t dmaDesc;
    dmaDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    dmaDesc.pNext = NULL;
    dmaDesc.flags = 0;
    dmaDesc.ordinal = 0;
    ze_host_mem_alloc_desc_t hmaDesc;
    hmaDesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    hmaDesc.pNext = NULL;
    hmaDesc.flags = 0;

    // TODO Check if devices support cross-device sharing?
    ze_device_handle_t ze_dev = ((CHIPDeviceLevel0*)getDevices()[0])->get();
    ze_dev = nullptr;  // Do not associate allocation

    ze_result_t status = zeMemAllocShared(ze_ctx, &dmaDesc, &hmaDesc, size,
                                          alignment, ze_dev, &ptr);

    // LZ_PROCESS_ERROR_MSG(
    //     "HipLZ could not allocate shared memory with error code:
    //     ", status);
    logDebug("LZ MEMORY ALLOCATE via calling zeMemAllocShared {} ", status);

    return ptr;
  } else if (memTy == CHIPMemoryType::Device) {
    ze_device_mem_alloc_desc_t dmaDesc;
    dmaDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    dmaDesc.pNext = NULL;
    dmaDesc.flags = 0;
    dmaDesc.ordinal = 0;

    // TODO Select proper device
    ze_device_handle_t ze_dev = ((CHIPDeviceLevel0*)getDevices()[0])->get();

    ze_result_t status =
        zeMemAllocDevice(ze_ctx, &dmaDesc, size, alignment, ze_dev, &ptr);
    LZ_PROCESS_ERROR_MSG(
        "HipLZ could not allocate device memory with error code: ", status);
    logDebug("LZ MEMORY ALLOCATE via calling zeMemAllocDevice {} ", status);

    return ptr;
  }
}
// CHIPDeviceLevelZero
// ***********************************************************************
CHIPDeviceLevel0::CHIPDeviceLevel0(ze_device_handle_t&& ze_dev_,
                                   CHIPContextLevel0* chip_ctx_)
    : CHIPDevice(chip_ctx_), ze_dev(ze_dev_), ze_ctx(chip_ctx_->get()) {
  assert(ctx != nullptr);
}

void CHIPDeviceLevel0::reset() {
  logCritical("CHIPDeviceLevel0::reset() not yet implemented");
  std::abort();
}
// CHIPQueueLevelZero
// ***********************************************************************
CHIPQueueLevel0::CHIPQueueLevel0(CHIPDeviceLevel0* chip_dev_)
    : CHIPQueue(chip_dev_) {
  // CHIPContextLevel0* chip_context_lz = (CHIPContextLevel0*)chip_context;
  auto ctx = chip_dev_->getContext();
  auto chip_context_lz = (CHIPContextLevel0*)ctx;

  ze_ctx = chip_context_lz->get();
  ze_dev = chip_dev_->get();

  logTrace(
      "CHIPQueueLevel0 constructor called via CHIPContextLevel0 and "
      "CHIPDeviceLevel0");

  // Discover all command queue groups
  uint32_t cmdqueueGroupCount = 0;
  zeDeviceGetCommandQueueGroupProperties(ze_dev, &cmdqueueGroupCount, nullptr);
  logDebug("CommandGroups found: {}", cmdqueueGroupCount);

  ze_command_queue_group_properties_t* cmdqueueGroupProperties =
      (ze_command_queue_group_properties_t*)malloc(
          cmdqueueGroupCount * sizeof(ze_command_queue_group_properties_t));
  zeDeviceGetCommandQueueGroupProperties(ze_dev, &cmdqueueGroupCount,
                                         cmdqueueGroupProperties);

  // Find a command queue type that support compute
  uint32_t computeQueueGroupOrdinal = cmdqueueGroupCount;
  for (uint32_t i = 0; i < cmdqueueGroupCount; ++i) {
    if (cmdqueueGroupProperties[i].flags &
        ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
      computeQueueGroupOrdinal = i;
      logDebug("Found compute command group");
      break;
    }
  }

  ze_command_queue_desc_t commandQueueDesc = {
      ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
      nullptr,
      computeQueueGroupOrdinal,
      0,  // index
      0,  // flags
      ZE_COMMAND_QUEUE_MODE_DEFAULT,
      ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  zeCommandQueueCreate(ze_ctx, ze_dev, &commandQueueDesc, &ze_q);

  ze_command_list_desc_t clDesc;
  clDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
  clDesc.flags = ZE_COMMAND_LIST_FLAG_EXPLICIT_ONLY;  // default hehaviour
  clDesc.commandQueueGroupOrdinal = 0;
  clDesc.pNext = nullptr;
  // TODO more devices support
  ze_result_t status = zeCommandListCreate(ze_ctx, ze_dev, &clDesc,
                                           &(chip_context_lz->ze_cmd_list));
  if (((CHIPContextLevel0*)chip_context)->get_cmd_list() == nullptr) {
    logCritical("Failed to initialize ze_cmd_list");
    std::abort();
  }

  // LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListCreate FAILED with return code
  // ",
  //                      status);
  logDebug("LZ COMMAND LIST CREATION via calling zeCommandListCreate {} ",
           status);

  chip_context->addQueue(this);
}
hipError_t CHIPQueueLevel0::memCopy(void* dst, const void* src, size_t size) {
  // ze_event_handle_t hSignalEvent =
  // GetSignalEvent(ze_q)->GetEventHandler();
  ze_command_list_handle_t ze_cmd_list =
      ((CHIPContextLevel0*)chip_context)->ze_cmd_list;
  assert(ze_cmd_list != nullptr);
  ze_result_t status = zeCommandListAppendMemoryCopy(ze_cmd_list, dst, src,
                                                     size, nullptr, 0, NULL);
  return hipSuccess;
}

// CHIPKernelLevelZero
// ***********************************************************************

const char* lzResultToString(ze_result_t status) {
  switch (status) {
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
