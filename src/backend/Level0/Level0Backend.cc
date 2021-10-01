#include "Level0Backend.hh"

HIPxxQueueLevel0::HIPxxQueueLevel0(HIPxxContextLevel0* _hipxx_ctx,
                                   HIPxxDeviceLevel0* _hipxx_dev) {
  hipxx_device = _hipxx_dev;
  hipxx_context = _hipxx_ctx;

  ze_ctx = _hipxx_ctx->get();
  ze_dev = _hipxx_dev->get();

  logTrace(
      "HIPxxQueueLevel0 constructor called via HIPxxContextLevel0 and "
      "HIPxxDeviceLevel0");

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
  ze_result_t status =
      zeCommandListCreate(ze_ctx, ze_dev, &clDesc, &(_hipxx_ctx->ze_cmd_list));
  if (((HIPxxContextLevel0*)hipxx_context)->get_cmd_list() == nullptr) {
    logCritical("Failed to initialize ze_cmd_list");
    std::abort();
  }

  // LZ_PROCESS_ERROR_MSG("HipLZ zeCommandListCreate FAILED with return code
  // ",
  //                      status);
  logDebug("LZ COMMAND LIST CREATION via calling zeCommandListCreate {} ",
           status);

  hipxx_context->add_queue(this);
}

hipError_t HIPxxContextLevel0::memCopy(void* dst, const void* src, size_t size,
                                       hipStream_t stream) {
  logTrace("HIPxxContextLevel0.memCopy");
  if (stream == nullptr) {
    get_default_queue()->memCopy(dst, src, size);
  } else {
    logCritical("Queue lookup not yet implemented");
    std::abort();
  }

  HIPxxQueueLevel0* hipxx_q = (HIPxxQueueLevel0*)get_default_queue();
  ze_result_t status = zeCommandQueueSynchronize(hipxx_q->get(), UINT64_MAX);
  if (status != ZE_RESULT_SUCCESS) {
    logCritical("Failed to memcopy");
    std::abort();
  }

  return hipSuccess;
}

hipError_t HIPxxQueueLevel0::memCopy(void* dst, const void* src, size_t size) {
  // ze_event_handle_t hSignalEvent = GetSignalEvent(ze_q)->GetEventHandler();
  ze_command_list_handle_t ze_cmd_list =
      ((HIPxxContextLevel0*)hipxx_context)->ze_cmd_list;
  assert(ze_cmd_list != nullptr);
  ze_result_t status = zeCommandListAppendMemoryCopy(ze_cmd_list, dst, src,
                                                     size, nullptr, 0, NULL);
  return hipSuccess;
}

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