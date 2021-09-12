#ifndef HIPXX_BACKEND_LEVEL0_H
#define HIPXX_BACKEND_LEVEL0_H

#include "../src/common.hh"
#include "../../HIPxxBackend.hh"
#include "../include/ze_api.h"

enum class LZMemoryType : unsigned { Host = 0, Device = 1, Shared = 2 };

// const char* lzResultToString(ze_result_t status) {
//   switch (status) {
//     case ZE_RESULT_SUCCESS:
//       return "ZE_RESULT_SUCCESS";
//     case ZE_RESULT_NOT_READY:
//       return "ZE_RESULT_NOT_READY";
//     case ZE_RESULT_ERROR_DEVICE_LOST:
//       return "ZE_RESULT_ERROR_DEVICE_LOST";
//     case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
//       return "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
//     case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
//       return "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
//     case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:
//       return "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE";
//     case ZE_RESULT_ERROR_MODULE_LINK_FAILURE:
//       return "ZE_RESULT_ERROR_MODULE_LINK_FAILURE";
//     case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS:
//       return "ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS";
//     case ZE_RESULT_ERROR_NOT_AVAILABLE:
//       return "ZE_RESULT_ERROR_NOT_AVAILABLE";
//     case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE:
//       return "ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE";
//     case ZE_RESULT_ERROR_UNINITIALIZED:
//       return "ZE_RESULT_ERROR_UNINITIALIZED";
//     case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
//       return "ZE_RESULT_ERROR_UNSUPPORTED_VERSION";
//     case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
//       return "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
//     case ZE_RESULT_ERROR_INVALID_ARGUMENT:
//       return "ZE_RESULT_ERROR_INVALID_ARGUMENT";
//     case ZE_RESULT_ERROR_INVALID_NULL_HANDLE:
//       return "ZE_RESULT_ERROR_INVALID_NULL_HANDLE";
//     case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
//       return "ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE";
//     case ZE_RESULT_ERROR_INVALID_NULL_POINTER:
//       return "ZE_RESULT_ERROR_INVALID_NULL_POINTER";
//     case ZE_RESULT_ERROR_INVALID_SIZE:
//       return "ZE_RESULT_ERROR_INVALID_SIZE";
//     case ZE_RESULT_ERROR_UNSUPPORTED_SIZE:
//       return "ZE_RESULT_ERROR_UNSUPPORTED_SIZE";
//     case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
//       return "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT";
//     case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
//       return "ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT";
//     case ZE_RESULT_ERROR_INVALID_ENUMERATION:
//       return "ZE_RESULT_ERROR_INVALID_ENUMERATION";
//     case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
//       return "ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION";
//     case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
//       return "ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT";
//     case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY:
//       return "ZE_RESULT_ERROR_INVALID_NATIVE_BINARY";
//     case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME:
//       return "ZE_RESULT_ERROR_INVALID_GLOBAL_NAME";
//     case ZE_RESULT_ERROR_INVALID_KERNEL_NAME:
//       return "ZE_RESULT_ERROR_INVALID_KERNEL_NAME";
//     case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME:
//       return "ZE_RESULT_ERROR_INVALID_FUNCTION_NAME";
//     case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
//       return "ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION";
//     case ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
//       return "ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION";
//     case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
//       return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX";
//     case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
//       return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE";
//     case ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
//       return "ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE";
//     case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED:
//       return "ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED";
//     case ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE:
//       return "ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE";
//     case ZE_RESULT_ERROR_OVERLAPPING_REGIONS:
//       return "ZE_RESULT_ERROR_OVERLAPPING_REGIONS";
//     case ZE_RESULT_ERROR_UNKNOWN:
//       return "ZE_RESULT_ERROR_UNKNOWN";
//     default:
//       return "Unknown Error Code";
//   }
// }
// #define LZ_LOG_ERROR(msg, status) \
//   logError("{} ({}) in {}:{}:{}\n", msg, lzResultToString(status), __FILE__,
//   \
//            __LINE__, __func__)

// #define LZ_PROCESS_ERROR_MSG(msg, status)                               \
//   do {                                                                  \
//     if (status != ZE_RESULT_SUCCESS && status != ZE_RESULT_NOT_READY) { \
//       LZ_LOG_ERROR(msg, status);                                        \
//       throw status;                                                     \
//     }                                                                   \
//   } while (0)

// #define LZ_PROCESS_ERROR(status) \
//   LZ_PROCESS_ERROR_MSG("Level Zero Error", status)

// #define LZ_RETURN_ERROR_MSG(msg, status)                                \
//   do {                                                                  \
//     if (status != ZE_RESULT_SUCCESS && status != ZE_RESULT_NOT_READY) { \
//       LZ_LOG_ERROR(msg, status);                                        \
//       return lzConvertResult(status);                                   \
//     }                                                                   \
//   } while (0)

// #define HIP_LOG_ERROR(msg, status)                                          \
//   logError("{} ({}) in {}:{}:{}\n", msg, hipGetErrorName(status), __FILE__, \
//            __LINE__, __func__)

// #define HIP_PROCESS_ERROR_MSG(msg, status)                    \
//   do {                                                        \
//     if (status != hipSuccess && status != hipErrorNotReady) { \
//       HIP_LOG_ERROR(msg, status);                             \
//       throw status;                                           \
//     }                                                         \
//   } while (0)

#define HIP_PROCESS_ERROR(status) HIP_PROCESS_ERROR_MSG("HIP Error", status)

#define HIP_RETURN_ERROR(status)                            \
  HIP_RETURN_ERROR_MSG("HIP Error", status)                 \
  if (status != hipSuccess && status != hipErrorNotReady) { \
    HIP_LOG_ERROR(msg, status);                             \
    return status;                                          \
  }                                                         \
  }                                                         \
  while (0)
class HIPxxContextLevel0;
class HIPxxDeviceLevel0;

class HIPxxQueueLevel0 : public HIPxxQueue {
 protected:
  ze_command_queue_handle_t hCommandQueue;
  ze_context_handle_t ze_ctx;
  ze_device_handle_t ze_dev;
  HIPxxQueueLevel0(ze_context_handle_t _ze_ctx, ze_device_handle_t _ze_dev)
      : ze_ctx(_ze_ctx), ze_dev(_ze_dev) {
    logTrace(
        "HIPxxQueueLevel0 constructor called via ze_context_handle_t and "
        "ze_device_handle_t");
    // Discover all command queue groups
    uint32_t cmdqueueGroupCount = 0;
    zeDeviceGetCommandQueueGroupProperties(ze_dev, &cmdqueueGroupCount,
                                           nullptr);
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
    zeCommandQueueCreate(ze_ctx, ze_dev, &commandQueueDesc, &hCommandQueue);
  }

 public:
  HIPxxQueueLevel0(HIPxxContextLevel0* _hipxx_ctx,
                   HIPxxDeviceLevel0* _hipxx_dev);

  virtual hipError_t launch(HIPxxExecItem* exec_item) override {
    logWarn("HIPxxQueueLevel0.launch() not yet implemented");
    return hipSuccess;
  };

  virtual hipError_t memCopy(void* dst, const void* src, size_t size) override {
    logWarn("HIPxxQueueLevel0.memCopy() not yet implemented");
    return hipSuccess;
  };
};

class HIPxxDeviceLevel0 : public HIPxxDevice {
  ze_device_handle_t ze_device;

 public:
  HIPxxDeviceLevel0(ze_device_handle_t&& _ze_device) : ze_device(_ze_device) {}
  virtual void populate_device_properties() override {
    logWarn("HIPxxDeviceLevel0.populate_device_properties not yet implemented");
  }
  virtual std::string get_name() override { return device_name; }
  ze_device_handle_t get() { return ze_device; }
};

class HIPxxContextLevel0 : public HIPxxContext {
  ze_context_handle_t ze_ctx;
  OpenCLFunctionInfoMap FuncInfos;

 public:
  HIPxxContextLevel0(ze_context_handle_t&& _ze_ctx) : ze_ctx(_ze_ctx){};

  void* allocate(size_t size, size_t alignment, LZMemoryType memTy) {
    void* ptr = 0;
    if (memTy == LZMemoryType::Shared) {
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
      ze_device_handle_t ze_dev = ((HIPxxDeviceLevel0*)get_devices()[0])->get();
      ze_dev = nullptr;  // Do not associate allocation

      ze_result_t status = zeMemAllocShared(ze_ctx, &dmaDesc, &hmaDesc, size,
                                            alignment, ze_dev, &ptr);

      // LZ_PROCESS_ERROR_MSG(
      //     "HipLZ could not allocate shared memory with error code: ",
      //     status);
      logDebug("LZ MEMORY ALLOCATE via calling zeMemAllocShared {} ", status);

      return ptr;
    } else if (memTy == LZMemoryType::Device) {
      ze_device_mem_alloc_desc_t dmaDesc;
      dmaDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
      dmaDesc.pNext = NULL;
      dmaDesc.flags = 0;
      dmaDesc.ordinal = 0;

      // TODO Select proper device
      ze_device_handle_t ze_dev = ((HIPxxDeviceLevel0*)get_devices()[0])->get();

      ze_result_t status =
          zeMemAllocDevice(ze_ctx, &dmaDesc, size, alignment, ze_dev, &ptr);
      // LZ_PROCESS_ERROR_MSG(
      //     "HipLZ could not allocate device memory with error code: ",
      //     status);
      logDebug("LZ MEMORY ALLOCATE via calling zeMemAllocDevice {} ", status);

      return ptr;
    }

    // HIP_PROCESS_ERROR_MSG("HipLZ could not recognize allocation options",
    //                       hipErrorNotSupported);
  }

  virtual void* allocate(size_t size) override {
    return allocate(size, 0x1000, LZMemoryType::Device);
  }

  ze_context_handle_t get() { return ze_ctx; }
  virtual hipError_t memCopy(void* dst, const void* src, size_t size,
                             hipStream_t stream) override {
    logWarn("HIPxxContextLevel0.memCopy not yet implemented");
    return hipSuccess;
  };
  virtual bool register_function_as_kernel(std::string* module_str,
                                           const void* HostFunctionPtr,
                                           const char* FunctionName) override {
    logWarn("HIPxxContextLevel0.register_function_as_kernel not implemented");

    // logDebug("HIPxxContextLevel0.register_function_as_kernel {} ",
    //          FunctionName);
    // uint8_t* funcIL = (uint8_t*)module_str->data();
    // size_t ilSize = module_str->length();
    // std::string funcName = FunctionName;

    // // Parse the SPIR-V fat binary to retrieve kernel function information
    // size_t numWords = ilSize / 4;
    // int32_t* binarydata = new int32_t[numWords + 1];
    // std::memcpy(binarydata, funcIL, ilSize);
    // // Extract kernel function information
    // bool res = parseSPIR(binarydata, numWords, FuncInfos);
    // delete[] binarydata;
    // if (!res) {
    //   logError("SPIR-V parsing failed\n");
    //   return false;
    // }

    // logDebug("LZ PARSE SPIR {} ", funcName);
    // ze_module_handle_t ze_module;
    // // Create module with global address aware
    // std::string compilerOptions =
    //     " -cl-std=CL2.0 -cl-take-global-address -cl-match-sincospi";
    // ze_module_desc_t moduleDesc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
    //                                nullptr,
    //                                ZE_MODULE_FORMAT_IL_SPIRV,
    //                                ilSize,
    //                                funcIL,
    //                                compilerOptions.c_str(),
    //                                nullptr};
    // ze_result_t status = zeModuleCreate(ze_ctx,
    // GetDevice()->GetDeviceHandle(),
    //                                     &moduleDesc, &ze_module, nullptr);

    // logDebug("LZ CREATE MODULE via calling zeModuleCreate {} ", status);

    // // Create kernel
    // ze_kernel_desc_t kernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr,
    //                                0,  // flags
    //                                funcName.c_str()};
    // ze_kernel_handle_t hKernel;
    // ze_result_t status = zeKernelCreate(hModule, &kernelDesc, &hKernel);
    // // LZ_PROCESS_ERROR_MSG("HipLZ zeKernelCreate FAILED with return code ",
    // //  status);

    // logDebug("LZ KERNEL CREATION via calling zeKernelCreate {} ", status);
    return true;
  }
};  // HIPxxContextLevel0

class HIPxxBackendLevel0 : public HIPxxBackend {
 public:
  virtual void initialize(std::string HIPxxPlatformStr,
                          std::string HIPxxDeviceTypeStr,
                          std::string HIPxxDeviceStr) override {
    logDebug("HIPxxBackendLevel0 Initialize");
    ze_result_t status;
    status = zeInit(0);
    logDebug("INITIALIZE LEVEL-0 (via calling zeInit) {}\n", status);

    ze_device_type_t ze_device_type;
    if (!HIPxxDeviceTypeStr.compare("GPU")) {
      ze_device_type = ZE_DEVICE_TYPE_GPU;
    } else if (!HIPxxDeviceTypeStr.compare("FPGA")) {
      ze_device_type = ZE_DEVICE_TYPE_FPGA;
    } else {
      logCritical("HIPXX_DEVICE_TYPE must be either GPU or FPGA");
    }
    int platform_idx = std::atoi(HIPxxPlatformStr.c_str());
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
    ze_driver_handle_t ze_driver = ze_drivers[platform_idx];

    assert(ze_driver != nullptr);
    // Load devices to device vector
    zeDeviceGet(ze_driver, &deviceCount, nullptr);
    ze_devices.resize(deviceCount);
    zeDeviceGet(ze_driver, &deviceCount, ze_devices.data());

    const ze_context_desc_t ctxDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr,
                                       0};
    // Filter in only devices of selected type and add them to the backend as
    // derivates of HIPxxDevice
    for (int i = 0; i < deviceCount; i++) {
      auto dev = ze_devices[i];
      ze_device_properties_t device_properties;
      zeDeviceGetProperties(dev, &device_properties);
      if (ze_device_type == device_properties.type) {
        HIPxxDeviceLevel0* hipxx_l0_dev = new HIPxxDeviceLevel0(std::move(dev));
        Backend->add_device(hipxx_l0_dev);
      }
    }  // End adding HIPxxDevices

    ze_context_handle_t ze_ctx;
    zeContextCreateEx(ze_driver, &ctxDesc, deviceCount, ze_devices.data(),
                      &ze_ctx);
    HIPxxContextLevel0* hipxx_l0_ctx =
        new HIPxxContextLevel0(std::move(ze_ctx));

    // Associate devices with contexts and vice versa
    // TODO Make this more automatic via constructor calls
    for (auto dev : Backend->get_devices()) {
      hipxx_l0_ctx->add_device(dev);
      dev->add_context(hipxx_l0_ctx);

      Backend->add_queue(
          new HIPxxQueueLevel0(hipxx_l0_ctx, (HIPxxDeviceLevel0*)dev));
    }
    Backend->add_context(hipxx_l0_ctx);
  }

  virtual void initialize() override {
    std::string empty;
    initialize(empty, empty, empty);
  }

  void uninitialize() override {
    logTrace("HIPxxBackendLevel0 uninitializing");
    logWarn("HIPxxBackendLevel0->uninitialize() not implemented");
  }
};  // HIPxxBackendLevel0

#endif