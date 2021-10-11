#ifndef CHIP_BACKEND_LEVEL0_H
#define CHIP_BACKEND_LEVEL0_H

#include "../src/common.hh"
#include "../../CHIPBackend.hh"
#include "../include/ze_api.h"
enum class LZMemoryType : unsigned { Host = 0, Device = 1, Shared = 2 };

const char* lzResultToString(ze_result_t status);

#define LZ_LOG_ERROR(msg, status)                                            \
  logError("{} ({}) in {}:{}:{}\n", msg, lzResultToString(status), __FILE__, \
           __LINE__, __func__)

#define LZ_PROCESS_ERROR_MSG(msg, status)                               \
  do {                                                                  \
    if (status != ZE_RESULT_SUCCESS && status != ZE_RESULT_NOT_READY) { \
      LZ_LOG_ERROR(msg, status);                                        \
      throw status;                                                     \
    }                                                                   \
  } while (0)

#define LZ_PROCESS_ERROR(status) \
  LZ_PROCESS_ERROR_MSG("Level Zero Error", status)

#define LZ_RETURN_ERROR_MSG(msg, status)                                \
  do {                                                                  \
    if (status != ZE_RESULT_SUCCESS && status != ZE_RESULT_NOT_READY) { \
      LZ_LOG_ERROR(msg, status);                                        \
      return lzConvertResult(status);                                   \
    }                                                                   \
  } while (0)

#define HIP_LOG_ERROR(msg, status)                                          \
  logError("{} ({}) in {}:{}:{}\n", msg, hipGetErrorName(status), __FILE__, \
           __LINE__, __func__)

#define HIP_PROCESS_ERROR_MSG(msg, status)                    \
  do {                                                        \
    if (status != hipSuccess && status != hipErrorNotReady) { \
      HIP_LOG_ERROR(msg, status);                             \
      throw status;                                           \
    }                                                         \
  } while (0)

#define HIP_PROCESS_ERROR(status) HIP_PROCESS_ERROR_MSG("HIP Error", status)

#define HIP_RETURN_ERROR(status)                            \
  HIP_RETURN_ERROR_MSG("HIP Error", status)                 \
  if (status != hipSuccess && status != hipErrorNotReady) { \
    HIP_LOG_ERROR(msg, status);                             \
    return status;                                          \
  }                                                         \
  }                                                         \
  while (0)
class CHIPContextLevel0;
class CHIPDeviceLevel0;
class CHIPKernelLevel0 : public CHIPKernel {
 protected:
  ze_kernel_handle_t ze_kernel;

 public:
  CHIPKernelLevel0(){};
  CHIPKernelLevel0(ze_kernel_handle_t _ze_kernel, std::string _funcName,
                   const void* _host_ptr) {
    ze_kernel = _ze_kernel;
    host_f_name = _funcName;
    host_f_ptr = _host_ptr;
    logTrace("CHIPKernelLevel0 constructor via ze_kernel_handle");
  }

  ze_kernel_handle_t get() { return ze_kernel; }
};

class CHIPQueueLevel0 : public CHIPQueue {
 protected:
  ze_command_queue_handle_t ze_q;
  ze_context_handle_t ze_ctx;
  ze_device_handle_t ze_dev;

 public:
  CHIPQueueLevel0(CHIPDeviceLevel0* hixx_dev_);

  virtual hipError_t launch(CHIPExecItem* exec_item) override {
    logWarn("CHIPQueueLevel0.launch() not yet implemented");
    return hipSuccess;
  };

  ze_command_queue_handle_t get() { return ze_q; }

  virtual hipError_t memCopy(void* dst, const void* src, size_t size) override;
};

class CHIPDeviceLevel0 : public CHIPDevice {
  ze_device_handle_t ze_dev;
  ze_context_handle_t ze_ctx;

 public:
  CHIPDeviceLevel0(ze_device_handle_t&& ze_dev_, ze_context_handle_t ze_ctx_)
      : ze_dev(ze_dev_), ze_ctx(ze_ctx_) {}
  virtual void populateDeviceProperties_() override {
    logWarn(
        "CHIPDeviceLevel0.populate_device_properties not yet "
        "implemented");
  }
  virtual std::string getName() override { return device_name; }
  ze_device_handle_t& get() { return ze_dev; }

  virtual void reset() override;
};

class CHIPContextLevel0 : public CHIPContext {
  ze_context_handle_t ze_ctx;
  OpenCLFunctionInfoMap FuncInfos;

 public:
  ze_command_list_handle_t ze_cmd_list;
  ze_command_list_handle_t get_cmd_list() { return ze_cmd_list; }
  CHIPContextLevel0(ze_context_handle_t&& _ze_ctx) : ze_ctx(_ze_ctx) {}

  void* allocate_(size_t size, size_t alignment,
                  CHIPMemoryType memTy) override {
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

    // HIP_PROCESS_ERROR_MSG("HipLZ could not recognize allocation
    // options",
    //                       hipErrorNotSupported);
    return nullptr;
  }

  void free_(void* ptr) override{};  // TODO
  ze_context_handle_t& get() { return ze_ctx; }
  virtual hipError_t memCopy(void* dst, const void* src, size_t size,
                             hipStream_t stream) override;

  // virtual bool registerFunctionAsKernel(std::string* module_str,
  //                                       const void* HostFunctionPtr,
  //                                       const char* FunctionName) override {
  //   logWarn(
  //       "CHIPContextLevel0.register_function_as_kernel not "
  //       "implemented");
  //   logDebug("CHIPContextLevel0.register_function_as_kernel {} ",
  //            FunctionName);
  //   uint8_t* funcIL = (uint8_t*)module_str->data();
  //   size_t ilSize = module_str->length();
  //   std::string funcName = FunctionName;

  //   // Parse the SPIR-V fat binary to retrieve kernel function
  //   size_t numWords = ilSize / 4;
  //   int32_t* binarydata = new int32_t[numWords + 1];
  //   std::memcpy(binarydata, funcIL, ilSize);
  //   // Extract kernel function information
  //   bool res = parseSPIR(binarydata, numWords, FuncInfos);
  //   delete[] binarydata;
  //   if (!res) {
  //     logError("SPIR-V parsing failed\n");
  //     return false;
  //   }

  //   logDebug("LZ PARSE SPIR {} ", funcName);
  //   ze_module_handle_t ze_module;
  //   // Create module with global address aware
  //   std::string compilerOptions =
  //       " -cl-std=CL2.0 -cl-take-global-address -cl-match-sincospi";
  //   ze_module_desc_t moduleDesc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
  //                                  nullptr,
  //                                  ZE_MODULE_FORMAT_IL_SPIRV,
  //                                  ilSize,
  //                                  funcIL,
  //                                  compilerOptions.c_str(),
  //                                  nullptr};
  //   for (CHIPDevice* chip_dev : getDevices()) {
  //     ze_device_handle_t ze_dev = ((CHIPDeviceLevel0*)chip_dev)->get();
  //     ze_result_t status =
  //         zeModuleCreate(ze_ctx, ze_dev, &moduleDesc, &ze_module, nullptr);
  //     logDebug("LZ CREATE MODULE via calling zeModuleCreate {} ", status);

  //     // Create kernel
  //     ze_kernel_handle_t ze_kernel;
  //     ze_kernel_desc_t kernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr,
  //                                    0,  // flags
  //                                    funcName.c_str()};
  //     status = zeKernelCreate(ze_module, &kernelDesc, &ze_kernel);

  //     // LZ_PROCESS_ERROR_MSG("HipLZ zeKernelCreate FAILED with return
  //     // code ", status);

  //     logDebug("LZ KERNEL CREATION via calling zeKernelCreate {} ", status);
  //     CHIPKernelLevel0* chip_ze_kernel =
  //         new CHIPKernelLevel0(ze_kernel, FunctionName, HostFunctionPtr);

  //     chip_dev->addKernel(chip_ze_kernel);
  //   }

  //   return true;
  // }
};  // CHIPContextLevel0

class CHIPBackendLevel0 : public CHIPBackend {
 public:
  virtual void initialize_(std::string CHIPPlatformStr,
                           std::string CHIPDeviceTypeStr,
                           std::string CHIPDeviceStr) override {
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
    CHIPContextLevel0* chip_l0_ctx = new CHIPContextLevel0(std::move(ze_ctx));

    // Filter in only devices of selected type and add them to the
    // backend as derivates of CHIPDevice
    for (int i = 0; i < deviceCount; i++) {
      auto dev = ze_devices[i];
      ze_device_properties_t device_properties;
      zeDeviceGetProperties(dev, &device_properties);
      if (ze_device_type == device_properties.type) {
        CHIPDeviceLevel0* chip_l0_dev =
            new CHIPDeviceLevel0(std::move(dev), ze_ctx);
        CHIPQueueLevel0* q = new CHIPQueueLevel0(chip_l0_dev);
        chip_l0_dev->addQueue(q);
        Backend->addDevice(chip_l0_dev);
        // TODO
        break;  // For now don't add more than one device
      }
    }  // End adding CHIPDevices

    Backend->addContext(chip_l0_ctx);
  }

  void uninitialize() override {
    logTrace("CHIPBackendLevel0 uninitializing");
    logWarn("CHIPBackendLevel0->uninitialize() not implemented");
  }
};  // CHIPBackendLevel0

#endif