#ifndef HIPXX_BACKEND_LEVEL0_H
#define HIPXX_BACKEND_LEVEL0_H

#include "../src/common.hh"
#include "../../HIPxxBackend.hh"
#include "../include/ze_api.h"

class HIPxxQueueLevel0 : public HIPxxQueue {
 private:
  ze_command_queue_handle_t hCommandQueue;
  ze_context_handle_t ze_ctx;
  ze_device_handle_t ze_dev;

 public:
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

  virtual hipError_t launch(HIPxxExecItem* exec_item) override {
    logWarn("HIPxxQueueLevel0.launch() not yet implemented");
    return hipSuccess;
  };

  virtual hipError_t memCopy(void* dst, const void* src, size_t size) override {
    logWarn("HIPxxQueueLevel0.memCopy() not yet implemented");
    return hipSuccess;
  };
};
class HIPxxContextLevel0 : public HIPxxContext {
  ze_context_handle_t ze_ctx;
  OpenCLFunctionInfoMap FuncInfos;

 public:
  HIPxxContextLevel0(ze_context_handle_t&& _ze_ctx) : ze_ctx(_ze_ctx){};
  virtual void* allocate(size_t size) override {
    logWarn("HIPxxContextLevel0.allocate not yet implemented");
    return nullptr;
  };
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

class HIPxxDeviceLevel0 : public HIPxxDevice {
  ze_device_handle_t ze_device;

 public:
  HIPxxDeviceLevel0(ze_device_handle_t&& _ze_device) : ze_device(_ze_device) {}
  virtual void populate_device_properties() override {}
  virtual std::string get_name() override {
    // TODO
    return std::string();
  }
  ze_device_handle_t get() { return ze_device; }
};

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
    for (auto dev : Backend->get_devices()) {
      hipxx_l0_ctx->add_device(dev);
      dev->add_context(hipxx_l0_ctx);
      Backend->add_queue(new HIPxxQueueLevel0(
          hipxx_l0_ctx->get(), ((HIPxxDeviceLevel0*)dev)->get()));
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