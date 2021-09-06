/**
 * @file Backend.hh
 * @author Paulius Velesko (pvelesko@gmail.com)
 * @brief OpenCL backend for HIPxx. HIPxxBackendOpenCL class definition with
 * inheritance from HIPxxBackend. Subsequent virtual function overrides.
 * @version 0.1
 * @date 2021-08-19
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef HIPXX_BACKEND_OPENCL_H
#define HIPXX_BACKEND_OPENCL_H

#define CL_TARGET_OPENCL_VERSION 210
#define CL_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 210
#define CL_HPP_MINIMUM_OPENCL_VERSION 200

#include <CL/cl_ext_intel.h>

#include <CL/opencl.hpp>

#include "../../HIPxxBackend.hh"
#include "exceptions.hh"
#include "common.hh"

class HIPxxContextOpenCL;
class HIPxxDeviceOpenCL;
class HIPxxExecItemOpenCL;
class HIPxxKernelOpenCL;
class HIPxxQueueOpenCL;

class HIPxxContextOpenCL : public HIPxxContext {
 public:
  cl::Context *cl_ctx;
  HIPxxContextOpenCL(cl::Context *ctx_in);

  void *allocate(size_t size) override;
};

class HIPxxDeviceOpenCL : public HIPxxDevice {
 public:
  cl::Device *cl_dev;
  cl::Context *cl_ctx;
  HIPxxDeviceOpenCL(HIPxxContextOpenCL *hipxx_ctx, cl::Device *dev_in, int idx);

  virtual void populate_device_properties() override;
  virtual std::string get_name() override;
};

class HIPxxQueueOpenCL : public HIPxxQueue {
 protected:
  // Any reason to make these private/protected?
  cl::Context *cl_ctx;
  cl::Device *cl_dev;
  cl::CommandQueue *cl_q;

 public:
  HIPxxQueueOpenCL() = delete;  // delete default constructor
  HIPxxQueueOpenCL(const HIPxxQueueOpenCL &) = delete;
  HIPxxQueueOpenCL(HIPxxContextOpenCL *_ctx, HIPxxDeviceOpenCL *_dev);
  ~HIPxxQueueOpenCL();

  virtual hipError_t launch(HIPxxKernel *kernel,
                            HIPxxExecItem *exec_item) override;
};

class HIPxxKernelOpenCL : public HIPxxKernel {
 public:
  cl::Kernel ocl_kernel;

  OCLFuncInfo *get_func_info() const;
  cl::Kernel get() const { return ocl_kernel; }
};

class HIPxxExecItemOpenCL : public HIPxxExecItem {
 public:
  cl::Kernel *kernel;
  virtual hipError_t launch(HIPxxKernel *hipxx_kernel) override {
    logTrace("HIPxxExecItemOpenCL->launch()");
    HIPxxQueueOpenCL *ocl_q = (HIPxxQueueOpenCL *)q;
    return (hipError_t)(ocl_q->launch(Kernel, this) == hipSuccess);
  };

  int setup_all_args(HIPxxKernelOpenCL *kernel) {
    OCLFuncInfo *FuncInfo = kernel->get_func_info();
    size_t NumLocals = 0;
    for (size_t i = 0; i < FuncInfo->ArgTypeInfo.size(); ++i) {
      if (FuncInfo->ArgTypeInfo[i].space == OCLSpace::Local) ++NumLocals;
    }
    // there can only be one dynamic shared mem variable, per cuda spec
    assert(NumLocals <= 1);

    if ((OffsetsSizes.size() + NumLocals) != FuncInfo->ArgTypeInfo.size()) {
      logError("Some arguments are still unset\n");
      return CL_INVALID_VALUE;
    }

    if (OffsetsSizes.size() == 0) return CL_SUCCESS;

    std::sort(OffsetsSizes.begin(), OffsetsSizes.end());
    if ((std::get<0>(OffsetsSizes[0]) != 0) ||
        (std::get<1>(OffsetsSizes[0]) == 0)) {
      logError("Invalid offset/size\n");
      return CL_INVALID_VALUE;
    }

    // check args are set
    if (OffsetsSizes.size() > 1) {
      for (size_t i = 1; i < OffsetsSizes.size(); ++i) {
        if ((std::get<0>(OffsetsSizes[i]) == 0) ||
            (std::get<1>(OffsetsSizes[i]) == 0) ||
            ((std::get<0>(OffsetsSizes[i - 1]) +
              std::get<1>(OffsetsSizes[i - 1])) >
             std::get<0>(OffsetsSizes[i]))) {
          logError("Invalid offset/size\n");
          return CL_INVALID_VALUE;
        }
      }
    }

    const unsigned char *start = ArgData.data();
    void *p;
    int err;
    for (cl_uint i = 0; i < OffsetsSizes.size(); ++i) {
      OCLArgTypeInfo &ai = FuncInfo->ArgTypeInfo[i];
      logDebug("ARG {}: OS[0]: {} OS[1]: {} \n      TYPE {} SPAC {} SIZE {}\n",
               i, std::get<0>(OffsetsSizes[i]), std::get<1>(OffsetsSizes[i]),
               (unsigned)ai.type, (unsigned)ai.space, ai.size);

      if (ai.type == OCLType::Pointer) {
        // TODO other than global AS ?
        assert(ai.size == sizeof(void *));
        assert(std::get<1>(OffsetsSizes[i]) == ai.size);
        p = *(void **)(start + std::get<0>(OffsetsSizes[i]));
        logDebug("setArg SVM {} to {}\n", i, p);
        err = ::clSetKernelArgSVMPointer(kernel->get().get(), i, p);
        if (err != CL_SUCCESS) {
          logDebug("clSetKernelArgSVMPointer failed with error {}\n", err);
          return err;
        }
      } else {
        size_t size = std::get<1>(OffsetsSizes[i]);
        size_t offs = std::get<0>(OffsetsSizes[i]);
        void *value = (void *)(start + offs);
        logDebug("setArg {} size {} offs {}\n", i, size, offs);
        err = ::clSetKernelArg(kernel->get().get(), i, size, value);
        if (err != CL_SUCCESS) {
          logDebug("clSetKernelArg failed with error {}\n", err);
          return err;
        }
      }
    }

    return setLocalSize(SharedMem, FuncInfo, kernel->get().get());
  }
};

class HIPxxBackendOpenCL : public HIPxxBackend {
 public:
  void initialize(std::string HIPxxPlatformStr, std::string HIPxxDeviceTypeStr,
                  std::string HIPxxDeviceStr) override {
    logDebug("HIPxxBackendOpenCL Initialize");
    std::vector<cl::Platform> Platforms;
    cl_int err = cl::Platform::get(&Platforms);
    if (err != CL_SUCCESS) {
      logCritical("Failed to get OpenCL platforms! {}", err);
      std::abort();
    }
    std::cout << "\nFound " << Platforms.size() << " OpenCL platforms:\n";
    for (int i = 0; i < Platforms.size(); i++) {
      std::cout << i << ". " << Platforms[i].getInfo<CL_PLATFORM_NAME>()
                << "\n";
    }

    std::vector<cl::Device> enabled_devices;
    std::vector<cl::Device> Devices;
    int selected_platform;
    int selected_device;
    cl_bitfield selected_dev_type = 0;

    try {
      if (!HIPxxDeviceStr.compare("all")) {  // Use all devices that match type
        selected_device = -1;
      } else {
        selected_device = std::stoi(HIPxxDeviceStr);
      }

      // Platform index in range?
      selected_platform = std::stoi(HIPxxPlatformStr);
      if ((selected_platform < 0) || (selected_platform >= Platforms.size()))
        throw InvalidPlatformOrDeviceNumber(
            "HIPXX_PLATFORM: platform number out of range");
      std::cout << "Selected Platform: " << selected_platform << ". "
                << Platforms[selected_platform].getInfo<CL_PLATFORM_NAME>()
                << "\n";

      // Device  index in range?
      err =  // Get All devices and print
          Platforms[selected_platform].getDevices(CL_DEVICE_TYPE_ALL, &Devices);
      for (int i = 0; i < Devices.size(); i++) {
        std::cout << i << ". " << Devices[i].getInfo<CL_DEVICE_NAME>() << "\n";
      }
      if (selected_device >= Devices.size())
        throw InvalidPlatformOrDeviceNumber(
            "HIPXX_DEVICE: device number out of range");
      if (selected_device == -1) {  // All devices enabled
        enabled_devices = Devices;
        logDebug("All Devices enabled\n", "");
      } else {
        enabled_devices.push_back(Devices[selected_device]);
        std::cout << "\nEnabled Devices:\n";
        std::cout << selected_device << ". "
                  << enabled_devices[0].getInfo<CL_DEVICE_NAME>() << "\n";
      }

      if (err != CL_SUCCESS)
        throw InvalidPlatformOrDeviceNumber(
            "HIPXX_DEVICE: can't get devices for platform");

      if (HIPxxDeviceTypeStr == "all")
        selected_dev_type = CL_DEVICE_TYPE_ALL;
      else if (HIPxxDeviceTypeStr == "cpu")
        selected_dev_type = CL_DEVICE_TYPE_CPU;
      else if (HIPxxDeviceTypeStr == "gpu")
        selected_dev_type = CL_DEVICE_TYPE_GPU;
      else if (HIPxxDeviceTypeStr == "default")
        selected_dev_type = CL_DEVICE_TYPE_DEFAULT;
      else if (HIPxxDeviceTypeStr == "accel")
        selected_dev_type = CL_DEVICE_TYPE_ACCELERATOR;
      else
        throw InvalidDeviceType(
            "Unknown value provided for HIPXX_DEVICE_TYPE\n");
      std::cout << "Using Devices of type " << HIPxxDeviceTypeStr << "\n";

    } catch (const InvalidDeviceType &e) {
      logCritical("{}\n", e.what());
      return;
    } catch (const InvalidPlatformOrDeviceNumber &e) {
      logCritical("{}\n", e.what());
      return;
    } catch (const std::invalid_argument &e) {
      logCritical(
          "Could not convert HIPXX_PLATFORM or HIPXX_DEVICES to a number");
      return;
    } catch (const std::out_of_range &e) {
      logCritical("HIPXX_PLATFORM or HIPXX_DEVICES is out of range", "");
      return;
    }

    std::vector<cl::Device> spirv_enabled_devices;
    for (cl::Device dev : enabled_devices) {
      std::string ver = dev.getInfo<CL_DEVICE_IL_VERSION>(&err);
      if ((err == CL_SUCCESS) && (ver.rfind("SPIR-V_1.", 0) == 0)) {
        spirv_enabled_devices.push_back(dev);
      }
    }

    // TODO uncomment this once testing on SPIR-V Enabled OpenCL HW
    // std::cout << "SPIR-V Enabled Devices: " << spirv_enabled_devices.size()
    //          << "\n";
    // for (int i = 0; i < spirv_enabled_devices.size(); i++) {
    //  std::cout << i << ". "
    //            << spirv_enabled_devices[i].getInfo<CL_DEVICE_NAME>() << "\n";
    //}

    // Create context which has devices
    // Create queues that have devices each of which has an associated context
    // TODO Change this to spirv_enabled_devices
    cl::Context *ctx = new cl::Context(enabled_devices);
    HIPxxContextOpenCL *hipxx_context = new HIPxxContextOpenCL(ctx);
    Backend->add_context(hipxx_context);
    for (int i = 0; i < enabled_devices.size(); i++) {
      auto dev = enabled_devices[i];
      HIPxxDeviceOpenCL *hipxx_dev =
          new HIPxxDeviceOpenCL(hipxx_context, &dev, i);
      logDebug("HIPxxDeviceOpenCL {}",
               hipxx_dev->cl_dev->getInfo<CL_DEVICE_NAME>());
      hipxx_dev->populate_device_properties();
      Backend->add_device(hipxx_dev);
      HIPxxQueueOpenCL *queue = new HIPxxQueueOpenCL(hipxx_context, hipxx_dev);
      Backend->add_queue(queue);
    }
    std::cout << "OpenCL Context Initialized.\n";
  };

  virtual void initialize() override {
    std::string empty;
    initialize(empty, empty, empty);
  }
  void uninitialize() override {
    logTrace("HIPxxBackendOpenCL uninitializing");
    logWarn("HIPxxBackendOpenCL->uninitialize() not implemented");
  }
};

class HIPxxEventOpenCL : public HIPxxEvent {
 protected:
  cl::Event *cl_event;
};

#endif