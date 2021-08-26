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

class HIPxxContextOpenCL : public HIPxxContext {
 public:
  cl::Context *cl_ctx;
  HIPxxContextOpenCL(cl::Context *ctx_in) {
    std::cout << "HIPxxContextOpenCL Initialized via OpenCL Context pointer.\n";
    cl_ctx = ctx_in;
  }
};

class HIPxxExecItemOpenCL : public HIPxxExecItem {
 public:
  cl::Kernel *kernel;
  virtual void run() override { std::cout << "HIPxxExecItemOpenCL run()\n"; };
};

class HIPxxDeviceOpenCL : public HIPxxDevice {
 public:
  cl::Device *cl_dev;
  cl::Context *cl_ctx;
  HIPxxDeviceOpenCL(cl::Context *ctx_in, cl::Device *dev_in) {
    std::cout << "HIPxxDeviceOpenCL initialized via OpenCL device pointer and "
                 "context pointer\n";
    cl_dev = dev_in;
    cl_ctx = ctx_in;
  }

  virtual std::string get_name() override {
    if (cl_dev == nullptr) {
      logCritical("HIPxxDeviceOpenCL.get_name() called on uninitialized ptr\n",
                  "");
      std::abort();
    }
    return std::string(cl_dev->getInfo<CL_DEVICE_NAME>());
  }
};

class HIPxxQueueOpenCL : public HIPxxQueue {
 protected:
  // Any reason to make these private/protected?
  cl::Context *cl_ctx;
  cl::Device *cl_dev;
  cl::CommandQueue *cl_q;

 public:
  HIPxxQueueOpenCL() = delete;  // delete default constructor
  HIPxxQueueOpenCL(const HIPxxQueueOpenCL &) =
      delete;  // delete copy constructor

  // HIPxxQueueOpenCL(HIPxxContextOpenCL *_ctx, HIPxxDeviceOpenCL *_dev) {
  //   std::cout << "HIPxxQueueOpenCL Initialized via context, device
  //   pointers\n"; cl_ctx = _ctx->cl_ctx; cl_dev = _dev->cl_dev; cl_q = new
  //   cl::CommandQueue(*cl_ctx, *cl_dev);
  // };

  // Can get device and context from one object? No - each device might have
  // multiple contexts
  HIPxxQueueOpenCL(HIPxxContextOpenCL *_ctx, HIPxxDeviceOpenCL *_dev) {
    std::cout << "HIPxxQueueOpenCL Initialized via context, device pointers\n";
    cl_ctx = _ctx->cl_ctx;
    cl_dev = _dev->cl_dev;
    cl_q = new cl::CommandQueue(*cl_ctx, *cl_dev);
    hipxx_device = _dev;
    hipxx_context = _ctx;
  };

  ~HIPxxQueueOpenCL() {
    delete cl_ctx;
    delete cl_dev;
  }

  virtual void submit(HIPxxExecItem *_e) override {
    std::cout << "HIPxxQueueOpenCL.submit()\n";
    HIPxxExecItemOpenCL *e = (HIPxxExecItemOpenCL *)_e;
    cl::Kernel kernel;  // HIPxxExecItem.get_kernel()
    _e->run();
  }
};

class HIPxxBackendOpenCL : public HIPxxBackend {
 public:
  void initialize(std::string HIPxxPlatformStr, std::string HIPxxDeviceTypeStr,
                  std::string HIPxxDeviceStr) override {
    std::cout << "HIPxxBackendOpenCL Initialize\n";
    std::vector<cl::Platform> Platforms;
    cl_int err = cl::Platform::get(&Platforms);
    if (err != CL_SUCCESS) {
      std::cout << "Failed to get OpenCL platforms! Exiting...\n";
      return;
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
      // logCritical("{}\n", e.what());
      logCritical("\nInvalidDeviceType", "");
      return;
    } catch (const InvalidPlatformOrDeviceNumber &e) {
      // logCritical("{}\n", e.what());
      logCritical("\nInvalidPlatformOrDeviceNumber", "");
      return;
    } catch (const std::invalid_argument &e) {
      logCritical(
          "\nCould not convert HIPXX_PLATFORM or HIPXX_DEVICES to a number\n",
          "");
      return;
    } catch (const std::out_of_range &e) {
      logCritical("\nHIPXX_PLATFORM or HIPXX_DEVICES is out of range\n", "");
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
    for (auto dev : enabled_devices) {
      HIPxxDeviceOpenCL *hipxx_dev = new HIPxxDeviceOpenCL(ctx, &dev);
      Backend->add_device(hipxx_dev);
      HIPxxQueueOpenCL *queue = new HIPxxQueueOpenCL(hipxx_context, hipxx_dev);
      // std::cout << "Adding Queue " << queue->dev
      Backend->add_queue(queue);
    }
    std::cout << "OpenCL Context Initialized.\n";
  };
};

#endif