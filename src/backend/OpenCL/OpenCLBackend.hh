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

class HIPxxContextOpenCL;
class HIPxxEventOpenCL : public HIPxxEvent {
 protected:
  cl::Event *cl_event;
};

class HIPxxContextOpenCL : public HIPxxContext {
 public:
  cl::Context *cl_ctx;
  HIPxxContextOpenCL(cl::Context *ctx_in) {
    logDebug("HIPxxContextOpenCL Initialized via OpenCL Context pointer.");
    cl_ctx = ctx_in;
  }

  void *allocate(size_t size) override {
    logWarn("HIPxxContextOpenCL->allocate() not yet implemented");
    return (void *)0xDEADBEEF;
  }
};

class HIPxxExecItemOpenCL : public HIPxxExecItem {
 public:
  cl::Kernel *kernel;
  virtual void run() override { logDebug("HIPxxExecItemOpenCL run()\n"); };
};

class HIPxxDeviceOpenCL : public HIPxxDevice {
 public:
  cl::Device *cl_dev;
  cl::Context *cl_ctx;
  HIPxxDeviceOpenCL(HIPxxContextOpenCL *hipxx_ctx, cl::Device *dev_in,
                    int idx) {
    logDebug(
        "HIPxxDeviceOpenCL initialized via OpenCL device pointer and context "
        "pointer");
    cl_dev = dev_in;
    cl_ctx = hipxx_ctx->cl_ctx;
    pcie_idx = idx;

    hipxx_ctx->add_device(this);
    hipxx_contexts.push_back(hipxx_ctx);
  }

  virtual void populate_device_properties() override {
    logTrace("HIPxxDeviceOpenCL->populate_device_properties()");
    cl_int err;
    std::string Temp;

    assert(cl_dev != nullptr);
    Temp = cl_dev->getInfo<CL_DEVICE_NAME>();
    strncpy(hip_device_props.name, Temp.c_str(), 255);
    hip_device_props.name[255] = 0;

    hip_device_props.totalGlobalMem =
        cl_dev->getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>(&err);

    hip_device_props.sharedMemPerBlock =
        cl_dev->getInfo<CL_DEVICE_LOCAL_MEM_SIZE>(&err);

    hip_device_props.maxThreadsPerBlock =
        cl_dev->getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&err);

    std::vector<size_t> wi = cl_dev->getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

    hip_device_props.maxThreadsDim[0] = wi[0];
    hip_device_props.maxThreadsDim[1] = wi[1];
    hip_device_props.maxThreadsDim[2] = wi[2];

    // Maximum configured clock frequency of the device in MHz.
    hip_device_props.clockRate =
        1000 * cl_dev->getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();

    hip_device_props.multiProcessorCount =
        cl_dev->getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    hip_device_props.l2CacheSize =
        cl_dev->getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();

    // not actually correct
    hip_device_props.totalConstMem =
        cl_dev->getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();

    // totally made up
    hip_device_props.regsPerBlock = 64;

    // The minimum subgroup size on an intel GPU
    if (cl_dev->getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) {
      std::vector<uint> sg = cl_dev->getInfo<CL_DEVICE_SUB_GROUP_SIZES_INTEL>();
      if (sg.begin() != sg.end())
        hip_device_props.warpSize = *std::min_element(sg.begin(), sg.end());
    }
    hip_device_props.maxGridSize[0] = hip_device_props.maxGridSize[1] =
        hip_device_props.maxGridSize[2] = 65536;
    hip_device_props.memoryClockRate = 1000;
    hip_device_props.memoryBusWidth = 256;
    hip_device_props.major = 2;
    hip_device_props.minor = 0;

    hip_device_props.maxThreadsPerMultiProcessor = 10;

    hip_device_props.computeMode = 0;
    hip_device_props.arch = {};

    Temp = cl_dev->getInfo<CL_DEVICE_EXTENSIONS>();
    if (Temp.find("cl_khr_global_int32_base_atomics") != std::string::npos)
      hip_device_props.arch.hasGlobalInt32Atomics = 1;
    else
      hip_device_props.arch.hasGlobalInt32Atomics = 0;

    if (Temp.find("cl_khr_local_int32_base_atomics") != std::string::npos)
      hip_device_props.arch.hasSharedInt32Atomics = 1;
    else
      hip_device_props.arch.hasSharedInt32Atomics = 0;

    if (Temp.find("cl_khr_int64_base_atomics") != std::string::npos) {
      hip_device_props.arch.hasGlobalInt64Atomics = 1;
      hip_device_props.arch.hasSharedInt64Atomics = 1;
    } else {
      hip_device_props.arch.hasGlobalInt64Atomics = 1;
      hip_device_props.arch.hasSharedInt64Atomics = 1;
    }

    if (Temp.find("cl_khr_fp64") != std::string::npos)
      hip_device_props.arch.hasDoubles = 1;
    else
      hip_device_props.arch.hasDoubles = 0;

    hip_device_props.clockInstructionRate = 2465;
    hip_device_props.concurrentKernels = 1;
    hip_device_props.pciDomainID = 0;
    hip_device_props.pciBusID = 0x10;
    hip_device_props.pciDeviceID = 0x40 + pcie_idx;
    hip_device_props.isMultiGpuBoard = 0;
    hip_device_props.canMapHostMemory = 1;
    hip_device_props.gcnArch = 0;
    hip_device_props.integrated = 0;
    hip_device_props.maxSharedMemoryPerMultiProcessor = 0;
  }
  virtual std::string get_name() override {
    if (cl_dev == nullptr) {
      logCritical("HIPxxDeviceOpenCL.get_name() called on uninitialized ptr");
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
    logDebug("HIPxxQueueOpenCL Initialized via context, device pointers");
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
    logDebug("HIPxxQueueOpenCL.submit()");
    HIPxxExecItemOpenCL *e = (HIPxxExecItemOpenCL *)_e;
    cl::Kernel kernel;  // HIPxxExecItem.get_kernel()
    _e->run();
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

  void uninitialize() override {
    logTrace("HIPxxBackendOpenCL uninitializing");
    logWarn("HIPxxBackendOpenCL->uninitialize() not implemented");
  }
};

#endif