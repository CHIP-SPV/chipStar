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
#include "spirv.hh"

class HIPxxContextOpenCL;
class HIPxxDeviceOpenCL;
class HIPxxExecItemOpenCL;
class HIPxxKernelOpenCL;
class HIPxxQueueOpenCL;
class HIPxxEventOpenCL;
class HIPxxBackendOpenCL;
class HIPxxModuleOpenCL;

class HIPxxModuleOpenCL : public HIPxxModule {
 protected:
  cl::Program program;

 public:
  virtual void compile(HIPxxDevice *hipxx_dev) override;
  cl::Program &get() { return program; }
};

class SVMemoryRegion {
  // ContextMutex should be enough

  std::map<void *, size_t> SvmAllocations;
  cl::Context Context;

 public:
  void init(cl::Context &C) { Context = C; }
  SVMemoryRegion &operator=(SVMemoryRegion &&rhs) {
    SvmAllocations = std::move(rhs.SvmAllocations);
    Context = std::move(rhs.Context);
    return *this;
  }

  void *allocate(cl::Context ctx, size_t size);
  bool free(void *p, size_t *size);
  bool hasPointer(const void *p);
  bool pointerSize(void *ptr, size_t *size);
  bool pointerInfo(void *ptr, void **pbase, size_t *psize);
  int memCopy(void *dst, const void *src, size_t size, cl::CommandQueue &queue);
  int memFill(void *dst, size_t size, const void *pattern, size_t patt_size,
              cl::CommandQueue &queue);
  void clear();
};

class HIPxxContextOpenCL : public HIPxxContext {
 public:
  SVMemoryRegion svm_memory;
  cl::Context *cl_ctx;
  HIPxxContextOpenCL(cl::Context *ctx_in);

  void *allocate(size_t size) override;
  virtual hipError_t memCopy(void *dst, const void *src, size_t size,
                             hipStream_t stream) override;
};

class HIPxxDeviceOpenCL : public HIPxxDevice {
 public:
  cl::Device *cl_dev;
  cl::Context *cl_ctx;
  HIPxxDeviceOpenCL(HIPxxContextOpenCL *hipxx_ctx, cl::Device *dev_in, int idx);

  virtual void populateDeviceProperties() override;
  virtual std::string getName() override;

  virtual void reset() override;
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

  virtual hipError_t launch(HIPxxExecItem *exec_item) override;
  virtual void finish() override;

  virtual hipError_t memCopy(void *dst, const void *src, size_t size) override;
  cl::CommandQueue *get() { return cl_q; }
};

class HIPxxKernelOpenCL : public HIPxxKernel {
 private:
  std::string name;
  size_t TotalArgSize;

 public:
  OCLFuncInfo *FuncInfo;
  cl::Kernel ocl_kernel;

  HIPxxKernelOpenCL(const cl::Kernel &&cl_kernel, std::string name_in,
                    const void *hostptr_in, int kernel_idx,
                    OpenCLFunctionInfoMap &FuncInfoMap)
      : name(name_in), ocl_kernel(cl_kernel) {
    // ocl_kernel = cl::Kernel(cl_kernel);
    host_f_name = name_in;
    host_f_ptr = hostptr_in;

    int err = 0;
    name = ocl_kernel.getInfo<CL_KERNEL_FUNCTION_NAME>(&err);
    if (err != CL_SUCCESS) {
      logError("clGetKernelInfo(CL_KERNEL_FUNCTION_NAME) failed: {}\n", err);
    }

    logDebug("Kernel {} is: {} \n", kernel_idx, name);

    auto it = FuncInfoMap.find(name);
    assert(it != FuncInfoMap.end());
    FuncInfo = it->second;

    // TODO attributes
    cl_uint NumArgs = ocl_kernel.getInfo<CL_KERNEL_NUM_ARGS>(&err);
    if (err != CL_SUCCESS) {
      logError("clGetKernelInfo(CL_KERNEL_NUM_ARGS) failed: {}\n", err);
    }

    assert(FuncInfo->ArgTypeInfo.size() == NumArgs);

    if (NumArgs > 0) {
      logDebug("Kernel {} numArgs: {} \n", name, NumArgs);
      logDebug("  RET_TYPE: {} {} {}\n", FuncInfo->retTypeInfo.size,
               (unsigned)FuncInfo->retTypeInfo.space,
               (unsigned)FuncInfo->retTypeInfo.type);
      for (auto &argty : FuncInfo->ArgTypeInfo) {
        logDebug("  ARG: SIZE {} SPACE {} TYPE {}\n", argty.size,
                 (unsigned)argty.space, (unsigned)argty.type);
        TotalArgSize += argty.size;
      }
    }
  }

  OCLFuncInfo *get_func_info() const { return FuncInfo; }
  std::string get_name() { return name; }
  cl::Kernel get() const { return ocl_kernel; }
  size_t getTotalArgSize() const { return TotalArgSize; };
};

class HIPxxExecItemOpenCL : public HIPxxExecItem {
 private:
  cl::Kernel *cl_kernel;

 public:
  OCLFuncInfo FuncInfo;
  virtual hipError_t launch(HIPxxKernel *hipxx_kernel) override;
  int setup_all_args(HIPxxKernelOpenCL *kernel);
  cl::Kernel *get() { return cl_kernel; }
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

      std::transform(HIPxxDeviceTypeStr.begin(), HIPxxDeviceTypeStr.end(),
                     HIPxxDeviceTypeStr.begin(), ::tolower);
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
    //            << spirv_enabled_devices[i].getInfo<CL_DEVICE_NAME>() <<
    //            "\n";
    //}

    // Create context which has devices
    // Create queues that have devices each of which has an associated context
    // TODO Change this to spirv_enabled_devices
    cl::Context *ctx = new cl::Context(enabled_devices);
    HIPxxContextOpenCL *hipxx_context = new HIPxxContextOpenCL(ctx);
    Backend->addContext(hipxx_context);
    for (int i = 0; i < enabled_devices.size(); i++) {
      cl::Device *dev = new cl::Device(enabled_devices[i]);
      HIPxxDeviceOpenCL *hipxx_dev =
          new HIPxxDeviceOpenCL(hipxx_context, dev, i);
      logDebug("HIPxxDeviceOpenCL {}",
               hipxx_dev->cl_dev->getInfo<CL_DEVICE_NAME>());
      hipxx_dev->populateDeviceProperties();
      Backend->addDevice(hipxx_dev);
      HIPxxQueueOpenCL *queue = new HIPxxQueueOpenCL(hipxx_context, hipxx_dev);
      Backend->addQueue(queue);
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