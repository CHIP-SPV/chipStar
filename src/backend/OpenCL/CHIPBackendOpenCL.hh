/**
 * @file Backend.hh
 * @author Paulius Velesko (pvelesko@gmail.com)
 * @brief OpenCL backend for CHIP. CHIPBackendOpenCL class definition with
 * inheritance from CHIPBackend. Subsequent virtual function overrides.
 * @version 0.1
 * @date 2021-08-19
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef CHIP_BACKEND_OPENCL_H
#define CHIP_BACKEND_OPENCL_H

#define CL_TARGET_OPENCL_VERSION 210
#define CL_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 210
#define CL_HPP_MINIMUM_OPENCL_VERSION 200

#include <CL/cl_ext_intel.h>

#include <CL/opencl.hpp>

#include "../../CHIPBackend.hh"
#include "exceptions.hh"
#include "spirv.hh"

class CHIPContextOpenCL;
class CHIPDeviceOpenCL;
class CHIPExecItemOpenCL;
class CHIPKernelOpenCL;
class CHIPQueueOpenCL;
class CHIPEventOpenCL;
class CHIPBackendOpenCL;
class CHIPModuleOpenCL;

class CHIPModuleOpenCL : public CHIPModule {
 protected:
  cl::Program program;

 public:
  CHIPModuleOpenCL(std::string *module_str) : CHIPModule(module_str){};
  virtual void compile(CHIPDevice *chip_dev) override;
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

class CHIPContextOpenCL : public CHIPContext {
 public:
  SVMemoryRegion svm_memory;
  cl::Context *cl_ctx;
  CHIPContextOpenCL(cl::Context *ctx_in);

  void *allocate_(size_t size, size_t alignment,
                  CHIPMemoryType mem_type) override;

  void free_(void *ptr) override{};
  virtual hipError_t memCopy(void *dst, const void *src, size_t size,
                             hipStream_t stream) override;
  cl::Context *get() { return cl_ctx; }
};

class CHIPDeviceOpenCL : public CHIPDevice {
 public:
  cl::Device *cl_dev;
  cl::Context *cl_ctx;
  CHIPDeviceOpenCL(CHIPContextOpenCL *chip_ctx, cl::Device *dev_in, int idx);

  cl::Device *get() { return cl_dev; }

  virtual void populateDeviceProperties_() override;
  virtual std::string getName() override;

  virtual void reset() override;

  virtual CHIPModuleOpenCL *addModule(std::string *module_str) override {
    CHIPModuleOpenCL *mod = new CHIPModuleOpenCL(module_str);
    chip_modules.push_back(mod);
    return mod;
  }
};

class CHIPQueueOpenCL : public CHIPQueue {
 protected:
  // Any reason to make these private/protected?
  cl::Context *cl_ctx;
  cl::Device *cl_dev;
  cl::CommandQueue *cl_q;

 public:
  CHIPQueueOpenCL() = delete;  // delete default constructor
  CHIPQueueOpenCL(const CHIPQueueOpenCL &) = delete;
  CHIPQueueOpenCL(CHIPDevice *chip_device);
  ~CHIPQueueOpenCL();

  virtual hipError_t launch(CHIPExecItem *exec_item) override;
  virtual void finish() override;

  virtual hipError_t memCopy(void *dst, const void *src, size_t size) override;
  virtual hipError_t memCopyAsync(void *dst, const void *src,
                                  size_t size) override;
  cl::CommandQueue *get() { return cl_q; }
};

class CHIPKernelOpenCL : public CHIPKernel {
 private:
  std::string name;
  size_t TotalArgSize;
  OCLFuncInfo *func_info;
  cl::Kernel ocl_kernel;

 public:
  CHIPKernelOpenCL(const cl::Kernel &&cl_kernel,
                   OpenCLFunctionInfoMap &func_info_map) {
    ocl_kernel = cl_kernel;

    int err = 0;
    name = ocl_kernel.getInfo<CL_KERNEL_FUNCTION_NAME>(&err);
    setName(name);
    if (err != CL_SUCCESS) {
      logError("clGetKernelInfo(CL_KERNEL_FUNCTION_NAME) failed: {}\n", err);
    }

    auto it = func_info_map.find(name);
    assert(it != func_info_map.end());
    func_info = it->second;

    // TODO attributes
    cl_uint NumArgs = ocl_kernel.getInfo<CL_KERNEL_NUM_ARGS>(&err);
    if (err != CL_SUCCESS) {
      logError("clGetKernelInfo(CL_KERNEL_NUM_ARGS) failed: {}\n", err);
    }

    assert(func_info->ArgTypeInfo.size() == NumArgs);

    if (NumArgs > 0) {
      logDebug("Kernel {} numArgs: {} \n", name, NumArgs);
      logDebug("  RET_TYPE: {} {} {}\n", func_info->retTypeInfo.size,
               (unsigned)func_info->retTypeInfo.space,
               (unsigned)func_info->retTypeInfo.type);
      for (auto &argty : func_info->ArgTypeInfo) {
        logDebug("  ARG: SIZE {} SPACE {} TYPE {}\n", argty.size,
                 (unsigned)argty.space, (unsigned)argty.type);
        TotalArgSize += argty.size;
      }
    }
  }

  OCLFuncInfo *get_func_info() const { return func_info; }
  std::string get_name() { return name; }
  cl::Kernel get() const { return ocl_kernel; }
  size_t getTotalArgSize() const { return TotalArgSize; };
};

class CHIPExecItemOpenCL : public CHIPExecItem {
 private:
  cl::Kernel *cl_kernel;

 public:
  OCLFuncInfo FuncInfo;
  virtual hipError_t launch(CHIPKernel *chip_kernel) override;
  int setup_all_args(CHIPKernelOpenCL *kernel);
  cl::Kernel *get() { return cl_kernel; }
};

class CHIPBackendOpenCL : public CHIPBackend {
 public:
  void initialize_(std::string CHIPPlatformStr, std::string CHIPDeviceTypeStr,
                   std::string CHIPDeviceStr) override {
    logDebug("CHIPBackendOpenCL Initialize");
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
      if (!CHIPDeviceStr.compare("all")) {  // Use all devices that match type
        selected_device = -1;
      } else {
        selected_device = std::stoi(CHIPDeviceStr);
      }

      // Platform index in range?
      selected_platform = std::stoi(CHIPPlatformStr);
      if ((selected_platform < 0) || (selected_platform >= Platforms.size()))
        throw InvalidPlatformOrDeviceNumber(
            "CHIP_PLATFORM: platform number out of range");
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
            "CHIP_DEVICE: device number out of range");
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
            "CHIP_DEVICE: can't get devices for platform");

      std::transform(CHIPDeviceTypeStr.begin(), CHIPDeviceTypeStr.end(),
                     CHIPDeviceTypeStr.begin(), ::tolower);
      if (CHIPDeviceTypeStr == "all")
        selected_dev_type = CL_DEVICE_TYPE_ALL;
      else if (CHIPDeviceTypeStr == "cpu")
        selected_dev_type = CL_DEVICE_TYPE_CPU;
      else if (CHIPDeviceTypeStr == "gpu")
        selected_dev_type = CL_DEVICE_TYPE_GPU;
      else if (CHIPDeviceTypeStr == "default")
        selected_dev_type = CL_DEVICE_TYPE_DEFAULT;
      else if (CHIPDeviceTypeStr == "accel")
        selected_dev_type = CL_DEVICE_TYPE_ACCELERATOR;
      else
        throw InvalidDeviceType(
            "Unknown value provided for CHIP_DEVICE_TYPE\n");
      std::cout << "Using Devices of type " << CHIPDeviceTypeStr << "\n";

    } catch (const InvalidDeviceType &e) {
      logCritical("{}\n", e.what());
      return;
    } catch (const InvalidPlatformOrDeviceNumber &e) {
      logCritical("{}\n", e.what());
      return;
    } catch (const std::invalid_argument &e) {
      logCritical(
          "Could not convert CHIP_PLATFORM or CHIP_DEVICES to a number");
      return;
    } catch (const std::out_of_range &e) {
      logCritical("CHIP_PLATFORM or CHIP_DEVICES is out of range", "");
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
    CHIPContextOpenCL *chip_context = new CHIPContextOpenCL(ctx);
    Backend->addContext(chip_context);
    for (int i = 0; i < enabled_devices.size(); i++) {
      cl::Device *dev = new cl::Device(enabled_devices[i]);
      CHIPDeviceOpenCL *chip_dev = new CHIPDeviceOpenCL(chip_context, dev, i);
      logDebug("CHIPDeviceOpenCL {}",
               chip_dev->cl_dev->getInfo<CL_DEVICE_NAME>());
      chip_dev->populateDeviceProperties();
      Backend->addDevice(chip_dev);
      CHIPQueueOpenCL *queue = new CHIPQueueOpenCL(chip_dev);
      chip_dev->addQueue(queue);
      Backend->addQueue(queue);
    }
    std::cout << "OpenCL Context Initialized.\n";
  };

  void uninitialize() override {
    logTrace("CHIPBackendOpenCL uninitializing");
    logWarn("CHIPBackendOpenCL->uninitialize() not implemented");
  }
};

class CHIPEventOpenCL : public CHIPEvent {
 protected:
  cl::Event *cl_event;
};

#endif