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

std::string resultToString(int status);

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

  virtual void reset() override;

  virtual CHIPModuleOpenCL *addModule(std::string *module_str) override {
    CHIPModuleOpenCL *mod = new CHIPModuleOpenCL(module_str);
    chip_modules.push_back(mod);
    return mod;
  }

  virtual void addQueue(unsigned int flags, int priority) override;
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
  CHIPKernelOpenCL(const cl::Kernel &&cl_kernel, std::string host_f_name_,
                   OCLFuncInfo func_info_);

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
  int setup_all_args(CHIPKernelOpenCL *kernel);
  cl::Kernel *get() { return cl_kernel; }
};

class CHIPBackendOpenCL : public CHIPBackend {
 public:
  void initialize_(std::string CHIPPlatformStr, std::string CHIPDeviceTypeStr,
                   std::string CHIPDeviceStr) override;

  void uninitialize() override;
};

class CHIPEventOpenCL : public CHIPEvent {
 protected:
  cl::Event *cl_event;
};

#endif