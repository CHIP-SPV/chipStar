#include "HIPxxBackendOpenCL.hh"

HIPxxDeviceOpenCL::HIPxxDeviceOpenCL(HIPxxContextOpenCL *hipxx_ctx,
                                     cl::Device *dev_in, int idx) {
  logDebug(
      "HIPxxDeviceOpenCL initialized via OpenCL device pointer and context "
      "pointer");
  cl_dev = dev_in;
  cl_ctx = hipxx_ctx->cl_ctx;
  pcie_idx = idx;

  hipxx_ctx->add_device(this);
  hipxx_contexts.push_back(hipxx_ctx);
}

void HIPxxDeviceOpenCL::populate_device_properties() {
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

std::string HIPxxDeviceOpenCL::get_name() {
  if (cl_dev == nullptr) {
    logCritical("HIPxxDeviceOpenCL.get_name() called on uninitialized ptr");
    std::abort();
  }
  return std::string(cl_dev->getInfo<CL_DEVICE_NAME>());
}

bool HIPxxContextOpenCL::register_function_as_kernel(
    std::string *module_str, const void *HostFunctionPtr,
    const char *FunctionName) {
  // TODO Most of this can go to Base class
  logTrace("HIPxxContextOpenCL.register_function_as_kernel()");

  OpenCLFunctionInfoMap FuncInfos;

  std::string binary = *module_str;
  size_t numWords = binary.size() / 4;
  int32_t *bindata = new int32_t[numWords + 1];
  std::memcpy(bindata, binary.data(), binary.size());
  bool res = parseSPIR(bindata, numWords, FuncInfos);
  delete[] bindata;
  if (!res) {
    logError("SPIR-V parsing failed\n");
    return false;
  }

  int err;
  std::vector<char> binary_vec(binary.begin(), binary.end());
  auto Program = cl::Program(*cl_ctx, binary_vec, false, &err);
  if (err != CL_SUCCESS) {
    logError("CreateProgramWithIL Failed: {}\n", err);
    return false;
  }

  for (HIPxxDevice *hipxx_dev : hipxx_devices) {
    HIPxxDeviceOpenCL *hipxx_ocl_device = (HIPxxDeviceOpenCL *)hipxx_dev;
    std::string name = hipxx_ocl_device->get_name();

    int build_failed = Program.build("-x spir -cl-kernel-arg-info");

    std::string log = Program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(
        *hipxx_ocl_device->cl_dev, &err);
    if (err != CL_SUCCESS) {
      logError("clGetProgramBuildInfo() Failed:{}\n", err);
      return false;
    }
    logDebug("Program BUILD LOG for device #{}:{}:\n{}\n",
             hipxx_ocl_device->pcie_idx, name, log);
    if (build_failed != CL_SUCCESS) {
      logError("clBuildProgram() Failed: {}\n", build_failed);
      return false;
    }

    std::vector<cl::Kernel> kernels;
    err = Program.createKernels(&kernels);
    if (err != CL_SUCCESS) {
      logError("clCreateKernels() Failed: {}\n", err);
      return false;
    }
    logDebug("Kernels in program: {} \n", kernels.size());
    for (int kernel_idx = 0; kernel_idx < kernels.size(); kernel_idx++) {
      HIPxxKernelOpenCL *hipxx_kernel = new HIPxxKernelOpenCL(
          std::move(kernels[kernel_idx]), std::string(FunctionName),
          HostFunctionPtr, kernel_idx, FuncInfos);
      hipxx_ocl_device->add_kernel(hipxx_kernel);
    }
  }  // Loop over devices in this context
  return true;
}