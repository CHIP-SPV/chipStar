#ifndef CL_HIP_ERROR_CONVERSION_HH
#define CL_HIP_ERROR_CONVERSION_HH

#include "../../CHIPException.hh"
#include <CL/cl.h>
#include <unordered_map>

using cl_hip_error_map_t = std::unordered_map<cl_int, hipError_t>;

// Map of OpenCL API calls to error conversion maps
const std::unordered_map<void *, cl_hip_error_map_t> CL_HIP_ERROR_MAPS = {
    {(void *)&clGetPlatformIDs,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clGetDeviceIDs,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_PLATFORM, hipErrorInvalidDevice},
      {CL_INVALID_DEVICE_TYPE, hipErrorInvalidDevice},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_DEVICE_NOT_FOUND, hipErrorNoDevice}}},

    {(void *)&clCreateContext,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_PLATFORM, hipErrorInvalidDevice},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_INVALID_DEVICE, hipErrorInvalidDevice},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clCreateCommandQueue,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_CONTEXT, hipErrorInvalidResourceHandle},
      {CL_INVALID_DEVICE, hipErrorInvalidDevice},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clCreateBuffer,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_CONTEXT, hipErrorInvalidResourceHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_INVALID_BUFFER_SIZE, hipErrorInvalidValue},
      {CL_INVALID_HOST_PTR, hipErrorInvalidValue},
      {CL_MEM_OBJECT_ALLOCATION_FAILURE, hipErrorOutOfMemory},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clCreateProgramWithSource,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_CONTEXT, hipErrorInvalidResourceHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clBuildProgram,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_PROGRAM, hipErrorInvalidResourceHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_INVALID_DEVICE, hipErrorInvalidDevice},
      {CL_INVALID_BINARY, hipErrorInvalidImage},
      {CL_INVALID_BUILD_OPTIONS, hipErrorInvalidValue},
      {CL_INVALID_OPERATION, hipErrorInvalidValue},
      {CL_COMPILER_NOT_AVAILABLE, hipErrorNotSupported},
      {CL_BUILD_PROGRAM_FAILURE, hipErrorInvalidImage},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clCreateKernel,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_PROGRAM, hipErrorInvalidResourceHandle},
      {CL_INVALID_PROGRAM_EXECUTABLE, hipErrorInvalidImage},
      {CL_INVALID_KERNEL_NAME, hipErrorInvalidDeviceFunction},
      {CL_INVALID_KERNEL_DEFINITION, hipErrorInvalidDeviceFunction},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clSetKernelArg,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_KERNEL, hipErrorInvalidHandle},
      {CL_INVALID_ARG_INDEX, hipErrorInvalidValue},
      {CL_INVALID_ARG_VALUE, hipErrorInvalidValue},
      {CL_INVALID_MEM_OBJECT, hipErrorInvalidResourceHandle},
      {CL_INVALID_SAMPLER, hipErrorInvalidResourceHandle},
      {CL_INVALID_ARG_SIZE, hipErrorInvalidValue}}},

    {(void *)&clEnqueueNDRangeKernel,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_PROGRAM_EXECUTABLE, hipErrorInvalidImage},
      {CL_INVALID_COMMAND_QUEUE, hipErrorInvalidResourceHandle},
      {CL_INVALID_KERNEL, hipErrorInvalidHandle},
      {CL_INVALID_CONTEXT, hipErrorInvalidResourceHandle},
      {CL_INVALID_KERNEL_ARGS, hipErrorInvalidValue},
      {CL_INVALID_WORK_DIMENSION, hipErrorInvalidValue},
      {CL_INVALID_WORK_GROUP_SIZE, hipErrorInvalidValue},
      {CL_INVALID_WORK_ITEM_SIZE, hipErrorInvalidValue},
      {CL_INVALID_GLOBAL_OFFSET, hipErrorInvalidValue},
      {CL_OUT_OF_RESOURCES, hipErrorOutOfMemory},
      {CL_MEM_OBJECT_ALLOCATION_FAILURE, hipErrorOutOfMemory},
      {CL_INVALID_EVENT_WAIT_LIST, hipErrorInvalidResourceHandle},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clEnqueueReadBuffer,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_COMMAND_QUEUE, hipErrorInvalidResourceHandle},
      {CL_INVALID_CONTEXT, hipErrorInvalidResourceHandle},
      {CL_INVALID_MEM_OBJECT, hipErrorInvalidResourceHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_INVALID_EVENT_WAIT_LIST, hipErrorInvalidResourceHandle},
      {CL_MEM_OBJECT_ALLOCATION_FAILURE, hipErrorOutOfMemory},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clEnqueueWriteBuffer,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_COMMAND_QUEUE, hipErrorInvalidResourceHandle},
      {CL_INVALID_CONTEXT, hipErrorInvalidResourceHandle},
      {CL_INVALID_MEM_OBJECT, hipErrorInvalidResourceHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_INVALID_EVENT_WAIT_LIST, hipErrorInvalidResourceHandle},
      {CL_MEM_OBJECT_ALLOCATION_FAILURE, hipErrorOutOfMemory},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clEnqueueCopyBuffer,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_COMMAND_QUEUE, hipErrorInvalidResourceHandle},
      {CL_INVALID_CONTEXT, hipErrorInvalidResourceHandle},
      {CL_INVALID_MEM_OBJECT, hipErrorInvalidResourceHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_INVALID_EVENT_WAIT_LIST, hipErrorInvalidResourceHandle},
      {CL_MEM_COPY_OVERLAP, hipErrorInvalidValue},
      {CL_MEM_OBJECT_ALLOCATION_FAILURE, hipErrorOutOfMemory},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clReleaseMemObject,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_MEM_OBJECT, hipErrorInvalidResourceHandle}}},

    {(void *)&clReleaseKernel,
     {{CL_SUCCESS, hipSuccess}, {CL_INVALID_KERNEL, hipErrorInvalidHandle}}},

    {(void *)&clReleaseProgram,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_PROGRAM, hipErrorInvalidResourceHandle}}},

    {(void *)&clReleaseCommandQueue,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_COMMAND_QUEUE, hipErrorInvalidResourceHandle}}},

    {(void *)&clReleaseContext,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_CONTEXT, hipErrorInvalidResourceHandle}}},

    {(void *)&clGetEventInfo,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_EVENT, hipErrorInvalidResourceHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_OUT_OF_RESOURCES, hipErrorOutOfMemory},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clWaitForEvents,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_EVENT, hipErrorInvalidResourceHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_INVALID_CONTEXT, hipErrorInvalidResourceHandle},
      {CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST, hipErrorUnknown}}},

    {(void *)&clFinish,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_COMMAND_QUEUE, hipErrorInvalidResourceHandle},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clFlush,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_COMMAND_QUEUE, hipErrorInvalidResourceHandle},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clGetPlatformInfo,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_PLATFORM, hipErrorInvalidDevice},
      {CL_INVALID_VALUE, hipErrorInvalidValue}}},

    {(void *)&clGetDeviceInfo,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_DEVICE, hipErrorInvalidDevice},
      {CL_INVALID_VALUE, hipErrorInvalidValue}}},

    {(void *)&clCreateContextFromType,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_PLATFORM, hipErrorInvalidDevice},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_DEVICE_NOT_AVAILABLE, hipErrorNoDevice},
      {CL_DEVICE_NOT_FOUND, hipErrorNoDevice},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clRetainContext,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_CONTEXT, hipErrorInvalidResourceHandle}}},

    {(void *)&clGetContextInfo,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_CONTEXT, hipErrorInvalidResourceHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue}}},

    {(void *)&clRetainCommandQueue,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_COMMAND_QUEUE, hipErrorInvalidResourceHandle}}},

    {(void *)&clGetCommandQueueInfo,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_COMMAND_QUEUE, hipErrorInvalidResourceHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue}}},

    {(void *)&clRetainMemObject,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_MEM_OBJECT, hipErrorInvalidResourceHandle}}},

    {(void *)&clGetMemObjectInfo,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_MEM_OBJECT, hipErrorInvalidResourceHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue}}},

    {(void *)&clRetainProgram,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_PROGRAM, hipErrorInvalidResourceHandle}}},

    {(void *)&clGetProgramInfo,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_PROGRAM, hipErrorInvalidResourceHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue}}},

    {(void *)&clGetProgramBuildInfo,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_PROGRAM, hipErrorInvalidResourceHandle},
      {CL_INVALID_DEVICE, hipErrorInvalidDevice},
      {CL_INVALID_VALUE, hipErrorInvalidValue}}},

    {(void *)&clRetainKernel,
     {{CL_SUCCESS, hipSuccess}, {CL_INVALID_KERNEL, hipErrorInvalidHandle}}},

    {(void *)&clGetKernelInfo,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_KERNEL, hipErrorInvalidHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue}}},

    {(void *)&clGetKernelWorkGroupInfo,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_DEVICE, hipErrorInvalidDevice},
      {CL_INVALID_KERNEL, hipErrorInvalidHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue}}},

    {(void *)&clGetEventProfilingInfo,
     {{CL_SUCCESS, hipSuccess},
      {CL_PROFILING_INFO_NOT_AVAILABLE, hipErrorNotSupported},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_INVALID_EVENT, hipErrorInvalidResourceHandle}}},

    {(void *)&clRetainEvent,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_EVENT, hipErrorInvalidResourceHandle}}},

    {(void *)&clReleaseEvent,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_EVENT, hipErrorInvalidResourceHandle}}},

    {(void *)&clEnqueueMarker,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_COMMAND_QUEUE, hipErrorInvalidResourceHandle}}},

    {(void *)&clEnqueueBarrier,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_COMMAND_QUEUE, hipErrorInvalidResourceHandle}}},

    {(void *)&clEnqueueWaitForEvents,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_COMMAND_QUEUE, hipErrorInvalidResourceHandle},
      {CL_INVALID_EVENT, hipErrorInvalidResourceHandle}}},

    {(void *)&clEnqueueMapBuffer,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_COMMAND_QUEUE, hipErrorInvalidResourceHandle},
      {CL_INVALID_CONTEXT, hipErrorInvalidResourceHandle},
      {CL_INVALID_MEM_OBJECT, hipErrorInvalidResourceHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_INVALID_EVENT_WAIT_LIST, hipErrorInvalidResourceHandle},
      {CL_MISALIGNED_SUB_BUFFER_OFFSET, hipErrorInvalidValue},
      {CL_MAP_FAILURE, hipErrorMapFailed},
      {CL_MEM_OBJECT_ALLOCATION_FAILURE, hipErrorOutOfMemory},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clEnqueueUnmapMemObject,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_COMMAND_QUEUE, hipErrorInvalidResourceHandle},
      {CL_INVALID_MEM_OBJECT, hipErrorInvalidResourceHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_INVALID_EVENT_WAIT_LIST, hipErrorInvalidResourceHandle},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clGetKernelArgInfo,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_ARG_INDEX, hipErrorInvalidValue},
      {CL_INVALID_KERNEL, hipErrorInvalidHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_KERNEL_ARG_INFO_NOT_AVAILABLE, hipErrorNotSupported}}},

    {(void *)&clEnqueueFillBuffer,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_COMMAND_QUEUE, hipErrorInvalidResourceHandle},
      {CL_INVALID_CONTEXT, hipErrorInvalidResourceHandle},
      {CL_INVALID_MEM_OBJECT, hipErrorInvalidResourceHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_INVALID_EVENT_WAIT_LIST, hipErrorInvalidResourceHandle},
      {CL_MEM_OBJECT_ALLOCATION_FAILURE, hipErrorOutOfMemory},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clCreateImage,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_CONTEXT, hipErrorInvalidResourceHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_INVALID_IMAGE_FORMAT_DESCRIPTOR, hipErrorInvalidValue},
      {CL_INVALID_IMAGE_SIZE, hipErrorInvalidValue},
      {CL_INVALID_HOST_PTR, hipErrorInvalidValue},
      {CL_IMAGE_FORMAT_NOT_SUPPORTED, hipErrorNotSupported},
      {CL_MEM_OBJECT_ALLOCATION_FAILURE, hipErrorOutOfMemory},
      {CL_INVALID_OPERATION, hipErrorInvalidValue},
      {CL_OUT_OF_RESOURCES, hipErrorOutOfMemory},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clEnqueueReadImage,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_COMMAND_QUEUE, hipErrorInvalidResourceHandle},
      {CL_INVALID_CONTEXT, hipErrorInvalidResourceHandle},
      {CL_INVALID_MEM_OBJECT, hipErrorInvalidResourceHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_INVALID_EVENT_WAIT_LIST, hipErrorInvalidResourceHandle},
      {CL_MEM_OBJECT_ALLOCATION_FAILURE, hipErrorOutOfMemory},
      {CL_OUT_OF_RESOURCES, hipErrorOutOfMemory},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clEnqueueWriteImage,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_COMMAND_QUEUE, hipErrorInvalidResourceHandle},
      {CL_INVALID_CONTEXT, hipErrorInvalidResourceHandle},
      {CL_INVALID_MEM_OBJECT, hipErrorInvalidResourceHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_INVALID_EVENT_WAIT_LIST, hipErrorInvalidResourceHandle},
      {CL_MEM_OBJECT_ALLOCATION_FAILURE, hipErrorOutOfMemory},
      {CL_OUT_OF_RESOURCES, hipErrorOutOfMemory},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clEnqueueCopyImage,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_COMMAND_QUEUE, hipErrorInvalidResourceHandle},
      {CL_INVALID_CONTEXT, hipErrorInvalidResourceHandle},
      {CL_INVALID_MEM_OBJECT, hipErrorInvalidResourceHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_INVALID_EVENT_WAIT_LIST, hipErrorInvalidResourceHandle},
      {CL_MEM_OBJECT_ALLOCATION_FAILURE, hipErrorOutOfMemory},
      {CL_OUT_OF_RESOURCES, hipErrorOutOfMemory},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clEnqueueMapImage,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_COMMAND_QUEUE, hipErrorInvalidResourceHandle},
      {CL_INVALID_CONTEXT, hipErrorInvalidResourceHandle},
      {CL_INVALID_MEM_OBJECT, hipErrorInvalidResourceHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_INVALID_EVENT_WAIT_LIST, hipErrorInvalidResourceHandle},
      {CL_INVALID_IMAGE_SIZE, hipErrorInvalidValue},
      {CL_MAP_FAILURE, hipErrorMapFailed},
      {CL_MEM_OBJECT_ALLOCATION_FAILURE, hipErrorOutOfMemory},
      {CL_OUT_OF_RESOURCES, hipErrorOutOfMemory},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clEnqueueFillImage,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_COMMAND_QUEUE, hipErrorInvalidResourceHandle},
      {CL_INVALID_CONTEXT, hipErrorInvalidResourceHandle},
      {CL_INVALID_MEM_OBJECT, hipErrorInvalidResourceHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_INVALID_EVENT_WAIT_LIST, hipErrorInvalidResourceHandle},
      {CL_MEM_OBJECT_ALLOCATION_FAILURE, hipErrorOutOfMemory},
      {CL_OUT_OF_RESOURCES, hipErrorOutOfMemory},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clGetSupportedImageFormats,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_CONTEXT, hipErrorInvalidResourceHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_OUT_OF_RESOURCES, hipErrorOutOfMemory},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clCreateSampler,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_CONTEXT, hipErrorInvalidResourceHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_INVALID_OPERATION, hipErrorInvalidValue},
      {CL_OUT_OF_RESOURCES, hipErrorOutOfMemory},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clCompileProgram,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_PROGRAM, hipErrorInvalidResourceHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_INVALID_DEVICE, hipErrorInvalidDevice},
      {CL_INVALID_BINARY, hipErrorInvalidImage},
      {CL_INVALID_BUILD_OPTIONS, hipErrorInvalidValue},
      {CL_INVALID_OPERATION, hipErrorInvalidValue},
      {CL_COMPILER_NOT_AVAILABLE, hipErrorNotSupported},
      {CL_OUT_OF_RESOURCES, hipErrorOutOfMemory},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clCreateKernelsInProgram,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_PROGRAM, hipErrorInvalidResourceHandle},
      {CL_INVALID_PROGRAM_EXECUTABLE, hipErrorInvalidImage},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_OUT_OF_RESOURCES, hipErrorOutOfMemory},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clGetDeviceAndHostTimer,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_DEVICE, hipErrorInvalidDevice},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_OUT_OF_RESOURCES, hipErrorOutOfMemory},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clGetHostTimer,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_DEVICE, hipErrorInvalidDevice},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_OUT_OF_RESOURCES, hipErrorOutOfMemory},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clSetKernelArgSVMPointer,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_KERNEL, hipErrorInvalidHandle},
      {CL_INVALID_ARG_INDEX, hipErrorInvalidValue},
      {CL_INVALID_ARG_VALUE, hipErrorInvalidValue},
      {CL_OUT_OF_RESOURCES, hipErrorOutOfMemory},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clSetUserEventStatus,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_EVENT, hipErrorInvalidResourceHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_INVALID_OPERATION, hipErrorInvalidValue}}},

    {(void *)&clEnqueueBarrierWithWaitList,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_COMMAND_QUEUE, hipErrorInvalidResourceHandle},
      {CL_INVALID_EVENT_WAIT_LIST, hipErrorInvalidResourceHandle},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clGetPlatformIDs,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clCreateProgramWithIL,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_CONTEXT, hipErrorInvalidResourceHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_INVALID_BINARY, hipErrorInvalidImage},
      {CL_OUT_OF_RESOURCES, hipErrorOutOfMemory},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},

    {(void *)&clGetKernelArgInfo,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_ARG_INDEX, hipErrorInvalidValue},
      {CL_INVALID_KERNEL, hipErrorInvalidHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_KERNEL_ARG_INFO_NOT_AVAILABLE, hipErrorNotSupported}}},

    {(void *)&clEnqueueMarkerWithWaitList,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_COMMAND_QUEUE, hipErrorInvalidResourceHandle},
      {CL_INVALID_EVENT_WAIT_LIST, hipErrorInvalidResourceHandle},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},
    {(void *)&clSetKernelArgSVMPointer,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_KERNEL, hipErrorInvalidHandle},
      {CL_INVALID_ARG_INDEX, hipErrorInvalidValue},
      {CL_INVALID_ARG_VALUE, hipErrorInvalidValue},
      {CL_OUT_OF_RESOURCES, hipErrorOutOfMemory},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},
    {(void *)&clEnqueueSVMMemcpy,
     {{CL_SUCCESS, hipSuccess},
      {CL_INVALID_COMMAND_QUEUE, hipErrorInvalidResourceHandle},
      {CL_INVALID_CONTEXT, hipErrorInvalidContext},
      {CL_INVALID_EVENT_WAIT_LIST, hipErrorInvalidResourceHandle},
      {CL_INVALID_VALUE, hipErrorInvalidValue},
      {CL_MEM_COPY_OVERLAP, hipErrorInvalidValue},
      {CL_OUT_OF_RESOURCES, hipErrorOutOfMemory},
      {CL_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},
};

// Function converting cl_int into hipError_t based on provided map
template <typename FuncPtr>
inline hipError_t hip_convert_error(cl_int clStatus, FuncPtr func) {
  auto it = CL_HIP_ERROR_MAPS.find((void *)func);
  if (it != CL_HIP_ERROR_MAPS.end()) {
    auto &errorMap = it->second;
    auto errorIt = errorMap.find(clStatus);
    if (errorIt != errorMap.end()) {
      return errorIt->second;
    }
  }
  return hipErrorTbd;
}

#undef CHIPERR_CHECK_LOG_AND_THROW_TABLE
#define CHIPERR_CHECK_LOG_AND_THROW_TABLE(func, ...)                           \
  do {                                                                         \
    if (clStatus != CL_SUCCESS) {                                              \
      hipError_t err = hip_convert_error(clStatus, func);                      \
      if (err == hipErrorTbd) {                                                \
        std::cerr << "Error: Unmapped API or API Error Code encountered at "   \
                  << __FILE__ << ":" << __LINE__ << std::endl;                 \
        std::cerr << "API call: " << #func << std::endl;                       \
        std::cerr << "Error code: " << resultToString(clStatus) << std::endl;  \
        std::abort();                                                          \
      }                                                                        \
      std::string error_msg = std::string(resultToString(clStatus));           \
      std::string custom_msg = std::string(__VA_ARGS__);                       \
      std::string msg_ = error_msg + " " + custom_msg;                         \
      CHIPERR_LOG_AND_THROW(msg_, err);                                        \
    }                                                                          \
  } while (0)

#undef CHIPERR_CHECK_LOG_AND_THROW
#define CHIPERR_CHECK_LOG_AND_THROW(errtype, ...)                              \
  do {                                                                         \
    if (clStatus != CL_SUCCESS) {                                              \
      std::string error_msg = std::string(resultToString(clStatus));           \
      std::string custom_msg = std::string(__VA_ARGS__);                       \
      std::string msg_ = error_msg + " " + custom_msg;                         \
      CHIPERR_LOG_AND_THROW(msg_, errtype);                                    \
    }                                                                          \
  } while (0)

#endif // CL_HIP_ERROR_CONVERSION_HH