/*
 * Copyright (c) 2021-24 chipStar developers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef ZE_HIP_ERROR_CONVERSION_HH
#define ZE_HIP_ERROR_CONVERSION_HH

#include "../../CHIPException.hh"
#include "ze_api.h"
#include <unordered_map>

using ze_hip_error_map_t = std::unordered_map<ze_result_t, hipError_t>;

// Map of Level Zero API calls to error conversion maps
const std::unordered_map<void *, ze_hip_error_map_t> ZE_HIP_ERROR_MAPS = {
    {(void *)&zeInit,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_ENUMERATION, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_UNSUPPORTED_VERSION, hipErrorNotSupported}}},
    {(void *)&zeDriverGet,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, hipErrorInvalidValue}}},
    {(void *)&zeDeviceGet,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_INVALID_ENUMERATION, hipErrorInvalidValue}}},
    {(void *)&zeContextCreate,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_ENUMERATION, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory},
      {ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY, hipErrorOutOfMemory}}},
    {(void *)&zeCommandQueueCreate,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_ENUMERATION, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory},
      {ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY, hipErrorOutOfMemory}}},
    {(void *)&zeCommandListCreate,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_ENUMERATION, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory},
      {ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY, hipErrorOutOfMemory}}},
    {(void *)&zeMemAllocHost,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_INVALID_ENUMERATION, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_UNSUPPORTED_SIZE, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},
    {(void *)&zeMemAllocDevice,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_INVALID_ENUMERATION, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_UNSUPPORTED_SIZE, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY, hipErrorOutOfMemory}}},
    {(void *)&zeMemFree,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, hipErrorInvalidValue}}},
    {(void *)&zeDriverGetExtensionProperties,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, hipErrorInvalidValue}}},
    {(void *)&zeContextCreateEx,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_ENUMERATION, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory},
      {ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY, hipErrorOutOfMemory}}},
    {(void *)&zeContextDestroy,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized}}},
    {(void *)&zeMemAllocShared,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_ENUMERATION, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_UNSUPPORTED_SIZE, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory},
      {ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY, hipErrorOutOfMemory}}},
    {(void *)&zeContextMakeMemoryResident,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory},
      {ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY, hipErrorOutOfMemory}}},
    {(void *)&zeDeviceGetProperties,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, hipErrorInvalidValue}}},
    {(void *)&zeDeviceGetMemoryProperties,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, hipErrorInvalidValue}}},
    {(void *)&zeDeviceGetComputeProperties,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, hipErrorInvalidValue}}},
    {(void *)&zeDeviceGetCacheProperties,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, hipErrorInvalidValue}}},
    {(void *)&zeDeviceGetModuleProperties,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, hipErrorInvalidValue}}},
    {(void *)&zeDeviceGetImageProperties,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, hipErrorInvalidValue}}},
    {(void *)&zeSamplerCreate,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_INVALID_ENUMERATION, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory},
      {ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY, hipErrorOutOfMemory}}},
    {(void *)&zeEventCreate,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_INVALID_ENUMERATION, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory},
      {ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY, hipErrorOutOfMemory}}},
    {(void *)&zeEventDestroy,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle}}},
    {(void *)&zeEventPoolCreate,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_INVALID_ENUMERATION, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory},
      {ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY, hipErrorOutOfMemory}}},
    {(void *)&zeEventPoolDestroy,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle}}},
    {(void *)&zeImageCreate,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_INVALID_ENUMERATION, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory},
      {ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY, hipErrorOutOfMemory}}},
    {(void *)&zeModuleCreate,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_INVALID_ENUMERATION, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_INVALID_NATIVE_BINARY, hipErrorInvalidImage},
      {ZE_RESULT_ERROR_INVALID_SIZE, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory},
      {ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY, hipErrorOutOfMemory},
      {ZE_RESULT_ERROR_MODULE_BUILD_FAILURE, hipErrorInvalidImage}}},
    {(void *)&zeModuleBuildLogDestroy,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle}}},
    {(void *)&zeModuleGetKernelNames,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, hipErrorInvalidValue}}},
    {(void *)&zeKernelCreate,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_INVALID_KERNEL_NAME, hipErrorInvalidDeviceFunction},
      {ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED, hipErrorInvalidImage}}},
    {(void *)&zeKernelSetArgumentValue,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE, hipErrorInvalidValue}}},
    {(void *)&zeCommandListAppendWriteGlobalTimestamp,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT,
       hipErrorInvalidResourceHandle}}},
    {(void *)&zeCommandListAppendMemoryCopy,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT,
       hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_OVERLAPPING_REGIONS, hipErrorInvalidValue}}},
    {(void *)&zeCommandListAppendBarrier,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT,
       hipErrorInvalidResourceHandle}}},
    {(void *)&zeCommandListReset,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle}}},
    {(void *)&zeCommandListClose,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle}}},
    {(void *)&zeCommandQueueExecuteCommandLists,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_SIZE, hipErrorInvalidValue}}},
    {(void *)&zeFenceCreate,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, hipErrorOutOfMemory}}},
    {(void *)&zeFenceDestroy,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle}}},
    {(void *)&zeFenceQueryStatus,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_NOT_READY, hipErrorNotReady}}},
    {(void *)&zeEventHostReset,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle}}},
    {(void *)&zeEventQueryStatus,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_NOT_READY, hipErrorNotReady}}},
    {(void *)&zeCommandListAppendMemoryFill,
     {{ZE_RESULT_SUCCESS, hipSuccess},
      {ZE_RESULT_ERROR_UNINITIALIZED, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_DEVICE_LOST, hipErrorNotInitialized},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, hipErrorInvalidResourceHandle},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_INVALID_SIZE, hipErrorInvalidValue},
      {ZE_RESULT_ERROR_INVALID_ARGUMENT, hipErrorInvalidValue}}}};

// function converting ze_result_t into hipError_t based on provided map
template <typename FuncPtr>
inline hipError_t hip_convert_error(ze_result_t zeStatus, FuncPtr func) {
  auto it = ZE_HIP_ERROR_MAPS.find((void *)func);
  if (it != ZE_HIP_ERROR_MAPS.end()) {
    auto &errorMap = it->second;
    auto errorIt = errorMap.find(zeStatus);
    if (errorIt != errorMap.end()) {
      return errorIt->second;
    }
  }
  return hipErrorTbd;
}

#define CHIPERR_CHECK_LOG_AND_THROW_TABLE(func, ...)                           \
  do {                                                                         \
    if (zeStatus != ZE_RESULT_SUCCESS) {                                         \
      hipError_t err = hip_convert_error(zeStatus, func);                        \
      if (err == hipErrorTbd) {                                                \
        std::cerr << "Error: Unmapped API or API Error Code encountered at "   \
                  << __FILE__ << ":" << __LINE__ << std::endl;                 \
        std::abort();                                                          \
      }                                                                        \
      std::string error_msg = std::string(resultToString(zeStatus));             \
      std::string custom_msg = std::string(__VA_ARGS__);                       \
      std::string msg_ = error_msg + " " + custom_msg;                         \
      CHIPERR_LOG_AND_THROW(msg_, err);                                        \
    }                                                                          \
  } while (0)

#define CHIPERR_CHECK_LOG_AND_ABORT(...)                                       \
  do {                                                                         \
    if (zeStatus != ZE_RESULT_SUCCESS) {                                         \
      std::string error_msg = std::string(resultToString(zeStatus));             \
      std::string custom_msg = std::string(__VA_ARGS__);                       \
      std::string msg_ = error_msg + " " + custom_msg;                         \
      std::cout << msg_ << std::endl;                                          \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

#undef CHIPERR_CHECK_LOG_AND_THROW
#define CHIPERR_CHECK_LOG_AND_THROW(errtype, ...)                              \
  do {                                                                         \
    if (zeStatus != ZE_RESULT_SUCCESS) {                                         \
      std::string error_msg = std::string(resultToString(zeStatus));             \
      std::string custom_msg = std::string(__VA_ARGS__);                       \
      std::string msg_ = error_msg + " " + custom_msg;                         \
      CHIPERR_LOG_AND_THROW(msg_, errtype);                                    \
    }                                                                          \
  } while (0)

#endif // ZE_HIP_ERROR_CONVERSION_HH
