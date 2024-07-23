#ifndef ZE_HIP_ERROR_CONVERSION_HH
#define ZE_HIP_ERROR_CONVERSION_HH

#include "../../CHIPException.hh"
#include "ze_api.h"
#include <string>
#include <unordered_map>

using ze_hip_error_map_t = std::unordered_map<ze_result_t, hipError_t>;

// default table for ze to hip error conversion (empty map)
#define DEFAULT_ZE_HIP_ERROR_MAP                                               \
  ze_hip_error_map_t {}
inline hipError_t default_ze_hip_error_convert(ze_result_t Status) {
  switch (Status) {
  case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED:
    return hipErrorInvalidKernelFile;
  default:
    return hipErrorTbd;
  }
}

// function converting ze_result_t into hipError_t based on provided map
inline hipError_t hip_convert_error(ze_result_t Status,
                                    ze_hip_error_map_t Map) {
  if (Map.empty()) {
    return default_ze_hip_error_convert(Status);
  }
  if (Map.find(Status) != Map.end()) {
    return Map[Status];
  }
  return hipErrorTbd;
}

#define CHIPERR_CHECK_LOG_AND_THROW_TABLE(status, success,                     \
                                          ze_errors_hip_errors_map, ...)       \
  do {                                                                         \
    if (status != success) {                                                   \
      hipError_t err = hip_convert_error(status, ze_errors_hip_errors_map);    \
      std::string error_msg = std::string(resultToString(status));             \
      std::string custom_msg = std::string(__VA_ARGS__);                       \
      std::string msg_ = error_msg + " " + custom_msg;                         \
      CHIPERR_LOG_AND_THROW(msg_, err);                                        \
    }                                                                          \
  } while (0)

#endif // ifdef guard
