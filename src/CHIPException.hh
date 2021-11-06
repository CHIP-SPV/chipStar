#ifndef CHIP_EXCEPTION_HH
#define CHIP_EXCEPTION_HH

#include "hip/hip_runtime_api.h"

class CHIPError {
  std::string msg;
  hipError_t err;

 public:
  CHIPError(std::string msg_ = "", hipError_t err_ = hipErrorUnknown)
      : msg(msg_), err(err_) {}
  virtual hipError_t toHIPError() { return err; }

  std::string getMsgStr() { return msg.c_str(); }
  std::string getErrStr() { return std::string(hipGetErrorName(err)); }
};

#define CHIPERR_LOG_AND_THROW(msg, errtype)                                \
  do {                                                                     \
    logError("{} ({}) in {}:{}:{}\n", CHIPError(msg, errtype).getErrStr(), \
             CHIPError(msg, errtype).getMsgStr(), __FILE__, __LINE__,      \
             __func__);                                                    \
    throw CHIPError(msg, errtype);                                         \
  } while (0)

#define CHIPERR_CHECK_LOG_AND_THROW(status, success, errtype, ...) \
  do {                                                             \
    if (status != success) {                                       \
      std::string error_msg = std::string(resultToString(status)); \
      std::string custom_msg = std::string(__VA_ARGS__);           \
      std::string msg_ = error_msg + " " + custom_msg;             \
      CHIPERR_LOG_AND_THROW(msg_, errtype);                        \
    }                                                              \
  } while (0)

#define CHIP_TRY try {
#define CHIP_CATCH                \
  }                               \
  catch (CHIPError _status) {     \
    RETURN(_status.toHIPError()); \
  }

#define CHIP_CATCH_NO_RETURN                         \
  }                                                  \
  catch (CHIPError _status) {                        \
    logError(hipGetErrorName(_status.toHIPError())); \
  }

#endif  // ifdef guard