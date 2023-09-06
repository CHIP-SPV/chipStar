#include <hip/hip_runtime_api.h>
#if defined(__HIPCC__) || defined(__HIP__)
#  error "expected C++ compilation mode."
#endif