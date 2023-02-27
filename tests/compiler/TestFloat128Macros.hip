// SPIR-V target does not support __float128. Some libraries use them
// when one of the following macros in the below are defined.

#if defined(__HIP_DEVICE_COMPILE__)
#  if defined(__FLOAT128__)
#    error Falsely advertised __FLOAT128__ macro on the device side compilation.
#  endif
#  if defined(__SIZEOF_FLOAT128__)
#    error Falsely advertised __SIZEOF_FLOAT128__ macro on the device side compilation.
#  endif
#endif

// A library known to enable __float128 when __FLOAT128__ or
// __SIZEOF_FLOAT128__ is defined and -std=gnu++<something> is set.
// The device side compilation has known to fail in such cases.
#include <sstream>
