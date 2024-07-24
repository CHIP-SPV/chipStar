#include "common.h"
#include "privH.h"

// Mangling
#define MATH_MANGLE(N) OCML_MANGLE_F16(N)
#define MATH_PRIVATE(N) MANGLE3(__ocmlpriv,N,f16)