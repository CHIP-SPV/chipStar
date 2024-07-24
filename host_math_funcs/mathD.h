#include "common.h"
#include "privD.h"

// Mangling
#define MATH_MANGLE(N) OCML_MANGLE_F64(N)
#define MATH_PRIVATE(N) MANGLE3(__ocmlpriv,N,f64)