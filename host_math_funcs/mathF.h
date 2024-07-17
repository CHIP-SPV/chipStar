#include "common.h"
#include "privF.h"

// Mangling
#define MATH_MANGLE(N) OCML_MANGLE_F32(N)
#define MATH_PRIVATE(N) MANGLE3(__ocmlpriv,N,f32)