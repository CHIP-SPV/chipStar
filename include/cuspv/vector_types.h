// this should be already included with default cuda_runtime.h
// cuda_runtime.h -> hip/hip_runtime.h -> hip/hip_vector_types.h

// in case it's not included indirectly by cuda_runtime.h, include it now
#include <hip/spirv_hip_vector_types.h>
