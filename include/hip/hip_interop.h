#ifndef HIP_INTEROP_H
#define HIP_INTEROP_H

#ifdef __cplusplus
#include <cstdint>
extern "C" {
#else
#include <stdint.h>
#endif

int hipGetBackendNativeHandles(uintptr_t Stream,
                               uintptr_t *NativeHandles,
                               int *NumHandles);

int hipInitFromNativeHandles(const uintptr_t *NativeHandles,
                             int NumHandles);

void* hipGetNativeEventFromHipEvent(void* HipEvent);

void* hipGetHipEventFromNativeEvent(void* NativeEvent);

#ifdef __cplusplus
}
#endif


#endif
