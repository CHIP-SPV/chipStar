#ifndef HIP_TO_CHIP_HH
#define HIP_TO_CHIP_HH

// Forward Declarations
class CHIPDevice;
class CHIPContext;
class CHIPModule;
class CHIPKernel;
class CHIPBackend;
class CHIPEvent;
class CHIPQueue;
class CHIPTexture;
/* implementation details */
typedef CHIPEvent *hipEvent_t;
typedef CHIPKernel *hipFunction_t;
typedef CHIPModule *hipModule_t;
typedef CHIPQueue *hipStream_t;
// typedef CHIPTexture *hipTextureObject_t;
typedef CHIPContext *hipCtx_t;
// TODO HIP tests assume this is int
// typedef CHIPDevice **hipDevice_t;
typedef int hipDevice_t;
typedef void *hipDeviceptr_t;

#endif