
## Support status

This is a (non-exhaustive) list of features currently (un)supported by CHIP-SPV.

### Host side

#### Unsupported / unimplemented APIs

* hipGraph API
* hipIpc API
* hipModuleOccupancy API
* hipTexRef (texture reference) API
* surface object / reference APIs
* hipMemcpyPeer, hipMemcpyPeerAsync, hipMemRangeGetAttribute, hipFuncGetAttributes,
  hipDeviceSetCacheConfig, hipDeviceGetCacheConfig,
  hipDeviceSetSharedMemConfig, hipDeviceGetSharedMemConfig,
  hipSetDeviceFlags,  hipGetDeviceFlags,
  hipDeviceCanAccessPeer, hipDeviceEnablePeerAccess,
  hipDeviceDisablePeerAccess, hipDeviceGetStreamPriorityRange,
  hipDevicePrimaryCtxRelease, hipDevicePrimaryCtxRetain,
  hipDevicePrimaryCtxSetFlags, hipMemPrefetchAsync, hipMemAdvise,
  hipModuleLoadData, hipModuleUnload, hipModuleLaunchKernel

#### partially supported
* Texture Objects of 1D/2D type are supported; 3D, LOD, Grad,
  Cubemap, Gather and Mipmapped textures are not supported.
* hiprtc: Referring global device variables, constants and texture
  references in the name expressions are not supported.

### Device side

#### Unsupported / unavailable
* __syncwarp(), __activemask()
* cooperative_groups.h header
* Address Space Predicate Functions, Address Space Conversion Functions
* alloca(), malloc(), free()
* Warp Reduce Functions, Warp Matrix Functions

#### Partially supported

* Warp functions (__all, __any, __ballot): only the non-sync versions are supported
* Shuffle functions (__shfl_{up,down,xor}): only the non-sync versions are supported
* assert(), __trap, __brkpt are not available but abort() is
* mathematical library: almost all single/double functions are available,
  but half-precision variants are not available

### Known issues

* warpSize might depend on the kernel instead of being a device constant

* hiprtc: Valid name expressions with a function pointer cast
  (e.g. '(void(*)(float *))akernel') fails with misleading
  messages. For example: "error: address of overloaded function
  'akernel' does not match required type ...". This issue prevents
  disambiguation of overloaded kernels.

* hiprtc: quoted strings are not preserved due to the way the hipcc
  handles arguments currently.  E.g. -DGREETING="Hello, World!" is
  treated as two argument ('-DGREETING=Hello,' and 'World!').
