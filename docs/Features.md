
## Support status

This is a (non-exhaustive) list of features currently (un)supported by CHIP-SPV.

### Host side

#### Unsupported / unimplemented APIs

* hipRtc API
* hipGraph API
* hipIpc API
* hipModuleOccupancy API
* hipTexRef (texture reference) API
* surface object / reference APIs
* hipMemcpyPeer, hipMemcpyPeerAsync, hipMemRangeGetAttribute,
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
    Cubemap, Gather and Mipmapped textures are not supported

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
