
## chipStar support matrix for host side

| Feature                       | HIP API # of funcs | # of impl in chipStar  |  chipStar missing / notes |
|-------------------------------|-----------|-----------|---------------------------|
| Device API                    |     23    |     17    | hipDeviceSetCacheConfig, hipDeviceGetCacheConfig, hipDeviceSetSharedMemConfig, hipDeviceGetSharedMemConfig, hipSetDeviceFlags, hipGetDeviceFlags |
| IPC API                       |     5     |     0     | hipIpcCloseMemHandle, hipIpcGetEventHandle, hipIpcGetMemHandle, hipIpcOpenEventHandle, hipIpcOpenMemHandle |
| Error API                     |     4     |     4     | |
| Stream API                    |     10    |     10    | |
| Event API                     |     7     |     7     | |
| Execution API                 |     10    |     7     | hipFuncSetSharedMemConfig, hipFuncSetCacheConfig, hipFuncGetAttributes only partially |
| Occupancy API                 |     7     |     0     | hipModuleOccupancyMaxPotentialBlockSize, hipModuleOccupancyMaxPotentialBlockSizeWithFlags, hipModuleOccupancyMaxActiveBlocksPerMultiprocessor, hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, hipOccupancyMaxActiveBlocksPerMultiprocessor, hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, hipOccupancyMaxPotentialBlockSize  |
| Mem Manag API                 |     47    |     42    | hipMemcpyPeer, hipMemcpyPeerAsync, hipMemPrefetchAsync, hipMemAdvise, hipMemRangeGetAttribute |
| Unified Addressing API        |     1     |     1     | |
| Peer Mem Access API           |     3     |     0     | hipDeviceCanAccessPeer, hipDeviceEnablePeerAccess, hipDeviceDisablePeerAccess |
| Texture Reference API (DEPR.) |     9     |     0     | ..all missing |
| Texture Object API            |     5     |     3     | hipGetTextureObjectResourceViewDesc, hipGetTextureObjectTextureDesc ; Texture Objects of 1D/2D type are supported; 3D, LOD, Grad, Cubemap, Gather and Mipmapped textures are not supported |
| Surface Object API            |     2     |     0     | hipCreateSurfaceObject, hipDestroySurfaceObject |
| Version API                   |     2     |     2     | |
| Graph API                     |     55    |     0     | ..all missing |
| Profiler API                  |     2     |     0     | hipProfilerStart, hipProfilerStop |
| Primary Context API           |     5     |     2     | hipDevicePrimaryCtxRelease, hipDevicePrimaryCtxRetain,  hipDevicePrimaryCtxSetFlags |
| Module API                    |     3     |     3     | hipModuleLaunchKernel has some caveats |
|                               |           |           | |
| Total                         |     200   |     98    | 49% |
