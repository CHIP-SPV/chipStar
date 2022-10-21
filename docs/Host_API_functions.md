
# List of HIP runtime API functions supported by CHIP-SPV

## **1. Device Management**

|   **CUDA**                                                |   **HIP**                         |  **CHIP-SPV**|
|-----------------------------------------------------------|-----------------------------------|:----------------:|

## DEVICE API

| `cudaChooseDevice`                                        | `hipChooseDevice`                 | Y |
| `cudaDeviceGetAttribute`                                  | `hipDeviceGetAttribute`           | Y |
| `cudaDeviceGetByPCIBusId`                                 | `hipDeviceGetByPCIBusId`          | Y |
| `cudaDeviceGetCacheConfig`                                | `hipDeviceGetCacheConfig`         | Y |
| `cudaDeviceGetLimit`                                      | `hipDeviceGetLimit`               | Y* |

| `cudaDeviceGetPCIBusId`                                   | `hipDeviceGetPCIBusId`            | Y |
| `cudaDeviceGetSharedMemConfig`                            | `hipDeviceGetSharedMemConfig`     | N |
| `cudaDeviceGetStreamPriorityRange`                        | `hipDeviceGetStreamPriorityRange` | Y |
| `cudaDeviceReset`                                         | `hipDeviceReset`                  | Y |
| `cudaDeviceSetCacheConfig`                                | `hipDeviceSetCacheConfig`         | Y |

| `cudaDeviceSetLimit`                                      | `hipDeviceSetLimit`               | Y |
| `cudaDeviceSetSharedMemConfig`                            | `hipDeviceSetSharedMemConfig`     | N |
| `cudaDeviceSynchronize`                                   | `hipDeviceSynchronize`            | Y |
| `cudaGetDevice`                                           | `hipGetDevice`                    | Y |
| `cudaGetDeviceCount`                                      | `hipGetDeviceCount`               | Y |

| `cudaGetDeviceFlags`                                      | `hipGetDeviceFlags`               | N |
| `cudaGetDeviceProperties`                                 | `hipGetDeviceProperties`          | Y |
| `cudaSetDevice`                                           | `hipSetDevice`                    | Y |
| `cudaSetDeviceFlags`                                      | `hipSetDeviceFlags`               | N |
| `cudaThreadSynchronize`                                   | `hipDeviceSynchronize`            | Y |

| `cudaThreadExit`                                          | `hipDeviceReset`                  | Y |
| `cudaThreadGetCacheConfig`                                | `hipDeviceGetCacheConfig`         | N |
| `cudaThreadSetCacheConfig`                                | `hipDeviceSetCacheConfig`         | N |


############## IPC API

| `cudaIpcCloseMemHandle`                                   | `hipIpcCloseMemHandle`            | N |
| `cudaIpcGetEventHandle`                                   | `hipIpcGetEventHandle`            | N |
| `cudaIpcGetMemHandle`                                     | `hipIpcGetMemHandle`              | N |
| `cudaIpcOpenEventHandle`                                  | `hipIpcOpenEventHandle`           | N |
| `cudaIpcOpenMemHandle`                                    | `hipIpcOpenMemHandle`             | N |

## **3. Error Handling**

|   **CUDA**                                                |   **HIP**                     |  **CHIP-SPV**|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaGetErrorName`                                        | `hipGetErrorName`             | Y |
| `cudaGetErrorString`                                      | `hipGetErrorString`           | Y |
| `cudaGetLastError`                                        | `hipGetLastError`             | Y |
| `cudaPeekAtLastError`                                     | `hipPeekAtLastError`          | Y |

## **4. Stream Management**

|   **CUDA**                                                |   **HIP**                     |  **CHIP-SPV**   |
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaStreamAddCallback`                                   | `hipStreamAddCallback`        | Y |
| `cudaStreamCreate`                                        | `hipStreamCreate`             | Y |
| `cudaStreamCreateWithFlags`                               | `hipStreamCreateWithFlags`    | Y |
| `cudaStreamCreateWithPriority`                            | `hipStreamCreateWithPriority` | Y |
| `cudaStreamDestroy`                                       | `hipStreamDestroy`            | Y |

| `cudaStreamGetFlags`                                      | `hipStreamGetFlags`           | Y |
| `cudaStreamGetPriority`                                   | `hipStreamGetPriority`        | Y |
| `cudaStreamQuery`                                         | `hipStreamQuery`              | Y |
| `cudaStreamSynchronize`                                   | `hipStreamSynchronize`        | Y |
| `cudaStreamWaitEvent`                                     | `hipStreamWaitEvent`          | Y |

## **5. Event Management**

|   **CUDA**                                                |   **HIP**                     |  **CHIP-SPV**   |
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaEventCreate`                                         | `hipEventCreate`              | Y |
| `cudaEventCreateWithFlags`                                | `hipEventCreateWithFlags`     | Y |
| `cudaEventDestroy`                                        | `hipEventDestroy`             | Y |
| `cudaEventElapsedTime`                                    | `hipEventElapsedTime`         | Y |
| `cudaEventQuery`                                          | `hipEventQuery`               | Y |
| `cudaEventRecord`                                         | `hipEventRecord`              | Y |
| `cudaEventSynchronize`                                    | `hipEventSynchronize`         | Y |


## **7. Execution Control**

|   **CUDA**                                                |   **HIP**                             |  **CHIP-SPV**   |
|-----------------------------------------------------------|---------------------------------------|:----------------:|
| `cudaFuncGetAttributes`                                   |`hipFuncGetAttributes`                 | Y*|
| `cudaFuncSetAttribute`                                    |`hipFuncSetAttribute`                  | Y |
| `cudaFuncSetCacheConfig`                                  |`hipFuncSetCacheConfig`                | N |
| `cudaFuncSetSharedMemConfig`                              |`hipFuncSetSharedMemConfig`            | N |
| `cudaLaunchKernel`                                        |`hipLaunchKernel`                      | Y |

| `cudaLaunchCooperativeKernel`                             |`hipLaunchCooperativeKernel`           | N    |
| `cudaLaunchCooperativeKernelMultiDevice`                  |`hipLaunchCooperativeKernelMultiDevice`| N    |
| `cudaConfigureCall`                                       | `hipConfigureCall`                    | Y |
| `cudaLaunch`                                              | `hipLaunchByPtr`                      | Y |
| `cudaSetupArgument`                                       | `hipSetupArgument`                    | Y |

## **8. Occupancy**

|   **CUDA**                                                |   **HIP**                                             |  **CHIP-SPV**|
|-----------------------------------------------------------|-------------------------------------------------------|:----------------:|
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor`           | `hipOccupancyMaxActiveBlocksPerMultiprocessor`         | N |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`  | `hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`| N |
| ?                                                         | `hipModuleOccupancyMaxPotentialBlockSize`              | N |
| ?                                                         | `hipModuleOccupancyMaxPotentialBlockSizeWithFlags`     | N |
| ?                                                         | `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor`   | N |
| ?                                                         | `hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`  N |
| ?                                                         | `hipOccupancyMaxPotentialBlockSize`                    | N |


## **9. Memory Management**

|   **CUDA**                                                |   **HIP**                     |  **CHIP-SPV**|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaArrayGetInfo`                                        |                               | ? |
| `cudaFree`                                                | `hipFree`                     | Y |
| `cudaFreeArray`                                           | `hipFreeArray`                | Y |
| `cudaFreeHost`                                            | `hipHostFree`                 | Y |
| `cudaGetSymbolAddress`                                    | `hipGetSymbolAddress`         | Y |

| `cudaGetSymbolSize`                                       | `hipGetSymbolSize`            | Y |
| `cudaHostAlloc`                                           | `hipHostMalloc`               | Y |
| `cudaHostGetDevicePointer`                                | `hipHostGetDevicePointer`     | Y |
| `cudaHostGetFlags`                                        | `hipHostGetFlags`             | Y |
| `cudaHostRegister`                                        | `hipHostRegister`             | Y |

| `cudaHostUnregister`                                      | `hipHostUnregister`           | Y |
| `cudaMalloc`                                              | `hipMalloc`                   | Y |
| `cudaMalloc3D`                                            | `hipMalloc3D`                 | Y |
| `cudaMalloc3DArray`                                       | `hipMalloc3DArray`            | Y |
| `cudaMallocArray`                                         | `hipMallocArray`              | Y |

| `cudaMallocHost`                                          | `hipHostMalloc`               | Y |
| `cudaMallocManaged`                                       | `hipMallocManaged`            | Y |
| `cudaMemGetInfo`                                          | `hipMemGetInfo`               | Y |
| `cudaMemcpy`                                              | `hipMemcpy`                   | Y |
| `cudaMemcpy2D`                                            | `hipMemcpy2D`                 | Y |

| `cudaMemcpy2DAsync`                                       | `hipMemcpy2DAsync`            | Y |
| `cudaMemcpy2DFromArray`                                   | `hipMemcpy2DFromArray`        | Y |
| `cudaMemcpy2DFromArrayAsync`                              | `hipMemcpy2DFromArrayAsync`   | Y |
| `cudaMemcpy2DToArray`                                     | `hipMemcpy2DToArray`          | Y |
| `cudaMemcpy3D`                                            | `hipMemcpy3D`                 | Y |

| `cudaMemcpy3DAsync`                                       | `hipMemcpy3DAsync`            | Y |
| `cudaMemcpyAsync`                                         | `hipMemcpyAsync`              | Y |
| `cudaMemcpyFromSymbol`                                    | `hipMemcpyFromSymbol`         | Y |
| `cudaMemcpyFromSymbolAsync`                               | `hipMemcpyFromSymbolAsync`    | Y |
| `cudaMemcpyPeer`                                          | `hipMemcpyPeer`               | N |

| `cudaMemcpyPeerAsync`                                     | `hipMemcpyPeerAsync`          | N |
| `cudaMemcpyToSymbol`                                      | `hipMemcpyToSymbol`           | Y |
| `cudaMemcpyToSymbolAsync`                                 | `hipMemcpyToSymbolAsync`      | Y |
| `cudaMemset`                                              | `hipMemset`                   | Y |
| `cudaMemset2D`                                            | `hipMemset2D`                 | Y |

| `cudaMemset2DAsync`                                       | `hipMemset2DAsync`            | Y |
| `cudaMemset3D`                                            | `hipMemset3D`                 | Y |
| `cudaMemset3DAsync`                                       | `hipMemset3DAsync`            | Y |
| `cudaMemsetAsync`                                         | `hipMemsetAsync`              | Y |
| `make_cudaExtent`                                         | `make_hipExtent`              | Y |

| `make_cudaPitchedPtr`                                     | `make_hipPitchedPtr`          | Y |
| `make_cudaPos`                                            | `make_hipPos`                 | Y |
| `cudaMemcpyFromArray`                                     | `hipMemcpyFromArray`          | Y |
| `cudaMemcpyToArray`                                       | `hipMemcpyToArray`            | Y |

| ?                                                         | `hipMemPrefetchAsync`         | N |
| ?                                                         | `hipMemAdvise`                | N |
| ?                                                         | `hipMemRangeGetAttribute`     | N |

## **11. Unified Addressing**

|   **CUDA**                                                |   **HIP**                     |  **CHIP-SPV**|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaPointerGetAttributes`                                | `hipPointerGetAttributes`     | Y |

## **12. Peer Device Memory Access**

|   **CUDA**                                                |   **HIP**                     |  **CHIP-SPV**|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaDeviceCanAccessPeer`                                 | `hipDeviceCanAccessPeer`      | N |
| `cudaDeviceDisablePeerAccess`                             | `hipDeviceDisablePeerAccess`  | N |
| `cudaDeviceEnablePeerAccess`                              | `hipDeviceEnablePeerAccess`   | N |

## **24. Texture Reference Management [DEPRECATED]**

|   **CUDA**                                                |   **HIP**                        |  **CHIP-SPV**|
|-----------------------------------------------------------|----------------------------------|:----------------:|
| `cudaBindTexture`                                         | `hipBindTexture`                 | Y |
| `cudaBindTexture2D`                                       | `hipBindTexture2D`               | Y |
| `cudaBindTextureToArray`                                  | `hipBindTextureToArray`          | Y |
| `cudaBindTextureToMipmappedArray`                         | `hipBindTextureToMipmappedArray` | Y |
| `cudaCreateChannelDesc`                                   | `hipCreateChannelDesc`           | Y |

| `cudaGetChannelDesc`                                      | `hipGetChannelDesc`              | Y |
| `cudaGetTextureAlignmentOffset`                           | `hipGetTextureAlignmentOffset`   | Y |
| `cudaGetTextureReference`                                 | `hipGetTextureReference`         | Y |
| `cudaUnbindTexture`                                       | `hipUnbindTexture`               | Y |

## **26. Texture Object Management**

|   **CUDA**                                                |   **HIP**                            |  **CHIP-SPV**|
|-----------------------------------------------------------|--------------------------------------|:----------------:|
| `cudaCreateTextureObject`                                 |`hipCreateTextureObject`              | Y |
| `cudaDestroyTextureObject`                                |`hipDestroyTextureObject`             | Y |
| `cudaGetTextureObjectResourceDesc`                        |`hipGetTextureObjectResourceDesc`     | Y |
| `cudaGetTextureObjectResourceViewDesc`                    |`hipGetTextureObjectResourceViewDesc` | N |
| `cudaGetTextureObjectTextureDesc`                         |`hipGetTextureObjectTextureDesc`      | N |

## **27. Surface Object Management**

|   **CUDA**                                                |   **HIP**                     |  **CHIP-SPV**|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaCreateSurfaceObject`                                 | `hipCreateSurfaceObject`      | N |
| `cudaDestroySurfaceObject`                                | `hipDestroySurfaceObject`     | N |

## **28. Version Management**

|   **CUDA**                                                |   **HIP**                     |  **CHIP-SPV**|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaDriverGetVersion`                                    | `hipDriverGetVersion`         | Y |
| `cudaRuntimeGetVersion`                                   | `hipRuntimeGetVersion`        | Y |

## **29. Graph Management**

|   **CUDA**                                                |   **HIP**                     |  **CHIP-SPV**|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaGraphAddChildGraphNode`                              |                               | N                |
| `cudaGraphAddDependencies`                                |                               | N                |
| `cudaGraphAddEmptyNode`                                   |                               | N                |
| `cudaGraphAddHostNode`                                    |                               | N                |
| `cudaGraphAddKernelNode`                                  |                               | N                |

| `cudaGraphAddMemcpyNode`                                  |                               | N                |
| `cudaGraphAddMemsetNode`                                  |                               | N                |
| `cudaGraphChildGraphNodeGetGraph`                         |                               | N                |
| `cudaGraphClone`                                          |                               | N                |
| `cudaGraphCreate`                                         |                               | N                |

| `cudaGraphDestroy`                                        |                               | N                |
| `cudaGraphDestroyNode`                                    |                               | N                |
| `cudaGraphExecDestroy`                                    |                               | N                |
| `cudaGraphGetEdges`                                       |                               | N                |
| `cudaGraphGetNodes`                                       |                               | N                |

| `cudaGraphGetRootNodes`                                   |                               | N                |
| `cudaGraphHostNodeGetParams`                              |                               | N                |
| `cudaGraphHostNodeSetParams`                              |                               | N                |
| `cudaGraphInstantiate`                                    |                               | N                |
| `cudaGraphExecKernelNodeSetParams`                        |                               | N                |

| `cudaGraphExecMemcpyNodeSetParams`                        |                               | N                |
| `cudaGraphExecMemsetNodeSetParams`                        |                               | N                |
| `cudaGraphExecHostNodeSetParams`                          |                               | N                |
| `cudaGraphExecUpdate`                                     |                               | N                |
| `cudaGraphKernelNodeGetParams`                            |                               | N                |

| `cudaGraphKernelNodeSetParams`                            |                               | N                |
| `cudaGraphLaunch`                                         |                               | N                |
| `cudaGraphMemcpyNodeGetParams`                            |                               | N                |
| `cudaGraphMemcpyNodeSetParams`                            |                               | N                |
| `cudaGraphMemsetNodeGetParams`                            |                               | N                |

| `cudaGraphMemsetNodeSetParams`                            |                               | N                |
| `cudaGraphNodeFindInClone`                                |                               | N                |
| `cudaGraphNodeGetDependencies`                            |                               | N                |
| `cudaGraphNodeGetDependentNodes`                          |                               | N                |
| `cudaGraphNodeGetType`                                    |                               | N                |

| `cudaGraphRemoveDependencies`                             |                               | N                |

... INCOMPLETE, there are 55 Graph API functions in CHIPBindings.cc


## **32. Profiler Control**

|   **CUDA**                                                |   **HIP**                     |  **CHIP-SPV**|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| `cudaProfilerStart`                                       | `hipProfilerStart`            | N |
| `cudaProfilerStop`                                        | `hipProfilerStop`             | N |


#### Primary Context API

|   **CUDA**                                                |   **HIP**                     |  **CHIP-SPV**|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| ?                                                         | `hipDevicePrimaryCtxGetState` | Y        |
| ?                                                         | `hipDevicePrimaryCtxRelease`  | N        |
| ?                                                         | `hipDevicePrimaryCtxRetain`   | N        |
| ?                                                         | `hipDevicePrimaryCtxReset`    | Y        |
| ?                                                         | `hipDevicePrimaryCtxSetFlags` | N        |


#### Module API

|   **CUDA**                                                |   **HIP**                     |  **CHIP-SPV**|
|-----------------------------------------------------------|-------------------------------|:----------------:|
| ?                                                         | `hipModuleLoadData`           | Y        |
| ?                                                         | `hipModuleUnload`             | Y        |
| ?                                                         | `hipModuleLaunchKernel`       | Y*       |

* partially supported (with some caveats)
