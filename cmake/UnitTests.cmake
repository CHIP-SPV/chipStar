# custom target to avoid tests that are known to fail
list(APPEND FAILING_FOR_ALL " ") 
list(APPEND CPU_OPENCL_FAILED_TESTS " ") 
list(APPEND DGPU_OPENCL_FAILED_TESTS " ") 
list(APPEND IGPU_OPENCL_FAILED_TESTS " ") 
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS " ")
list(APPEND IGPU_LEVEL0_RCL_TESTS " ")
list(APPEND IGPU_LEVEL0_ICL_TESTS " ")
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS " ") 
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS " ") 
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS " ") 
list(APPEND CPU_POCL_FAILED_TESTS " ") 
list(APPEND GPU_POCL_FAILED_TESTS " ")  # TODO
list(APPEND NON_PARALLEL_TESTS " ")

list(APPEND NON_PARALLEL_TESTS "hipMultiThreadAddCallback") # added after adding MKL back into testing
list(APPEND NON_PARALLEL_TESTS "TestLargeGlobalVar")
list(APPEND NON_PARALLEL_TESTS "cuda-asyncAPI")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMemcpy_Negative")
list(APPEND NON_PARALLEL_TESTS "firstTouch")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMemcpy_HalfMemCopy")
list(APPEND NON_PARALLEL_TESTS "Unit_hipStreamBeginCapture_ColligatedStrmCapture_defaultflag")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMemcpyWithStream_TestkindDtoH")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMemcpyWithStream_TestkindDefault")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMemsetFunctional_ZeroValue_2D")
list(APPEND NON_PARALLEL_TESTS "Unit_hipHostMalloc_NonCoherent")
list(APPEND NON_PARALLEL_TESTS "Unit_hipStreamAddCallback_WithCreatedStream")
list(APPEND NON_PARALLEL_TESTS "cuda-sortnet")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMemset3DAsync_SeekSetArrayPortion")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMemcpyToFromSymbol_SyncAndAsync")
list(APPEND NON_PARALLEL_TESTS "MatrixMultiply")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMemcpy2DFromArray_PinnedMemSameGPU")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMemset3D_SeekSetArrayPortion")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMemsetFunctional_PartialSet_2D")
list(APPEND NON_PARALLEL_TESTS "VecAdd")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMallocPitch_ValidatePitch")
list(APPEND NON_PARALLEL_TESTS "Unit_hipHostMalloc_CoherentAccess")
list(APPEND NON_PARALLEL_TESTS "cuda-qrng")
list(APPEND NON_PARALLEL_TESTS "cuda-reduction")
list(APPEND NON_PARALLEL_TESTS "TestLargeKernelArgLists")
list(APPEND NON_PARALLEL_TESTS "Unit_hipStreamAddCallback_WithDefaultStream")
list(APPEND NON_PARALLEL_TESTS "TestWholeProgramCompilation")
list(APPEND NON_PARALLEL_TESTS "hip_async_binomial")
list(APPEND NON_PARALLEL_TESTS "BinomialOption")
list(APPEND NON_PARALLEL_TESTS "shuffles")
list(APPEND NON_PARALLEL_TESTS "cuda-convolutionSeparable")
list(APPEND NON_PARALLEL_TESTS "cuda-binomialoptions")
list(APPEND NON_PARALLEL_TESTS "clock")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMemcpyWithStream_TestwithTwoStream")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMemcpyWithStream_TestDtoDonSameDevice")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMemcpy_H2H-H2D-D2H-H2PinMem - int")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMemcpy_H2H-H2D-D2H-H2PinMem - float")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMemcpy_H2H-H2D-D2H-H2PinMem - double")
list(APPEND NON_PARALLEL_TESTS "broadcast2")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMemsetFunctional_SmallSize_3D")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMemcpy_KernelLaunch - int")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMemcpy_KernelLaunch - float")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMemcpy_KernelLaunch - double")
list(APPEND NON_PARALLEL_TESTS "fp16")
list(APPEND NON_PARALLEL_TESTS "SimpleConvolution")
list(APPEND NON_PARALLEL_TESTS "Unit_hipHostMalloc_Basic")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMalloc_LoopRegressionAllocFreeCycles")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMultiThreadStreams1_AsyncAsync")
list(APPEND NON_PARALLEL_TESTS "cuda-scan")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMemset3DAsync_ConcurrencyMthread")
list(APPEND NON_PARALLEL_TESTS "cuda-bandwidthTest")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMemsetAsync_QueueJobsMultithreaded")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMemset2DAsync_MultiThread")
list(APPEND NON_PARALLEL_TESTS "DCT")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMemsetFunctional_ZeroSize_3D")
list(APPEND NON_PARALLEL_TESTS "cuda-matrixMul")
list(APPEND NON_PARALLEL_TESTS "cuda-FDTD3d")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMultiThreadStreams1_AsyncSync")
list(APPEND NON_PARALLEL_TESTS "FastWalshTransform")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMultiStream_sameDevice")
list(APPEND NON_PARALLEL_TESTS "Unit_hipStreamCreate_MultistreamBasicFunctionalities")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMemcpy_MultiThreadWithSerialization")
list(APPEND NON_PARALLEL_TESTS "dwtHaar1D")
list(APPEND NON_PARALLEL_TESTS "Unit_hipHostRegister_Memcpy - int")
list(APPEND NON_PARALLEL_TESTS "Unit_hipHostRegister_Memcpy - float")
list(APPEND NON_PARALLEL_TESTS "Unit_hipHostRegister_Memcpy - double")
list(APPEND NON_PARALLEL_TESTS "TestStlFunctionsDouble")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMemset_SetMemoryWithOffset")
list(APPEND NON_PARALLEL_TESTS "Unit_hipMemsetAsync_SetMemoryWithOffset")
list(APPEND NON_PARALLEL_TESTS "BitonicSort")
list(APPEND NON_PARALLEL_TESTS "FloydWarshall")

# This test gets enabled only if LLVM' FileCheck tool is found in PATH.
# It fails with "error: cannot find ROCm device library;
#  provide its path via '--rocm-path' or '--rocm-device-lib-path', or pass
#  '-nogpulib' to build without ROCm device library"
list(APPEND FAILING_FOR_ALL "abort") # causes hung processes on i915 driver
list(APPEND FAILING_FOR_ALL "abort2") # causes hung processes on i915 driver
list(APPEND FAILING_FOR_ALL "TestAssert") # causes hung processes on i915 driver
list(APPEND FAILING_FOR_ALL "TestAssertFail")# causes hung processes on i915 driver 
list(APPEND FAILING_FOR_ALL "Unit_hipGraphDestroyNode_DestroyDependencyNode") # SEGFAULT
list(APPEND FAILING_FOR_ALL "Unit_hipGraphAddHostNode_ClonedGraphwithHostNode") # Correctness
list(APPEND FAILING_FOR_ALL "Unit_hipStreamPerThread_Basic") # SyncQueues refactor
list(APPEND FAILING_FOR_ALL "Unit_hipStreamAddCallback_MultipleThreads") # Timeout
list(APPEND FAILING_FOR_ALL "Unit_hipMultiStream_multimeDevice") # Timeout on OpenCL cpu but needs investigating
list(APPEND FAILING_FOR_ALL "Unit_hipGraphAddEventWaitNode_MultipleRun") # Failed for level0 dgpu imm
list(APPEND FAILING_FOR_ALL "Unit_hipCreateTextureObject_ArgValidation") # Subprocess aborted
list(APPEND FAILING_FOR_ALL "Unit_hipCreateTextureObject_LinearResource") # Subprocess aborted
list(APPEND FAILING_FOR_ALL "Unit_hipCreateTextureObject_Pitch2DResource") # Subprocess aborted
list(APPEND FAILING_FOR_ALL "Unit_hipGetChannelDesc_CreateAndGet") # Failed
list(APPEND FAILING_FOR_ALL "Unit_hipTextureObj1DCheckRGBAModes - array") # Failed
list(APPEND FAILING_FOR_ALL "Unit_hipTextureObj1DCheckSRGBAModes - array") # Failed
list(APPEND FAILING_FOR_ALL "Unit_hipTextureObj1DCheckRGBAModes - buffer") # Failed
list(APPEND FAILING_FOR_ALL "Unit_hipTextureObj1DCheckSRGBAModes - buffer") # Failed
list(APPEND FAILING_FOR_ALL "Unit_hipTextureObj2DCheckRGBAModes") # Failed
list(APPEND FAILING_FOR_ALL "Unit_hipTextureObj2DCheckSRGBAModes") # Failed
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_MaxTexture_Default - uint") # Failed
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_MaxTexture_Default - int4") # Failed
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_MaxTexture_Default - ushort") # Failed
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_MaxTexture_Default - short2") # Failed
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_MaxTexture_Default - char") # Failed
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_MaxTexture_Default - char4") # Failed
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_MaxTexture_Default - float2") # Failed
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_MaxTexture_Default - float4") # Failed
list(APPEND FAILING_FOR_ALL "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - char") # Failed
list(APPEND FAILING_FOR_ALL "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - uchar2") # Failed
list(APPEND FAILING_FOR_ALL "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - short4") # Failed
list(APPEND FAILING_FOR_ALL "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - float2") # Failed
list(APPEND FAILING_FOR_ALL "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - float4") # Failed
list(APPEND FAILING_FOR_ALL "Unit_HMM_OverSubscriptionTst") # Seems AMD-specific, crashes the driver
list(APPEND FAILING_FOR_ALL "Unit_hipMallocPitch_KernelLaunch - int") # Correctess
list(APPEND FAILING_FOR_ALL "Unit_hipMallocPitch_KernelLaunch - float") # Correctness
list(APPEND FAILING_FOR_ALL "Unit_hipMallocPitch_KernelLaunch - double") # Correctness
list(APPEND FAILING_FOR_ALL "constant_fold_lgamma_r") # Unknown
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2float_rd_float") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2float_rn_float") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2float_ru_float") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2float_rz_float") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2hiint_int") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2int_rd_int") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2int_rn_int") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2int_ru_int") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2int_rz_int") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2ll_rd_longlong") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2ll_rn_longlong") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2ll_ru_longlong") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2ll_rz_longlong") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2loint_int") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2uint_rd_unsigned") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2uint_rn_unsigned") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2uint_ru_unsigned") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2uint_rz_unsigned") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2ull_rd_ulonglong") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2ull_rn_ulonglong") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2ull_ru_ulonglong") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2ull_rz_ulonglong") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2int_rd_int") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2int_rn_int") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2int_ru_int") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2int_rz_int") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2ll_rd_longlong") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2ll_rn_longlong") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2ll_ru_longlong") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2ll_rz_longlong") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2uint_rd_unsigned") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2uint_rn_unsigned") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2uint_ru_unsigned") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2uint_rz_unsigned") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2ull_rd_ulonglong") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2ull_rn_ulonglong") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2ull_ru_ulonglong") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2ull_rz_ulonglong") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___hiloint2double_double") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___int2double_rn_double") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___int2float_rd_float") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___int2float_rn_float") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___int2float_ru_float") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___int2float_rz_float") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ll2double_rd_double") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ll2double_rn_double") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ll2double_ru_double") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ll2double_rz_double") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ll2float_rd_float") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ll2float_rn_float") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ll2float_ru_float") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ll2float_rz_float") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___uint2double_rn_double") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___uint2float_rd_float") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___uint2float_rn_float") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___uint2float_ru_float") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___uint2float_rz_float") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ull2double_rd_double") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ull2double_rn_double") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ull2double_ru_double") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ull2double_rz_double") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ull2float_rd_float") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ull2float_rn_float") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ull2float_ru_float") # Unimplemented
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ull2float_rz_float") # Unimplemented
list(APPEND FAILING_FOR_ALL "hipStreamSemantics") # SEGFAULT - likely due to main thread exiting without calling join
# Not included in any target because no driver (so far) reports support
# for indirect calls. Despite this, this test is known to pass on Intel
# OpenCL CPU & GPU and Intel Level Zero (however, your mileage may vary).
list(APPEND FAILING_FOR_ALL "TestIndirectCall")

# CPU OpenCL Unit Test Failures
list(APPEND CPU_OPENCL_FAILED_TESTS "cuda-binomialoptions") # Timeout
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMultiThreadStreams2") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "hipStreamSemantics") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "deviceMallocCompile") # Unimplemented
list(APPEND CPU_OPENCL_FAILED_TESTS "fp16") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "2d_shuffle") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "hipDynamicShared2") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "unroll") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "hipConstantTestDeviceSymbol") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "cuda-simpleCallback") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "cuda-qrng") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddEmptyNode_NegTest") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddDependencies_NegTest") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddEventRecordNode_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddEventWaitNode_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraph_BasicFunctional") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNode_BasicFunctional") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphClone_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphInstantiateWithFlags_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphInstantiateWithFlags_DependencyGraph") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddHostNode_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphChildGraphNodeGetGraph_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphNodeFindInClone_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecHostNodeSetParams_ClonedGraphwithHostNode") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecHostNodeSetParams_BasicFunc") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemsetNodeSetParams_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemsetNodeSetParams_Functional") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsToSymbol_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphDestroyNode_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphGetNodes_Functional") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphGetNodes_CapturedStream") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphGetNodes_ParamValidation") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphGetRootNodes_Functional") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphGetRootNodes_CapturedStream") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphGetRootNodes_ParamValidation") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphHostNodeSetParams_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNode1D_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_OrgGraphAsChildGraph") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_CloneChildGraph") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_MultipleChildNodes") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_SingleChildNode") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphNodeGetType_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams1D_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams1D_Functional") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphGetEdges_Functionality") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphGetEdges_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphRemoveDependencies_Func_StrmCapture") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphRemoveDependencies_ChangeComputeFunc") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphRemoveDependencies_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphInstantiate_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecUpdate_Negative_Basic") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecUpdate_Negative_TypeChange") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecUpdate_Functional") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecEventRecordNodeSetEvent_Functional") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecEventRecordNodeSetEvent_VerifyEventNotChanged") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecEventRecordNodeSetEvent_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphEventWaitNodeSetEvent_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemsetNodeGetParams_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemsetNodeSetParams_InvalidParams") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsFromSymbol_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsFromSymbol_Functional") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphEventRecordNodeGetEvent_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphEventRecordNodeSetEvent_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphEventWaitNodeGetEvent_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams_Functional") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_BasicFunctional") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_hipStreamPerThread") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Basic") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_defaultflag") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_blockingflag") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_diffflags") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_diffprio") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_multiplestrms") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_ColligatedStrmCapture_func") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Global") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_ThreadLocal") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Relaxed") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_CapturingFromWithinStrms") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_DetectingInvalidCapture") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_CapturingMultGraphsFrom1Strm") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_EndingCapturewhenCaptureInProgress") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_nestedStreamCapture") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_streamReuse") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_captureComplexGraph") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_captureEmptyStreams") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamIsCapturing_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamIsCapturing_Functional_Basic") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamIsCapturing_Functional") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamIsCapturing_hipStreamPerThread") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_BasicFunctional") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_hipStreamPerThread") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_UniqueID") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_ArgValidation") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamEndCapture_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamEndCapture_Thread_Negative") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsFromSymbol_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecEventWaitNodeSetEvent_SetAndVerifyMemory") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecEventWaitNodeSetEvent_VerifyEventNotChanged") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecEventWaitNodeSetEvent_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemsetNode_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddKernelNode_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeGetParams_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeGetParams_Functional") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParams_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParams_Functional") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphKernelNodeGetParams_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphKernelNodeSetParams_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphKernelNodeSetParams_Functional") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphKernelNodeGetSetParams_Functional") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecKernelNodeSetParams_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecKernelNodeSetParams_Functional") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphLaunch_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParams1D_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsToSymbol_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsToSymbol_Functional") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphNodeGetDependentNodes_Functional") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphNodeGetDependentNodes_ParamValidation") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphNodeGetDependencies_Functional") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphNodeGetDependencies_ParamValidation") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphHostNodeGetParams_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecChildGraphNodeSetParams_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecChildGraphNodeSetParams_BasicFunc") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecChildGraphNodeSetParams_ChildTopology") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_BasicFunctional") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_hipStreamPerThread") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_UniqueID") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_ParamValidation") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_1") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_2") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_3") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_4") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipUserObjectCreate_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipUserObjectRelease_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipUserObjectRetain_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipUserObj_Negative_Test") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphRetainUserObject_Functional_1") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphRetainUserObject_Functional_2") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphRetainUserObject_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphReleaseUserObject_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphRetainUserObject_Negative_Basic") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphRetainUserObject_Negative_Null_Object") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - int") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - float") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - double") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3D_ValidatePitch") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemAllocPitch_ValidatePitch") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3D_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemAllocPitch_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipHostGetFlags_flagCombos") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipHostGetFlags_DifferentThreads") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipHostGetFlags_InvalidArgs") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipHostGetDevicePointer_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocManaged_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemset_Negative_InvalidPtr") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemset_Negative_OutOfBoundsSize") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemset_Negative_OutOfBoundsPtr") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemset3D_Negative_InvalidPtr") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemset3D_Negative_ModifiedPtr") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemset3D_Negative_InvalidSizes") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemset3D_Negative_OutOfBounds") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemPrefetchAsync_NonPageSz") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipPtrGetAttribute_Simple") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_Basic") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_BasicAlloc") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_BasicTrim") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_BasicReuse") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_Opportunistic") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_Default") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMalloc_ArgumentValidation") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipHostGetDevicePointer_NullCheck") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_PartialSet_1D") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_DiffSizes") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MultiThread") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_DifferentChannelSizes") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_NullArrayPtr") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_NullDescPtr") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_BadFlags") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_8bitFloat - float") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_8bitFloat - float2") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_8bitFloat - float4") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_BadNumberOfBits") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_3ChannelElement") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_ChannelAfterZeroChannel") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_InvalidChannelFormat") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_NumericLimit") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_DiffSizes") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_MultiThread") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_NullArrayPtr") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_NullDescPtr") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_ZeroHeight") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_InvalidFlags") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_BadChannelLayout") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_8BitFloat") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_DifferentChannelSizes") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_BadChannelSize") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_NumericLimit") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3D_MultipleDataTypes - uint8_t") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3D_MultipleDataTypes - int") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3D_MultipleDataTypes - float") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3D_HosttoDevice") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3D_ExtentValidation") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - uint8_t") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - int") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - float") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_HosttoDevice") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_ExtentValidation") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipPointerGetAttribute_MemoryTypes") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipPointerGetAttribute_KernelUpdation") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipPointerGetAttribute_BufferID") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipPointerGetAttribute_MappedMem") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipPointerGetAttribute_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDrvPtrGetAttributes_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDrvPtrGetAttributes_Functional") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemGetInfo_DifferentMallocSmall") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemGetInfo_ParaSmall") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemGetInfo_ParaNonDiv") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemGetInfo_ParaMultiSmall") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemGetInfo_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTDev - char") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTDev - int") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTDev - float2") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTDev - float4") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTHost - char") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTHost - int") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTHost - float2") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTHost - float4") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTArray - char") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTArray - int") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTArray - float2") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTArray - float4") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemset2DASyncMulti") # Timeout
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemset3DASyncMulti") # Timeout
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetFlags_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamDestroy_Negative_DoubleDestroy") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamDestroy_Negative_NullStream") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamSynchronize_UninitializedStream") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamAddCallback_StrmSyncTiming") # Timeout
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipEvent") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipEventIpc") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipEventSynchronize_Default_Positive") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipEventSynchronize_NoEventRecord_Positive") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetPCIBusId_Negative_PartialFill") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetCacheConfig_Positive_Default") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetCacheConfig_Positive_Basic") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetCacheConfig_Positive_Threaded") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_HipDeviceGetCacheConfig_Negative_Parameters") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDeviceSynchronize_Positive_Nullstream") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDeviceSynchronize_Functional") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDeviceTotalMem_NegTst") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGetSetDeviceFlags_NullptrFlag") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGetSetDeviceFlags_InvalidFlag") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGetSetDeviceFlags_ValidFlag") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGetSetDeviceFlags_SetThenGet") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGetSetDeviceFlags_Threaded") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGetDeviceFlags_Positive_Context") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGetSetDevice_MultiThreaded") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipSetGetDevice_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetUuid_Positive") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetUuid_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetDefaultMemPool_Positive_Basic") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetDefaultMemPool_Negative_Parameters") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDeviceSetLimit_SetGet") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Positive_Default") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Positive_Basic") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Positive_Threaded") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Negative_Parameters") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDeviceReset_Positive_Basic") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDeviceReset_Positive_Threaded") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDeviceSetMemPool_Positive_Basic") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDeviceSetMemPool_Negative_Parameters") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetMemPool_Positive_Default") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetMemPool_Positive_Basic") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetMemPool_Positive_Threaded") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetMemPool_Negative_Parameters") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDriverGetVersion_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipIpcOpenMemHandle_Negative_Open_In_Creating_Process") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipIpcOpenMemHandle_Negative_Open_In_Two_Contexts_Same_Device") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipIpcGetMemHandle_Positive_Unique_Handles_Separate_Allocations") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipIpcGetMemHandle_Positive_Unique_Handles_Reused_Memory") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipIpcGetMemHandle_Negative_Handle_For_Freed_Memory") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipIpcGetMemHandle_Negative_Out_Of_Bound_Pointer") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipIpcCloseMemHandle_Positive_Reference_Counting") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipIpcCloseMemHandle_Negative_Close_In_Originating_Process") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_printf_flags") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_printf_specifier") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipLaunchBounds_With_maxThreadsPerBlock_Check") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipLaunchBounds_With_maxThreadsPerBlock_blocksPerCU_Check") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipClassKernel_Value") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGetLastError_Positive_Basic") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGetLastError_Positive_Threaded") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipPeekAtLastError_Positive_Basic") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipPeekAtLastError_Positive_Threaded") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "fp16") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipEvent") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "syncthreadsExitedThreads") # Timeout
list(APPEND CPU_OPENCL_FAILED_TESTS "hipMultiThreadAddCallback") # SEGFAULT

# iGPU OpenCL Unit Test Failures
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMultiThreadStreams1_AsyncSame") # invalid free()
list(APPEND IGPU_OPENCL_FAILED_TESTS "TestStlFunctionsDouble") # Runs out of resoruces with -j16?
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamPerThread_MultiThread") 
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamPerThread_DeviceReset_1") 
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_PartialSet_3D") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemset3DAsync_MemsetWithExtent") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemset2DAsync_MultiThread") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "hipStreamSemantics") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "deviceMallocCompile") # Unimplemented
list(APPEND IGPU_OPENCL_FAILED_TESTS "cuda-simpleCallback") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddEmptyNode_NegTest") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddDependencies_NegTest") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddEventRecordNode_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddEventWaitNode_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraph_BasicFunctional") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNode_BasicFunctional") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphClone_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphInstantiateWithFlags_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphInstantiateWithFlags_DependencyGraph") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddHostNode_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphChildGraphNodeGetGraph_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphNodeFindInClone_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecHostNodeSetParams_ClonedGraphwithHostNode") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecHostNodeSetParams_BasicFunc") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemsetNodeSetParams_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemsetNodeSetParams_Functional") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsToSymbol_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphDestroyNode_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphGetNodes_Functional") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphGetNodes_CapturedStream") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphGetNodes_ParamValidation") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphGetRootNodes_Functional") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphGetRootNodes_CapturedStream") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphGetRootNodes_ParamValidation") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphHostNodeSetParams_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNode1D_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_OrgGraphAsChildGraph") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_CloneChildGraph") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_MultipleChildNodes") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_SingleChildNode") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphNodeGetType_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams1D_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams1D_Functional") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphGetEdges_Functionality") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphGetEdges_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphRemoveDependencies_Func_StrmCapture") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphRemoveDependencies_ChangeComputeFunc") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphRemoveDependencies_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphInstantiate_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecUpdate_Negative_Basic") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecUpdate_Negative_TypeChange") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecUpdate_Functional") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecEventRecordNodeSetEvent_Functional") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecEventRecordNodeSetEvent_VerifyEventNotChanged") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecEventRecordNodeSetEvent_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphEventWaitNodeSetEvent_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemsetNodeGetParams_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemsetNodeSetParams_InvalidParams") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsFromSymbol_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsFromSymbol_Functional") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphEventRecordNodeGetEvent_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphEventRecordNodeSetEvent_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphEventWaitNodeGetEvent_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams_Functional") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_BasicFunctional") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_hipStreamPerThread") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Basic") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_defaultflag") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_blockingflag") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_diffflags") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_diffprio") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_multiplestrms") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_ColligatedStrmCapture_func") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Global") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_ThreadLocal") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Relaxed") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_CapturingFromWithinStrms") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_DetectingInvalidCapture") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_CapturingMultGraphsFrom1Strm") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_EndingCapturewhenCaptureInProgress") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_nestedStreamCapture") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_streamReuse") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_captureComplexGraph") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_captureEmptyStreams") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamIsCapturing_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamIsCapturing_Functional_Basic") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamIsCapturing_Functional") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamIsCapturing_hipStreamPerThread") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_BasicFunctional") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_hipStreamPerThread") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_UniqueID") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_ArgValidation") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamEndCapture_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamEndCapture_Thread_Negative") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsFromSymbol_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecEventWaitNodeSetEvent_SetAndVerifyMemory") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecEventWaitNodeSetEvent_VerifyEventNotChanged") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecEventWaitNodeSetEvent_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemsetNode_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddKernelNode_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeGetParams_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeGetParams_Functional") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParams_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParams_Functional") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphKernelNodeGetParams_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphKernelNodeSetParams_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphKernelNodeSetParams_Functional") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphKernelNodeGetSetParams_Functional") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecKernelNodeSetParams_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecKernelNodeSetParams_Functional") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphLaunch_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParams1D_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsToSymbol_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsToSymbol_Functional") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphNodeGetDependentNodes_Functional") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphNodeGetDependentNodes_ParamValidation") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphNodeGetDependencies_Functional") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphNodeGetDependencies_ParamValidation") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphHostNodeGetParams_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecChildGraphNodeSetParams_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecChildGraphNodeSetParams_BasicFunc") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecChildGraphNodeSetParams_ChildTopology") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_BasicFunctional") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_hipStreamPerThread") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_UniqueID") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_ParamValidation") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_1") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_2") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_3") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_4") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipUserObjectCreate_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipUserObjectRelease_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipUserObjectRetain_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipUserObj_Negative_Test") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphRetainUserObject_Functional_1") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphRetainUserObject_Functional_2") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphRetainUserObject_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphReleaseUserObject_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphRetainUserObject_Negative_Basic") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphRetainUserObject_Negative_Null_Object") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - int") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - float") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - double") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3D_ValidatePitch") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemAllocPitch_ValidatePitch") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3D_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemAllocPitch_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipHostGetFlags_flagCombos") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipHostGetFlags_DifferentThreads") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipHostGetFlags_InvalidArgs") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipHostGetDevicePointer_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMallocManaged_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemset_Negative_InvalidPtr") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemset_Negative_OutOfBoundsSize") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemset_Negative_OutOfBoundsPtr") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemset3D_Negative_InvalidPtr") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemset3D_Negative_ModifiedPtr") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemset3D_Negative_InvalidSizes") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemset3D_Negative_OutOfBounds") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemPrefetchAsync_NonPageSz") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipPtrGetAttribute_Simple") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_Basic") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_BasicAlloc") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_BasicTrim") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_BasicReuse") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_Opportunistic") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_Default") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc_ArgumentValidation") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipHostGetDevicePointer_NullCheck") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_DiffSizes") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MultiThread") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_DifferentChannelSizes") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_NullArrayPtr") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_NullDescPtr") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_BadFlags") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_8bitFloat - float") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_8bitFloat - float2") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_8bitFloat - float4") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_BadNumberOfBits") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_3ChannelElement") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_ChannelAfterZeroChannel") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_InvalidChannelFormat") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_NumericLimit") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_DiffSizes") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_MultiThread") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_NullArrayPtr") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_NullDescPtr") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_ZeroHeight") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_InvalidFlags") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_BadChannelLayout") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_8BitFloat") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_DifferentChannelSizes") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_BadChannelSize") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_NumericLimit") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3D_MultipleDataTypes - uint8_t") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3D_MultipleDataTypes - int") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3D_MultipleDataTypes - float") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3D_HosttoDevice") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3D_ExtentValidation") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - uint8_t") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - int") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - float") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_HosttoDevice") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_ExtentValidation") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipPointerGetAttribute_MemoryTypes") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipPointerGetAttribute_KernelUpdation") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipPointerGetAttribute_BufferID") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipPointerGetAttribute_MappedMem") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipPointerGetAttribute_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDrvPtrGetAttributes_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDrvPtrGetAttributes_Functional") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemGetInfo_DifferentMallocSmall") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemGetInfo_ParaSmall") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemGetInfo_ParaNonDiv") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemGetInfo_ParaMultiSmall") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemGetInfo_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTDev - char") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTDev - int") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTDev - float2") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTDev - float4") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTHost - char") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTHost - int") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTHost - float2") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTHost - float4") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTArray - char") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTArray - int") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTArray - float2") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTArray - float4") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetASyncMulti") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int8_t") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int16_t") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - uint32_t") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemset2DASyncMulti") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemset3DASyncMulti") # Timeout
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetFlags_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamDestroy_Negative_DoubleDestroy") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamDestroy_Negative_NullStream") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamSynchronize_UninitializedStream") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipEvent") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipEventIpc") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipEventSynchronize_Default_Positive") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipEventSynchronize_NoEventRecord_Positive") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetPCIBusId_Negative_PartialFill") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetCacheConfig_Positive_Default") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetCacheConfig_Positive_Basic") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetCacheConfig_Positive_Threaded") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_HipDeviceGetCacheConfig_Negative_Parameters") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceSynchronize_Positive_Nullstream") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceSynchronize_Functional") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceTotalMem_NegTst") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGetSetDeviceFlags_NullptrFlag") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGetSetDeviceFlags_InvalidFlag") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGetSetDeviceFlags_ValidFlag") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGetSetDeviceFlags_SetThenGet") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGetSetDeviceFlags_Threaded") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGetDeviceFlags_Positive_Context") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGetSetDevice_MultiThreaded") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipSetGetDevice_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetUuid_Positive") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetUuid_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetDefaultMemPool_Positive_Basic") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetDefaultMemPool_Negative_Parameters") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceSetLimit_SetGet") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Positive_Default") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Positive_Basic") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Positive_Threaded") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Negative_Parameters") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceReset_Positive_Basic") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceReset_Positive_Threaded") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceSetMemPool_Positive_Basic") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceSetMemPool_Negative_Parameters") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetMemPool_Positive_Default") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetMemPool_Positive_Basic") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetMemPool_Positive_Threaded") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetMemPool_Negative_Parameters") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDriverGetVersion_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipIpcOpenMemHandle_Negative_Open_In_Creating_Process") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipIpcOpenMemHandle_Negative_Open_In_Two_Contexts_Same_Device") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipIpcGetMemHandle_Positive_Unique_Handles_Separate_Allocations") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipIpcGetMemHandle_Positive_Unique_Handles_Reused_Memory") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipIpcGetMemHandle_Negative_Handle_For_Freed_Memory") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipIpcGetMemHandle_Negative_Out_Of_Bound_Pointer") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipIpcCloseMemHandle_Positive_Reference_Counting") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipIpcCloseMemHandle_Negative_Close_In_Originating_Process") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_printf_flags") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_printf_specifier") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipLaunchBounds_With_maxThreadsPerBlock_Check") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipLaunchBounds_With_maxThreadsPerBlock_blocksPerCU_Check") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGetLastError_Positive_Basic") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGetLastError_Positive_Threaded") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipPeekAtLastError_Positive_Basic") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipPeekAtLastError_Positive_Threaded") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamAddCallback_StrmSyncTiming") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "cuda-simpleCallback") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParams_Functional") # Subprocess aborted

# textures fail when USM is enabled
if(CHIP_USE_INTEL_USM)
  list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipTextureFetch_vector")
  list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj2D_Check")
  list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - float")
  list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - int")
  list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - unsigned char")
  list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - int16_t")
  list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - char")
  list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - unsigned int")
  list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipCreateTextureObject_tex1DfetchVerification")
  list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_tex1Dfetch_CheckModes")
  list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj1DCheckModes")
  list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj2DCheckModes")
endif()

# dGPU OpenCL Unit Test Failures
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphEventRecordNodeSetEvent_SetEventProperty") # flaky
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddEventRecordNode_Functional_ElapsedTime") # flaky
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipEventRecord") # flaky
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMultiThreadStreams1_AsyncSame") # invalid free()
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamPerThread_MultiThread") 
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamPerThread_DeviceReset_1") 
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTextureFetch_vector") 
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_tex1Dfetch_CheckModes") 
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemFaultStackAllocation_Check") 
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamAddCallback_ParamTst_Positive") # Only happens in ctest -j $(nproc): timeout
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj2D_Check") # Unkown
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - float") # Unkown
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - int") # Unkown
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - unsigned char") # Unkown
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - int16_t") # Unkown
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - char") # Unkown
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - unsigned int") # Unkown
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipCreateTextureObject_tex1DfetchVerification") # Unkown
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj1DCheckModes") # Unkown
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj2DCheckModes") # Unkown
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc_AllocateAndPoolBuffers") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc_Multithreaded_MultiGPU") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMultiThreadStreams2") # Timeout
list(APPEND DGPU_OPENCL_FAILED_TESTS "hipStreamSemantics") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "deviceMallocCompile") # Unimplemented
list(APPEND DGPU_OPENCL_FAILED_TESTS "hipTestDeviceSymbol") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "cuda-simpleCallback") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddEmptyNode_NegTest") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddDependencies_NegTest") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddEventRecordNode_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddEventWaitNode_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraph_BasicFunctional") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNode_BasicFunctional") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphClone_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphInstantiateWithFlags_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphInstantiateWithFlags_DependencyGraph") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddHostNode_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphChildGraphNodeGetGraph_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphNodeFindInClone_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecHostNodeSetParams_ClonedGraphwithHostNode") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecHostNodeSetParams_BasicFunc") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemsetNodeSetParams_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemsetNodeSetParams_Functional") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsToSymbol_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphDestroyNode_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphGetNodes_Functional") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphGetNodes_CapturedStream") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphGetNodes_ParamValidation") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphGetRootNodes_Functional") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphGetRootNodes_CapturedStream") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphGetRootNodes_ParamValidation") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphHostNodeSetParams_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNode1D_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_OrgGraphAsChildGraph") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_CloneChildGraph") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_MultipleChildNodes") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_SingleChildNode") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphNodeGetType_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams1D_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams1D_Functional") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphGetEdges_Functionality") # Timeout
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphGetEdges_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphRemoveDependencies_Func_StrmCapture") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphRemoveDependencies_ChangeComputeFunc") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphRemoveDependencies_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphInstantiate_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecUpdate_Negative_Basic") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecUpdate_Negative_TypeChange") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecUpdate_Functional") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecEventRecordNodeSetEvent_Functional") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecEventRecordNodeSetEvent_VerifyEventNotChanged") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecEventRecordNodeSetEvent_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphEventWaitNodeSetEvent_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemsetNodeGetParams_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemsetNodeSetParams_InvalidParams") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsFromSymbol_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsFromSymbol_Functional") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphEventRecordNodeGetEvent_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphEventRecordNodeSetEvent_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphEventWaitNodeGetEvent_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams_Functional") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_BasicFunctional") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_hipStreamPerThread") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Basic") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_defaultflag") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_blockingflag") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_diffflags") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_diffprio") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_multiplestrms") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_ColligatedStrmCapture_func") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Global") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_ThreadLocal") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Relaxed") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_CapturingFromWithinStrms") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_DetectingInvalidCapture") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_CapturingMultGraphsFrom1Strm") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_EndingCapturewhenCaptureInProgress") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_nestedStreamCapture") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_streamReuse") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_captureComplexGraph") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_captureEmptyStreams") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamIsCapturing_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamIsCapturing_Functional_Basic") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamIsCapturing_Functional") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamIsCapturing_hipStreamPerThread") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_BasicFunctional") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_hipStreamPerThread") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_UniqueID") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_ArgValidation") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamEndCapture_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamEndCapture_Thread_Negative") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsFromSymbol_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecEventWaitNodeSetEvent_SetAndVerifyMemory") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecEventWaitNodeSetEvent_VerifyEventNotChanged") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecEventWaitNodeSetEvent_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemsetNode_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddKernelNode_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeGetParams_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeGetParams_Functional") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParams_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParams_Functional") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphKernelNodeGetParams_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphKernelNodeSetParams_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphKernelNodeSetParams_Functional") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphKernelNodeGetSetParams_Functional") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecKernelNodeSetParams_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecKernelNodeSetParams_Functional") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphLaunch_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParams1D_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsToSymbol_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsToSymbol_Functional") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphNodeGetDependentNodes_Functional") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphNodeGetDependentNodes_ParamValidation") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphNodeGetDependencies_Functional") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphNodeGetDependencies_ParamValidation") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphHostNodeGetParams_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecChildGraphNodeSetParams_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecChildGraphNodeSetParams_BasicFunc") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecChildGraphNodeSetParams_ChildTopology") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_BasicFunctional") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_hipStreamPerThread") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_UniqueID") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_ParamValidation") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_1") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_2") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_3") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_4") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipUserObjectCreate_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipUserObjectRelease_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipUserObjectRetain_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipUserObj_Negative_Test") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphRetainUserObject_Functional_1") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphRetainUserObject_Functional_2") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphRetainUserObject_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphReleaseUserObject_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphRetainUserObject_Negative_Basic") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphRetainUserObject_Negative_Null_Object") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - int") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - float") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - double") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3D_ValidatePitch") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemAllocPitch_ValidatePitch") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3D_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemAllocPitch_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostGetFlags_flagCombos") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostGetFlags_DifferentThreads") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostGetFlags_InvalidArgs") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostGetDevicePointer_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMallocManaged_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemset_Negative_InvalidPtr") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemset_Negative_OutOfBoundsSize") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemset_Negative_OutOfBoundsPtr") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemset3D_Negative_InvalidPtr") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemset3D_Negative_ModifiedPtr") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemset3D_Negative_InvalidSizes") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemset3D_Negative_OutOfBounds") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemPrefetchAsync_NonPageSz") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipPtrGetAttribute_Simple") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_Basic") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_BasicAlloc") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_BasicTrim") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_BasicReuse") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_Opportunistic") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_Default") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc_ArgumentValidation") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostGetDevicePointer_NullCheck") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_SmallSize_hipMemset") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_SmallSize_hipMemsetD32") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_SmallSize_hipMemsetD16") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_SmallSize_hipMemsetD8") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_PartialSet_1D") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_DiffSizes") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MultiThread") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_DifferentChannelSizes") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_NullArrayPtr") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_NullDescPtr") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_BadFlags") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_8bitFloat - float") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_8bitFloat - float2") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_8bitFloat - float4") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_BadNumberOfBits") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_3ChannelElement") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_ChannelAfterZeroChannel") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_InvalidChannelFormat") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_Negative_NumericLimit") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_DiffSizes") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_MultiThread") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_NullArrayPtr") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_NullDescPtr") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_ZeroHeight") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_InvalidFlags") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_BadChannelLayout") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_8BitFloat") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_DifferentChannelSizes") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_BadChannelSize") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_NumericLimit") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3D_MultipleDataTypes - uint8_t") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3D_MultipleDataTypes - int") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3D_MultipleDataTypes - float") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3D_HosttoDevice") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3D_ExtentValidation") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - uint8_t") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - int") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - float") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_HosttoDevice") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_ExtentValidation") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipPointerGetAttribute_MemoryTypes") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipPointerGetAttribute_KernelUpdation") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipPointerGetAttribute_BufferID") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipPointerGetAttribute_MappedMem") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipPointerGetAttribute_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDrvPtrGetAttributes_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDrvPtrGetAttributes_Functional") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemGetInfo_DifferentMallocSmall") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemGetInfo_ParaSmall") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemGetInfo_ParaNonDiv") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemGetInfo_ParaMultiSmall") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemGetInfo_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTDev - char") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTDev - int") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTDev - float2") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTDev - float4") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTHost - char") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTHost - int") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTHost - float2") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTHost - float4") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTArray - char") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTArray - int") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTArray - float2") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipFreeMultiTArray - float4") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetASyncMulti") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int8_t") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int16_t") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - uint32_t") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemset2DASyncMulti") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemset3DASyncMulti") # Timeout
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetFlags_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamDestroy_Negative_DoubleDestroy") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamDestroy_Negative_NullStream") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamSynchronize_UninitializedStream") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamAddCallback_StrmSyncTiming") # Timeout
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipEventIpc") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipEventSynchronize_Default_Positive") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipEventSynchronize_NoEventRecord_Positive") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetPCIBusId_Negative_PartialFill") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetCacheConfig_Positive_Default") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetCacheConfig_Positive_Basic") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetCacheConfig_Positive_Threaded") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_HipDeviceGetCacheConfig_Negative_Parameters") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceSynchronize_Positive_Nullstream") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceSynchronize_Functional") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceTotalMem_NegTst") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGetSetDeviceFlags_NullptrFlag") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGetSetDeviceFlags_InvalidFlag") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGetSetDeviceFlags_ValidFlag") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGetSetDeviceFlags_SetThenGet") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGetSetDeviceFlags_Threaded") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGetDeviceFlags_Positive_Context") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGetSetDevice_MultiThreaded") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipSetGetDevice_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetUuid_Positive") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetUuid_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetDefaultMemPool_Positive_Basic") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetDefaultMemPool_Negative_Parameters") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceSetLimit_SetGet") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Positive_Default") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Positive_Basic") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Positive_Threaded") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Negative_Parameters") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceReset_Positive_Basic") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceReset_Positive_Threaded") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceSetMemPool_Positive_Basic") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceSetMemPool_Negative_Parameters") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetMemPool_Positive_Default") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetMemPool_Positive_Basic") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetMemPool_Positive_Threaded") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetMemPool_Negative_Parameters") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDriverGetVersion_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipIpcOpenMemHandle_Negative_Open_In_Creating_Process") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipIpcOpenMemHandle_Negative_Open_In_Two_Contexts_Same_Device") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipIpcGetMemHandle_Positive_Unique_Handles_Separate_Allocations") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipIpcGetMemHandle_Positive_Unique_Handles_Reused_Memory") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipIpcGetMemHandle_Negative_Handle_For_Freed_Memory") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipIpcGetMemHandle_Negative_Out_Of_Bound_Pointer") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipIpcCloseMemHandle_Positive_Reference_Counting") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipIpcCloseMemHandle_Negative_Close_In_Originating_Process") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_printf_flags") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_printf_specifier") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipLaunchBounds_With_maxThreadsPerBlock_Check") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipLaunchBounds_With_maxThreadsPerBlock_blocksPerCU_Check") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGetLastError_Positive_Basic") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGetLastError_Positive_Threaded") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipPeekAtLastError_Positive_Basic") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipPeekAtLastError_Positive_Threaded") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "cuda-simpleCallback") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipEvent") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_PartialSet_3D") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "hipMultiThreadAddCallback") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParams_Functional") # Subprocess aborted

# dGPU Level Zero Unit Test Failures
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipEvent") # Failing for ICL https://github.com/intel/compute-runtime/issues/668
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "hipKernelLaunchIsNonBlocking") 

list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipGraphAddEventRecordNode_MultipleRun") # 
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemsetAsync_QueueJobsMultithreaded") # only happens when ctest -j $(nproc) RCL
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemset3DAsync_ConcurrencyMthread") # only happens when ctest -j $(nproc) RCL
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "cuda-bandwidthTest") # only happens when ctest -j $(nproc) RCL
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemset3D_SeekSetSlice") # only happens when ctest -j $(nproc) RCL
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemset3DAsync_MemsetWithExtent") # only happens when ctest -j $(nproc) RCL
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemsetFunctional_PartialSet_3D") # only happens when ctest -j $(nproc) RCL

list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMultiThreadDevice_NearZero") # 
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemset2DAsync_MultiThread") # Race condition 
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "hipStreamSemantics") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "deviceMallocCompile") # Unimplemented
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "cuda-simpleCallback") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddEmptyNode_NegTest") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddDependencies_NegTest") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddEventRecordNode_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddEventWaitNode_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraph_BasicFunctional") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddMemcpyNode_BasicFunctional") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphClone_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphInstantiateWithFlags_Negative") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphInstantiateWithFlags_DependencyGraph") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddHostNode_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphChildGraphNodeGetGraph_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphNodeFindInClone_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecHostNodeSetParams_ClonedGraphwithHostNode") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecHostNodeSetParams_BasicFunc") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecMemsetNodeSetParams_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecMemsetNodeSetParams_Functional") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsToSymbol_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphDestroyNode_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphGetNodes_Functional") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphGetNodes_CapturedStream") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphGetNodes_ParamValidation") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphGetRootNodes_Functional") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphGetRootNodes_CapturedStream") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphGetRootNodes_ParamValidation") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphHostNodeSetParams_Negative") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddMemcpyNode1D_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_OrgGraphAsChildGraph") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_CloneChildGraph") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_MultipleChildNodes") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_SingleChildNode") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphNodeGetType_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams1D_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams1D_Functional") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphGetEdges_Functionality") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphGetEdges_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphRemoveDependencies_Func_StrmCapture") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphRemoveDependencies_ChangeComputeFunc") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphRemoveDependencies_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphInstantiate_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecUpdate_Negative_Basic") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecUpdate_Negative_TypeChange") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecUpdate_Functional") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecEventRecordNodeSetEvent_Functional") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecEventRecordNodeSetEvent_VerifyEventNotChanged") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecEventRecordNodeSetEvent_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphEventWaitNodeSetEvent_Negative") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphMemsetNodeGetParams_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphMemsetNodeSetParams_InvalidParams") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsFromSymbol_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsFromSymbol_Functional") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphEventRecordNodeGetEvent_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphEventRecordNodeSetEvent_Negative") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphEventWaitNodeGetEvent_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams_Functional") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_BasicFunctional") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_hipStreamPerThread") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_Basic") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_defaultflag") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_blockingflag") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_diffflags") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_diffprio") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_multiplestrms") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_ColligatedStrmCapture_func") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Global") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_ThreadLocal") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Relaxed") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_CapturingFromWithinStrms") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_DetectingInvalidCapture") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_CapturingMultGraphsFrom1Strm") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_EndingCapturewhenCaptureInProgress") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_nestedStreamCapture") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_streamReuse") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_captureComplexGraph") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_captureEmptyStreams") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamIsCapturing_Negative") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamIsCapturing_Functional_Basic") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamIsCapturing_Functional") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamIsCapturing_hipStreamPerThread") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_BasicFunctional") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_hipStreamPerThread") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_UniqueID") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_ArgValidation") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamEndCapture_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamEndCapture_Thread_Negative") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsFromSymbol_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecEventWaitNodeSetEvent_SetAndVerifyMemory") # Timeout
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecEventWaitNodeSetEvent_VerifyEventNotChanged") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecEventWaitNodeSetEvent_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddMemsetNode_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddKernelNode_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphMemcpyNodeGetParams_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphMemcpyNodeGetParams_Functional") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParams_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphKernelNodeGetParams_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphKernelNodeSetParams_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphKernelNodeSetParams_Functional") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphKernelNodeGetSetParams_Functional") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecKernelNodeSetParams_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecKernelNodeSetParams_Functional") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphLaunch_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParams1D_Negative") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsToSymbol_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsToSymbol_Functional") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphNodeGetDependentNodes_Functional") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphNodeGetDependentNodes_ParamValidation") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphNodeGetDependencies_Functional") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphNodeGetDependencies_ParamValidation") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphHostNodeGetParams_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecChildGraphNodeSetParams_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecChildGraphNodeSetParams_BasicFunc") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecChildGraphNodeSetParams_ChildTopology") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_BasicFunctional") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_hipStreamPerThread") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_UniqueID") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_ParamValidation") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_1") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_2") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_3") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_4") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipUserObjectCreate_Negative") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipUserObjectRelease_Negative") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipUserObjectRetain_Negative") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipUserObj_Negative_Test") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphRetainUserObject_Functional_1") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphRetainUserObject_Functional_2") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphRetainUserObject_Negative") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphReleaseUserObject_Negative") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphRetainUserObject_Negative_Basic") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphRetainUserObject_Negative_Null_Object") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - int") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - float") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - double") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc3D_ValidatePitch") # Bus error
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemAllocPitch_ValidatePitch") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc3D_Negative") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemAllocPitch_Negative") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipHostGetFlags_flagCombos") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipHostGetFlags_DifferentThreads") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipHostGetFlags_InvalidArgs") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipHostGetDevicePointer_Negative") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocManaged_Negative") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemset_Negative_InvalidPtr") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemset_Negative_OutOfBoundsSize") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemset_Negative_OutOfBoundsPtr") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemset3D_Negative_InvalidPtr") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemset3D_Negative_ModifiedPtr") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemset3D_Negative_InvalidSizes") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemset3D_Negative_OutOfBounds") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemset_SetMemoryWithOffset") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemsetAsync_SetMemoryWithOffset") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemPrefetchAsync_NonPageSz") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipPtrGetAttribute_Simple") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemPoolApi_Basic") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemPoolApi_BasicAlloc") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemPoolApi_BasicTrim") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemPoolApi_BasicReuse") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemPoolApi_Opportunistic") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemPoolApi_Default") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc_ArgumentValidation") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipHostGetDevicePointer_NullCheck") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_DiffSizes") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_MultiThread") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_Negative_DifferentChannelSizes") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_Negative_NullArrayPtr") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_Negative_NullDescPtr") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_Negative_BadFlags") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_Negative_8bitFloat - float") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_Negative_8bitFloat - float2") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_Negative_8bitFloat - float4") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_Negative_BadNumberOfBits") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_Negative_3ChannelElement") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_Negative_ChannelAfterZeroChannel") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_Negative_InvalidChannelFormat") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_Negative_NumericLimit") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc3DArray_DiffSizes") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc3DArray_MultiThread") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_NullArrayPtr") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_NullDescPtr") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_ZeroHeight") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_InvalidFlags") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_BadChannelLayout") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_8BitFloat") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_DifferentChannelSizes") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_BadChannelSize") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_NumericLimit") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDrvMemcpy3D_MultipleDataTypes - uint8_t") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDrvMemcpy3D_MultipleDataTypes - int") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDrvMemcpy3D_MultipleDataTypes - float") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDrvMemcpy3D_HosttoDevice") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDrvMemcpy3D_ExtentValidation") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - uint8_t") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - int") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - float") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_HosttoDevice") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_ExtentValidation") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipPointerGetAttribute_MemoryTypes") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipPointerGetAttribute_KernelUpdation") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipPointerGetAttribute_BufferID") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipPointerGetAttribute_MappedMem") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipPointerGetAttribute_Negative") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDrvPtrGetAttributes_Negative") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDrvPtrGetAttributes_Functional") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemGetInfo_DifferentMallocSmall") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemGetInfo_ParaSmall") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemGetInfo_ParaNonDiv") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemGetInfo_ParaMultiSmall") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemGetInfo_Negative") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipFreeMultiTDev - char") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipFreeMultiTDev - int") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipFreeMultiTDev - float2") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipFreeMultiTDev - float4") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipFreeMultiTHost - char") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipFreeMultiTHost - int") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipFreeMultiTHost - float2") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipFreeMultiTHost - float4") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipFreeMultiTArray - char") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipFreeMultiTArray - int") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipFreeMultiTArray - float2") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipFreeMultiTArray - float4") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemsetASyncMulti") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int8_t") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int16_t") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemsetDASyncMulti - uint32_t") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemset2DASyncMulti") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemset3DASyncMulti") # Bus error
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamGetFlags_Negative") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamDestroy_Negative_DoubleDestroy") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamDestroy_Negative_NullStream") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamSynchronize_UninitializedStream") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamAddCallback_StrmSyncTiming") # Timeout
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipEventIpc") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipEventSynchronize_Default_Positive") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipEventSynchronize_NoEventRecord_Positive") # Timeout
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetPCIBusId_Negative_PartialFill") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetCacheConfig_Positive_Default") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetCacheConfig_Positive_Basic") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetCacheConfig_Positive_Threaded") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_HipDeviceGetCacheConfig_Negative_Parameters") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceSynchronize_Positive_Nullstream") # Timeout
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceSynchronize_Functional") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceTotalMem_NegTst") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGetSetDeviceFlags_NullptrFlag") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGetSetDeviceFlags_InvalidFlag") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGetSetDeviceFlags_ValidFlag") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGetSetDeviceFlags_SetThenGet") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGetSetDeviceFlags_Threaded") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGetDeviceFlags_Positive_Context") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGetSetDevice_MultiThreaded") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipSetGetDevice_Negative") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetUuid_Positive") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetUuid_Negative") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetDefaultMemPool_Positive_Basic") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetDefaultMemPool_Negative_Parameters") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceSetLimit_SetGet") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Positive_Default") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Positive_Basic") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Positive_Threaded") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Negative_Parameters") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceReset_Positive_Basic") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceReset_Positive_Threaded") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceSetMemPool_Positive_Basic") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceSetMemPool_Negative_Parameters") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetMemPool_Positive_Default") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetMemPool_Positive_Basic") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetMemPool_Positive_Threaded") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetMemPool_Negative_Parameters") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDriverGetVersion_Negative") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipIpcOpenMemHandle_Negative_Open_In_Creating_Process") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipIpcOpenMemHandle_Negative_Open_In_Two_Contexts_Same_Device") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipIpcGetMemHandle_Positive_Unique_Handles_Separate_Allocations") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipIpcGetMemHandle_Positive_Unique_Handles_Reused_Memory") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipIpcGetMemHandle_Negative_Handle_For_Freed_Memory") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipIpcGetMemHandle_Negative_Out_Of_Bound_Pointer") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipIpcCloseMemHandle_Positive_Reference_Counting") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipIpcCloseMemHandle_Negative_Close_In_Originating_Process") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_printf_flags") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_printf_specifier") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipLaunchBounds_With_maxThreadsPerBlock_Check") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipLaunchBounds_With_maxThreadsPerBlock_blocksPerCU_Check") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGetLastError_Positive_Basic") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGetLastError_Positive_Threaded") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipPeekAtLastError_Positive_Basic") # Failed
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipPeekAtLastError_Positive_Threaded") # Subprocess aborted
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemFaultStackAllocation_Check") # SEGFAULT
list(APPEND DGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParams_Functional") # Subprocess aborted

# iGPU Level Zero Unit Test Failures
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMultiThreadDevice_NearZero") # only happens when ctest -j $(nproc) RCL
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemsetFunctional_PartialSet_3D") # only happens when ctest -j $(nproc) RCL

list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "hip_sycl_interop") # Timeout Using MKL 2023.2.3 
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "hip_sycl_interop_no_buffers") # Timeout Using MKL 2023.2.3 
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemset2DAsync_MultiThread") # Race condition 
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "hipStreamSemantics") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "deviceMallocCompile") # Unimplemented
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "cuda-simpleCallback") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddEmptyNode_NegTest") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddDependencies_NegTest") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddEventRecordNode_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddEventWaitNode_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraph_BasicFunctional") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddMemcpyNode_BasicFunctional") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphClone_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphInstantiateWithFlags_Negative") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphInstantiateWithFlags_DependencyGraph") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddHostNode_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphChildGraphNodeGetGraph_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphNodeFindInClone_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecHostNodeSetParams_ClonedGraphwithHostNode") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecHostNodeSetParams_BasicFunc") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecMemsetNodeSetParams_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecMemsetNodeSetParams_Functional") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsToSymbol_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphDestroyNode_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphGetNodes_Functional") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphGetNodes_CapturedStream") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphGetNodes_ParamValidation") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphGetRootNodes_Functional") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphGetRootNodes_CapturedStream") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphGetRootNodes_ParamValidation") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphHostNodeSetParams_Negative") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddMemcpyNode1D_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_OrgGraphAsChildGraph") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_CloneChildGraph") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_MultipleChildNodes") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_SingleChildNode") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphNodeGetType_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams1D_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams1D_Functional") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphGetEdges_Functionality") # Timeout
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphGetEdges_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphRemoveDependencies_Func_StrmCapture") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphRemoveDependencies_ChangeComputeFunc") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphRemoveDependencies_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphInstantiate_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecUpdate_Negative_Basic") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecUpdate_Negative_TypeChange") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecUpdate_Functional") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecEventRecordNodeSetEvent_Functional") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecEventRecordNodeSetEvent_VerifyEventNotChanged") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecEventRecordNodeSetEvent_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphEventWaitNodeSetEvent_Negative") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphMemsetNodeGetParams_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphMemsetNodeSetParams_InvalidParams") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsFromSymbol_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsFromSymbol_Functional") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphEventRecordNodeGetEvent_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphEventRecordNodeSetEvent_Negative") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphEventWaitNodeGetEvent_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams_Functional") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_hipStreamPerThread") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_Basic") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_defaultflag") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_blockingflag") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_diffflags") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_diffprio") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_multiplestrms") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_ColligatedStrmCapture_func") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Global") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_ThreadLocal") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Relaxed") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_CapturingFromWithinStrms") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_DetectingInvalidCapture") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_CapturingMultGraphsFrom1Strm") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_EndingCapturewhenCaptureInProgress") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_nestedStreamCapture") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_streamReuse") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_captureComplexGraph") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamBeginCapture_captureEmptyStreams") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamIsCapturing_Negative") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamIsCapturing_Functional_Basic") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamIsCapturing_Functional") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamIsCapturing_hipStreamPerThread") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_BasicFunctional") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_hipStreamPerThread") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_UniqueID") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_ArgValidation") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamEndCapture_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamEndCapture_Thread_Negative") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsFromSymbol_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecEventWaitNodeSetEvent_SetAndVerifyMemory") # Timeout
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecEventWaitNodeSetEvent_VerifyEventNotChanged") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecEventWaitNodeSetEvent_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddMemsetNode_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphAddKernelNode_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphMemcpyNodeGetParams_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphMemcpyNodeGetParams_Functional") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParams_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphKernelNodeGetParams_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphKernelNodeSetParams_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphKernelNodeSetParams_Functional") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphKernelNodeGetSetParams_Functional") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecKernelNodeSetParams_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecKernelNodeSetParams_Functional") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphLaunch_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParams1D_Negative") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsToSymbol_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsToSymbol_Functional") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphNodeGetDependentNodes_Functional") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphNodeGetDependentNodes_ParamValidation") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphNodeGetDependencies_Functional") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphNodeGetDependencies_ParamValidation") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphHostNodeGetParams_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecChildGraphNodeSetParams_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecChildGraphNodeSetParams_BasicFunc") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphExecChildGraphNodeSetParams_ChildTopology") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_BasicFunctional") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_hipStreamPerThread") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_UniqueID") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_ParamValidation") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_1") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_2") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_3") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_4") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipUserObjectCreate_Negative") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipUserObjectRelease_Negative") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipUserObjectRetain_Negative") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipUserObj_Negative_Test") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphRetainUserObject_Functional_1") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphRetainUserObject_Functional_2") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphRetainUserObject_Negative") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphReleaseUserObject_Negative") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphRetainUserObject_Negative_Basic") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphRetainUserObject_Negative_Null_Object") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - int") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - float") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - double") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc3D_ValidatePitch") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemAllocPitch_ValidatePitch") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc3D_Negative") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemAllocPitch_Negative") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipHostGetFlags_flagCombos") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipHostGetFlags_DifferentThreads") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipHostGetFlags_InvalidArgs") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipHostGetDevicePointer_Negative") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocManaged_Negative") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemset_Negative_InvalidPtr") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemset_Negative_OutOfBoundsSize") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemset_Negative_OutOfBoundsPtr") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemset3D_Negative_InvalidPtr") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemset3D_Negative_ModifiedPtr") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemset3D_Negative_InvalidSizes") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemset3D_Negative_OutOfBounds") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemset_SetMemoryWithOffset") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemsetAsync_SetMemoryWithOffset") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemPrefetchAsync_NonPageSz") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipPtrGetAttribute_Simple") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemPoolApi_Basic") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemPoolApi_BasicAlloc") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemPoolApi_BasicTrim") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemPoolApi_BasicReuse") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemPoolApi_Opportunistic") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemPoolApi_Default") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc_ArgumentValidation") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipHostGetDevicePointer_NullCheck") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_DiffSizes") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_MultiThread") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_Negative_DifferentChannelSizes") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_Negative_NullArrayPtr") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_Negative_NullDescPtr") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_Negative_BadFlags") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_Negative_8bitFloat - float") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_Negative_8bitFloat - float2") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_Negative_8bitFloat - float4") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_Negative_BadNumberOfBits") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_Negative_3ChannelElement") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_Negative_ChannelAfterZeroChannel") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_Negative_InvalidChannelFormat") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMallocArray_Negative_NumericLimit") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc3DArray_DiffSizes") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc3DArray_MultiThread") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_NullArrayPtr") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_NullDescPtr") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_ZeroHeight") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_InvalidFlags") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_BadChannelLayout") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_8BitFloat") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_DifferentChannelSizes") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_BadChannelSize") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_NumericLimit") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDrvMemcpy3D_MultipleDataTypes - uint8_t") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDrvMemcpy3D_MultipleDataTypes - int") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDrvMemcpy3D_MultipleDataTypes - float") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDrvMemcpy3D_HosttoDevice") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDrvMemcpy3D_ExtentValidation") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - uint8_t") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - int") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - float") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_HosttoDevice") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_ExtentValidation") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipPointerGetAttribute_MemoryTypes") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipPointerGetAttribute_KernelUpdation") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipPointerGetAttribute_BufferID") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipPointerGetAttribute_MappedMem") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipPointerGetAttribute_Negative") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDrvPtrGetAttributes_Negative") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDrvPtrGetAttributes_Functional") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemGetInfo_DifferentMallocSmall") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemGetInfo_ParaSmall") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemGetInfo_ParaNonDiv") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemGetInfo_ParaMultiSmall") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemGetInfo_Negative") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipFreeMultiTDev - char") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipFreeMultiTDev - int") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipFreeMultiTDev - float2") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipFreeMultiTDev - float4") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipFreeMultiTHost - char") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipFreeMultiTHost - int") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipFreeMultiTHost - float2") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipFreeMultiTHost - float4") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipFreeMultiTArray - char") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipFreeMultiTArray - int") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipFreeMultiTArray - float2") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipFreeMultiTArray - float4") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemsetASyncMulti") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int8_t") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int16_t") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemsetDASyncMulti - uint32_t") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemset2DASyncMulti") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemset3DASyncMulti") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamGetFlags_Negative") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamDestroy_Negative_DoubleDestroy") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamDestroy_Negative_NullStream") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamSynchronize_UninitializedStream") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipStreamAddCallback_StrmSyncTiming") # Timeout
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipEventIpc") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipEventSynchronize_Default_Positive") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipEventSynchronize_NoEventRecord_Positive") # Timeout
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetPCIBusId_Negative_PartialFill") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetCacheConfig_Positive_Default") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetCacheConfig_Positive_Basic") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetCacheConfig_Positive_Threaded") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_HipDeviceGetCacheConfig_Negative_Parameters") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceSynchronize_Positive_Nullstream") # Timeout
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceSynchronize_Functional") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceTotalMem_NegTst") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGetSetDeviceFlags_NullptrFlag") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGetSetDeviceFlags_InvalidFlag") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGetSetDeviceFlags_ValidFlag") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGetSetDeviceFlags_SetThenGet") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGetSetDeviceFlags_Threaded") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGetDeviceFlags_Positive_Context") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGetSetDevice_MultiThreaded") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipSetGetDevice_Negative") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetUuid_Positive") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetUuid_Negative") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetDefaultMemPool_Positive_Basic") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetDefaultMemPool_Negative_Parameters") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceSetLimit_SetGet") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Positive_Default") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Positive_Basic") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Positive_Threaded") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Negative_Parameters") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceReset_Positive_Basic") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceReset_Positive_Threaded") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceSetMemPool_Positive_Basic") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceSetMemPool_Negative_Parameters") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetMemPool_Positive_Default") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetMemPool_Positive_Basic") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetMemPool_Positive_Threaded") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDeviceGetMemPool_Negative_Parameters") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipDriverGetVersion_Negative") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipIpcOpenMemHandle_Negative_Open_In_Creating_Process") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipIpcOpenMemHandle_Negative_Open_In_Two_Contexts_Same_Device") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipIpcGetMemHandle_Positive_Unique_Handles_Separate_Allocations") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipIpcGetMemHandle_Positive_Unique_Handles_Reused_Memory") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipIpcGetMemHandle_Negative_Handle_For_Freed_Memory") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipIpcGetMemHandle_Negative_Out_Of_Bound_Pointer") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipIpcCloseMemHandle_Positive_Reference_Counting") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipIpcCloseMemHandle_Negative_Close_In_Originating_Process") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_printf_flags") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_printf_specifier") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipLaunchBounds_With_maxThreadsPerBlock_Check") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipLaunchBounds_With_maxThreadsPerBlock_blocksPerCU_Check") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGetLastError_Positive_Basic") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGetLastError_Positive_Threaded") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipPeekAtLastError_Positive_Basic") # Failed
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipPeekAtLastError_Positive_Threaded") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipMemFaultStackAllocation_Check") # SEGFAULT
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "hipKernelLaunchIsNonBlocking") # Timeout
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParams_Functional") # Subprocess aborted
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS "syncthreadsExitedThreads") # Timeout
list(APPEND IGPU_LEVEL0_BASE_FAILED_TESTS
  "Unit_hipMalloc_AllocateAndPoolBuffers") # Flaky. An event related issue.

list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMallocPitch_KernelLaunch") # Segfault in Catch2 upon de-init
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMultiThreadStreams2") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "hipStreamSemantics") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDeviceSynchronize_Positive_Nullstream") # failing for LLVM16
list(APPEND CPU_POCL_FAILED_TESTS "deviceMallocCompile") # Unimplemented
list(APPEND CPU_POCL_FAILED_TESTS "abort") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "fp16_math") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "fp16_half2_math") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "hip_async_binomial") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "BinomialOption") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "DCT") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "dwtHaar1D") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Histogram") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "RecursiveGaussian") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "cuda-simpleCallback") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "cuda-convolutionSeparable") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "cuda-histogram") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "cuda-binomialoptions") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "cuda-mergesort") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "cuda-scalarprod") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "cuda-scan") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "cuda-sortnet") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "cuda-FDTD3d") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "cuda-sobolqrng") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphAddEmptyNode_NegTest") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphAddDependencies_NegTest") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphAddEventRecordNode_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphAddEventWaitNode_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraph_BasicFunctional") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNode_BasicFunctional") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphClone_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphInstantiateWithFlags_Negative") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphInstantiateWithFlags_DependencyGraph") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphAddHostNode_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeFromSymbol_GlobalMemory") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeFromSymbol_GlobalConstMemory") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeFromSymbol_GlobalMemoryWithKernel") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphChildGraphNodeGetGraph_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphNodeFindInClone_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphExecHostNodeSetParams_ClonedGraphwithHostNode") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphExecHostNodeSetParams_BasicFunc") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeToSymbol_GlobalMemory") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeToSymbol_GlobalConstMemory") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeToSymbol_MemcpyToSymbolNodeWithKernel") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphExecMemsetNodeSetParams_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphExecMemsetNodeSetParams_Functional") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsToSymbol_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsToSymbol_Functional") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphDestroyNode_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphGetNodes_Functional") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphGetNodes_CapturedStream") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphGetNodes_ParamValidation") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphGetRootNodes_Functional") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphGetRootNodes_CapturedStream") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphGetRootNodes_ParamValidation") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphHostNodeSetParams_Negative") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNode1D_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_OrgGraphAsChildGraph") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_CloneChildGraph") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_MultipleChildNodes") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_SingleChildNode") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphNodeGetType_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams1D_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams1D_Functional") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphGetEdges_Functionality") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphGetEdges_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphRemoveDependencies_Func_StrmCapture") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphRemoveDependencies_ChangeComputeFunc") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphRemoveDependencies_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphInstantiate_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphExecUpdate_Negative_Basic") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphExecUpdate_Negative_TypeChange") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphExecUpdate_Functional") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphExecEventRecordNodeSetEvent_Functional") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphExecEventRecordNodeSetEvent_VerifyEventNotChanged") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphExecEventRecordNodeSetEvent_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphEventWaitNodeSetEvent_Negative") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphMemsetNodeGetParams_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphMemsetNodeSetParams_InvalidParams") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsFromSymbol_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsFromSymbol_Functional") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphEventRecordNodeGetEvent_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphEventRecordNodeSetEvent_Negative") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphEventWaitNodeGetEvent_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams_Functional") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamBeginCapture_hipStreamPerThread") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Basic") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_defaultflag") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_blockingflag") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_diffflags") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_diffprio") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamBeginCapture_multiplestrms") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamBeginCapture_ColligatedStrmCapture_func") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Global") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_ThreadLocal") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Relaxed") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamBeginCapture_CapturingFromWithinStrms") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamBeginCapture_DetectingInvalidCapture") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamBeginCapture_CapturingMultGraphsFrom1Strm") # Timeout
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamBeginCapture_EndingCapturewhenCaptureInProgress") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamBeginCapture_nestedStreamCapture") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamBeginCapture_streamReuse") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamBeginCapture_captureComplexGraph") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamBeginCapture_captureEmptyStreams") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamIsCapturing_Negative") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamIsCapturing_Functional_Basic") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamIsCapturing_Functional") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamIsCapturing_hipStreamPerThread") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_BasicFunctional") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_hipStreamPerThread") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_UniqueID") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_ArgValidation") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamEndCapture_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamEndCapture_Thread_Negative") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsFromSymbol_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsFromSymbol_Functional") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphExecEventWaitNodeSetEvent_SetAndVerifyMemory") # Timeout
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphExecEventWaitNodeSetEvent_VerifyEventNotChanged") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphExecEventWaitNodeSetEvent_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphAddMemsetNode_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphAddKernelNode_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeGetParams_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeGetParams_Functional") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParams_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParams_Functional") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphKernelNodeGetParams_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphKernelNodeSetParams_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphKernelNodeSetParams_Functional") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphKernelNodeGetSetParams_Functional") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphExecKernelNodeSetParams_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphExecKernelNodeSetParams_Functional") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphLaunch_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParams1D_Negative") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsToSymbol_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsToSymbol_Functional") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphNodeGetDependentNodes_Functional") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphNodeGetDependentNodes_ParamValidation") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphNodeGetDependencies_Functional") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphNodeGetDependencies_ParamValidation") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphHostNodeGetParams_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphExecChildGraphNodeSetParams_Negative") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphExecChildGraphNodeSetParams_BasicFunc") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphExecChildGraphNodeSetParams_ChildTopology") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_BasicFunctional") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_hipStreamPerThread") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_UniqueID") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_ParamValidation") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_1") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_2") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_3") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_4") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipUserObjectCreate_Negative") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipUserObjectRelease_Negative") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipUserObjectRetain_Negative") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipUserObj_Negative_Test") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphRetainUserObject_Functional_1") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphRetainUserObject_Functional_2") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphRetainUserObject_Negative") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphReleaseUserObject_Negative") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphRetainUserObject_Negative_Basic") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGraphRetainUserObject_Negative_Null_Object") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - int") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - float") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - double") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMalloc3D_ValidatePitch") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemAllocPitch_ValidatePitch") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMalloc3D_Negative") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemAllocPitch_Negative") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipHostGetFlags_flagCombos") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipHostGetFlags_DifferentThreads") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipHostGetFlags_InvalidArgs") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipHostGetDevicePointer_Negative") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMallocManaged_Negative") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemset_Negative_InvalidPtr") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemset_Negative_OutOfBoundsSize") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemset_Negative_OutOfBoundsPtr") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemset3D_Negative_InvalidPtr") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemset3D_Negative_ModifiedPtr") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemset3D_Negative_InvalidSizes") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemset3D_Negative_OutOfBounds") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemset2D_BasicFunctional") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemset2DAsync_BasicFunctional") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemPrefetchAsync_NonPageSz") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipPtrGetAttribute_Simple") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemPoolApi_Basic") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemPoolApi_BasicAlloc") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemPoolApi_BasicTrim") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemPoolApi_BasicReuse") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemPoolApi_Opportunistic") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemPoolApi_Default") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemset2DAsync_WithKernel") # Timeout
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemset2DAsync_MultiThread") # Timeout
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMalloc_ArgumentValidation") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipHostGetDevicePointer_NullCheck") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemsetFunctional_PartialSet_1D") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemsetFunctional_ZeroValue_2D") # Timeout
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMallocArray_DiffSizes") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMallocArray_MultiThread") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMallocArray_Negative_DifferentChannelSizes") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMallocArray_Negative_NullArrayPtr") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMallocArray_Negative_NullDescPtr") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMallocArray_Negative_BadFlags") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMallocArray_Negative_8bitFloat - float") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMallocArray_Negative_8bitFloat - float2") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMallocArray_Negative_8bitFloat - float4") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMallocArray_Negative_BadNumberOfBits") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMallocArray_Negative_3ChannelElement") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMallocArray_Negative_ChannelAfterZeroChannel") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMallocArray_Negative_InvalidChannelFormat") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMallocArray_Negative_NumericLimit") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMalloc3DArray_DiffSizes") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMalloc3DArray_MultiThread") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_NullArrayPtr") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_NullDescPtr") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_ZeroHeight") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_InvalidFlags") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_BadChannelLayout") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_8BitFloat") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_DifferentChannelSizes") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_BadChannelSize") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_NumericLimit") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDrvMemcpy3D_MultipleDataTypes - uint8_t") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDrvMemcpy3D_MultipleDataTypes - int") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDrvMemcpy3D_MultipleDataTypes - float") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDrvMemcpy3D_HosttoDevice") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDrvMemcpy3D_ExtentValidation") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - uint8_t") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - int") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - float") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_HosttoDevice") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_ExtentValidation") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipPointerGetAttribute_MemoryTypes") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipPointerGetAttribute_KernelUpdation") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipPointerGetAttribute_BufferID") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipPointerGetAttribute_MappedMem") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipPointerGetAttribute_Negative") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDrvPtrGetAttributes_Negative") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDrvPtrGetAttributes_Functional") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemGetInfo_DifferentMallocSmall") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemGetInfo_ParaSmall") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemGetInfo_ParaNonDiv") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemGetInfo_ParaMultiSmall") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemGetInfo_Negative") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipFreeMultiTDev - char") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipFreeMultiTDev - int") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipFreeMultiTDev - float2") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipFreeMultiTDev - float4") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipFreeMultiTHost - char") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipFreeMultiTHost - int") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipFreeMultiTHost - float2") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipFreeMultiTHost - float4") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipFreeMultiTArray - char") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipFreeMultiTArray - int") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipFreeMultiTArray - float2") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipFreeMultiTArray - float4") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemset2DASyncMulti") # Timeout
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemset3DASyncMulti") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamGetFlags_Negative") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamDestroy_Negative_DoubleDestroy") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamDestroy_Negative_NullStream") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamSynchronize_UninitializedStream") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipStreamAddCallback_StrmSyncTiming") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipEventIpc") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipEventSynchronize_Default_Positive") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipEventSynchronize_NoEventRecord_Positive") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDeviceGetPCIBusId_Negative_PartialFill") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDeviceGetCacheConfig_Positive_Default") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDeviceGetCacheConfig_Positive_Basic") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDeviceGetCacheConfig_Positive_Threaded") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_HipDeviceGetCacheConfig_Negative_Parameters") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDeviceTotalMem_NegTst") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGetSetDeviceFlags_NullptrFlag") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGetSetDeviceFlags_InvalidFlag") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGetSetDeviceFlags_ValidFlag") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGetSetDeviceFlags_SetThenGet") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGetSetDeviceFlags_Threaded") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGetDeviceFlags_Positive_Context") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGetSetDevice_MultiThreaded") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipSetGetDevice_Negative") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDeviceGetUuid_Positive") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDeviceGetUuid_Negative") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDeviceGetDefaultMemPool_Positive_Basic") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDeviceGetDefaultMemPool_Negative_Parameters") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDeviceSetLimit_SetGet") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Positive_Default") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Positive_Basic") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Positive_Threaded") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Negative_Parameters") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDeviceReset_Positive_Basic") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDeviceReset_Positive_Threaded") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDeviceSetMemPool_Positive_Basic") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDeviceSetMemPool_Negative_Parameters") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDeviceGetMemPool_Positive_Default") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDeviceGetMemPool_Positive_Basic") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDeviceGetMemPool_Positive_Threaded") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDeviceGetMemPool_Negative_Parameters") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDriverGetVersion_Negative") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipIpcOpenMemHandle_Negative_Open_In_Creating_Process") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipIpcOpenMemHandle_Negative_Open_In_Two_Contexts_Same_Device") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipIpcGetMemHandle_Positive_Unique_Handles_Separate_Allocations") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipIpcGetMemHandle_Positive_Unique_Handles_Reused_Memory") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipIpcGetMemHandle_Negative_Handle_For_Freed_Memory") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipIpcGetMemHandle_Negative_Out_Of_Bound_Pointer") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipIpcCloseMemHandle_Positive_Reference_Counting") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipIpcCloseMemHandle_Negative_Close_In_Originating_Process") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_printf_flags") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_printf_specifier") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipLaunchBounds_With_maxThreadsPerBlock_Check") # NUMERICAL
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipLaunchBounds_With_maxThreadsPerBlock_blocksPerCU_Check") # NUMERICAL
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGetLastError_Positive_Basic") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipGetLastError_Positive_Threaded") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipPeekAtLastError_Positive_Basic") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipPeekAtLastError_Positive_Threaded") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipClassKernel_Friend") # SEGFAULT
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDeviceSynchronize_Functional") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "hip_sycl_interop") # #terminate called after throwing an instance of 'sycl::_V1::runtime_error' what():  No device of requested type available
list(APPEND CPU_POCL_FAILED_TESTS "hip_sycl_interop_no_buffers") # #terminate called after throwing an instance of 'sycl::_V1::runtime_error' what():  No device of requested type available
# broken tests, they all try to write outside allocated memory;
# valgrind + pocl shows:
#
#==11492== Invalid write of size 1
#==11492==    at 0x5605B83: pocl_fill_aligned_buf_with_pattern (pocl_util.c:2590)
#==11492==    by 0x562B1F5: pocl_driver_svm_fill (common_driver.c:444)
#==11492==    by 0x5626EBF: pocl_exec_command (common.c:693)
#==11492==    by 0x562205F: pthread_scheduler_get_work (pthread_scheduler.c:529)
#==11492==    by 0x56221B7: pocl_pthread_driver_thread (pthread_scheduler.c:588)
#==11492==    by 0x5009B42: start_thread (pthread_create.c:442)
#==11492==    by 0x509ABB3: clone (clone.S:100)
#==11492==  Address 0x114baf00 is 0 bytes after a block of size 16,384 alloc'd
#==11492==    by 0x561EE67: pocl_basic_svm_alloc (basic.c:841)
#==11492==    by 0x561411A: POclSVMAlloc (clSVMAlloc.c:98)
#==11492==    by 0x4966168: SVMemoryRegion::allocate(unsigned long) (source/chip-spv/src/backend>
#
# running with older PoCL: "double free or corruption"
#
# running with Intel CPU runtime or new PoCL: hipErrorRuntimeMemory (CL_INVALID_VALUE )
# in CHIPBackendOpenCL.cc:1048:memFillAsyncImpl
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemset2D_BasicFunctional") # Timeout
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemset2DAsync_BasicFunctional") # Timeout
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemset2DAsync_WithKernel") # Timeout
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemset2DAsync_MultiThread") # Timeout
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemsetFunctional_ZeroValue_2D") # Timeout
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemsetASyncMulti") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int8_t") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int16_t") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - uint32_t") # Failed

# The following tests fail for LLVM 15 Debug & Release : Cannot find symbol _Z4sqrtDh in kernel library
list(APPEND CPU_POCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___dsqrt_rd_double") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___dsqrt_rn_double") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___dsqrt_ru_double") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___dsqrt_rz_double") # Failed

# Fails for LLVM 15 Debug: SPIR-V Parser: Failed to find size for type id 83
list(APPEND CPU_POCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest_rnorm_double") # Failed

# This causes an LLVM codegen crash with cold kcache, but with hot
# kcache it passes. Also it passes with the 'basic' driver.
list(APPEND CPU_POCL_FAILED_TESTS "TestUndefKernelArg")

list(APPEND ALL_FAILED_TESTS ${FAILING_FOR_ALL})
list(APPEND ALL_FAILED_TESTS ${DGPU_OPENCL_FAILED_TESTS})
list(APPEND ALL_FAILED_TESTS ${IGPU_OPENCL_FAILED_TESTS})
list(APPEND ALL_FAILED_TESTS ${CPU_OPENCL_FAILED_TESTS})
list(APPEND ALL_FAILED_TESTS ${DGPU_LEVEL0_BASE_FAILED_TESTS})
list(APPEND ALL_FAILED_TESTS ${DGPU_LEVEL0_RCL_FAILED_TESTS})
list(APPEND ALL_FAILED_TESTS ${DGPU_LEVEL0_ICL_FAILED_TESTS})
list(APPEND ALL_FAILED_TESTS ${IGPU_LEVEL0_BASE_FAILED_TESTS})
list(APPEND ALL_FAILED_TESTS ${IGPU_LEVEL0_RCL_FAILED_TESTS})
list(APPEND ALL_FAILED_TESTS ${IGPU_LEVEL0_ICL_FAILED_TESTS})
list(APPEND ALL_FAILED_TESTS ${CPU_POCL_FAILED_TESTS})

list(APPEND DGPU_OPENCL_FAILED_TESTS ${FAILING_FOR_ALL})
list(APPEND IGPU_OPENCL_FAILED_TESTS ${FAILING_FOR_ALL})
list(APPEND CPU_OPENCL_FAILED_TESTS ${FAILING_FOR_ALL})
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS ${FAILING_FOR_ALL} ${DGPU_LEVEL0_BASE_FAILED_TESTS} ${DGPU_LEVEL0_RCL_FAILED_TESTS})
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS ${FAILING_FOR_ALL} ${DGPU_LEVEL0_BASE_FAILED_TESTS} ${DGPU_LEVEL0_ICL_FAILED_TESTS})
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS ${FAILING_FOR_ALL} ${IGPU_LEVEL0_BASE_FAILED_TESTS} ${IGPU_LEVEL0_RCL_FAILED_TESTS})
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS ${FAILING_FOR_ALL} ${IGPU_LEVEL0_BASE_FAILED_TESTS} ${IGPU_LEVEL0_ICL_FAILED_TESTS})
list(APPEND CPU_POCL_FAILED_TESTS ${FAILING_FOR_ALL})

list(REMOVE_DUPLICATES ALL_FAILED_TESTS)

string(REGEX REPLACE ";" "\$|" DGPU_OPENCL_FAILED_TESTS_STR "${DGPU_OPENCL_FAILED_TESTS}")
string(REGEX REPLACE ";" "\$|" IGPU_OPENCL_FAILED_TESTS_STR "${IGPU_OPENCL_FAILED_TESTS}")
string(REGEX REPLACE ";" "\$|"  CPU_OPENCL_FAILED_TESTS_STR "${CPU_OPENCL_FAILED_TESTS}")
string(REGEX REPLACE ";" "\$|" DGPU_LEVEL0_RCL_FAILED_TESTS_STR "${DGPU_LEVEL0_RCL_FAILED_TESTS}")
string(REGEX REPLACE ";" "\$|" DGPU_LEVEL0_ICL_FAILED_TESTS_STR "${DGPU_LEVEL0_ICL_FAILED_TESTS}")
string(REGEX REPLACE ";" "\$|" IGPU_LEVEL0_RCL_FAILED_TESTS_STR "${IGPU_LEVEL0_RCL_FAILED_TESTS}")
string(REGEX REPLACE ";" "\$|" IGPU_LEVEL0_ICL_FAILED_TESTS_STR "${IGPU_LEVEL0_ICL_FAILED_TESTS}")
string(REGEX REPLACE ";" "\$|"    CPU_POCL_FAILED_TESTS_STR "${CPU_POCL_FAILED_TESTS}")
string(REGEX REPLACE ";" "\$|" ALL_FAILED_TESTS_STR "${ALL_FAILED_TESTS}")
string(REGEX REPLACE ";" "\$|" NON_PARALLEL_TESTS_STR "${NON_PARALLEL_TESTS}")

string(CONCAT DGPU_OPENCL_FAILED_TESTS_STR ${DGPU_OPENCL_FAILED_TESTS_STR} "\$|")
string(CONCAT IGPU_OPENCL_FAILED_TESTS_STR ${IGPU_OPENCL_FAILED_TESTS_STR} "\$|")
string(CONCAT CPU_OPENCL_FAILED_TESTS_STR ${CPU_OPENCL_FAILED_TESTS_STR} "\$|")
string(CONCAT DGPU_LEVEL0_RCL_FAILED_TESTS_STR ${DGPU_LEVEL0_RCL_FAILED_TESTS_STR} "\$|")
string(CONCAT DGPU_LEVEL0_ICL_FAILED_TESTS_STR ${DGPU_LEVEL0_ICL_FAILED_TESTS_STR} "\$|")
string(CONCAT IGPU_LEVEL0_RCL_FAILED_TESTS_STR ${IGPU_LEVEL0_RCL_FAILED_TESTS_STR} "\$|")
string(CONCAT IGPU_LEVEL0_ICL_FAILED_TESTS_STR ${IGPU_LEVEL0_ICL_FAILED_TESTS_STR} "\$|")
string(CONCAT CPU_POCL_FAILED_TESTS_STR ${CPU_POCL_FAILED_TESTS_STR} "\$|")
string(CONCAT ALL_FAILED_TESTS_STR ${ALL_FAILED_TESTS_STR} "\$|")
string(CONCAT NON_PARALLEL_TESTS_STR ${NON_PARALLEL_TESTS_STR} "\$|")

FILE(WRITE "${CMAKE_BINARY_DIR}/test_lists/dgpu_opencl_failed_tests.txt" "\"${DGPU_OPENCL_FAILED_TESTS_STR}\"")
FILE(WRITE "${CMAKE_BINARY_DIR}/test_lists/igpu_opencl_failed_tests.txt" "\"${IGPU_OPENCL_FAILED_TESTS_STR}\"")
FILE(WRITE "${CMAKE_BINARY_DIR}/test_lists/cpu_opencl_failed_tests.txt" "\"${CPU_OPENCL_FAILED_TESTS_STR}\"")
FILE(WRITE "${CMAKE_BINARY_DIR}/test_lists/dgpu_level0_failed_reg_tests.txt" "\"${DGPU_LEVEL0_RCL_FAILED_TESTS_STR}\"")
FILE(WRITE "${CMAKE_BINARY_DIR}/test_lists/dgpu_level0_failed_imm_tests.txt" "\"${DGPU_LEVEL0_ICL_FAILED_TESTS_STR}\"")
FILE(WRITE "${CMAKE_BINARY_DIR}/test_lists/igpu_level0_failed_reg_tests.txt" "\"${IGPU_LEVEL0_RCL_FAILED_TESTS_STR}\"")
FILE(WRITE "${CMAKE_BINARY_DIR}/test_lists/igpu_level0_failed_imm_tests.txt" "\"${IGPU_LEVEL0_ICL_FAILED_TESTS_STR}\"")
FILE(WRITE "${CMAKE_BINARY_DIR}/test_lists/cpu_pocl_failed_tests.txt" "\"${CPU_POCL_FAILED_TESTS_STR}\"")
FILE(WRITE "${CMAKE_BINARY_DIR}/test_lists/all_failed_tests.txt" "\"${ALL_FAILED_TESTS_STR}\"")
FILE(WRITE "${CMAKE_BINARY_DIR}/test_lists/non_parallel_tests.txt" "\"${NON_PARALLEL_TESTS_STR}\"")
