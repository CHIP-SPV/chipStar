# custom target to avoid tests that are known to fail
#
# Note that this list only contains tests external to CHIP-SPV,
# such as those frome HIP's testsuite; the internal tests
# should be disabled based on value ENABLE_FAILING_TESTS option
#  Necessary for some reason
list(APPEND  CPU_OPENCL_FAILED_TESTS " ") 
list(APPEND DGPU_OPENCL_FAILED_TESTS " ") 
list(APPEND IGPU_OPENCL_FAILED_TESTS " ") 
list(APPEND IGPU_LEVEL0_FAILED_TESTS " ")
list(APPEND DGPU_LEVEL0_FAILED_TESTS " ") 

# CPU OpenCL Unit Test Failures
list(APPEND CPU_OPENCL_FAILED_TESTS "fp16") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "2d_shuffle") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "hipDynamicShared2") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "unroll") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "hipConstantTestDeviceSymbol") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "cuda-simpleCallback") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "cuda-qrng") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___fmaf_ieee_rd_float") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___fmaf_ieee_rn_float") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___fmaf_ieee_ru_float") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___fmaf_ieee_rz_float") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___funnelshift_l_int") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___funnelshift_lc_int") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___funnelshift_r_int") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___funnelshift_rc_int") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddEmptyNode_NegTest") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddDependencies_NegTest") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddEventRecordNode_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddEventWaitNode_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraph_BasicFunctional") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraph_SimpleGraphWithKernel") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNode_BasicFunctional") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphClone_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphInstantiateWithFlags_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphInstantiateWithFlags_DependencyGraph") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddHostNode_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeFromSymbol_GlobalMemory") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeFromSymbol_GlobalConstMemory") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeFromSymbol_GlobalMemoryWithKernel") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphChildGraphNodeGetGraph_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphNodeFindInClone_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecHostNodeSetParams_ClonedGraphwithHostNode") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecHostNodeSetParams_BasicFunc") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeToSymbol_GlobalMemory") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeToSymbol_GlobalConstMemory") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeToSymbol_MemcpyToSymbolNodeWithKernel") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemsetNodeSetParams_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemsetNodeSetParams_Functional") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsToSymbol_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsToSymbol_Functional") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphDestroyNode_Negative") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphDestroyNode_DestroyDependencyNode") # SEGFAULT
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
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsFromSymbol_Functional") # Failed
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
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_Flags - int") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_Flags - float") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_Flags - double") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_Negative - int") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_Negative - float") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_Negative - double") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipHostUnregister_MemoryNotAccessableAfterUnregister") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipHostUnregister_NullPtr") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipHostUnregister_NotRegisteredPointer") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipHostUnregister_AlreadyUnregisteredPointer") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3D_ValidatePitch") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemAllocPitch_ValidatePitch") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3D_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocPitch_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemAllocPitch_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocPitch_KernelLaunch - float") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipHostGetFlags_flagCombos") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipHostGetFlags_DifferentThreads") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipHostGetFlags_InvalidArgs") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipHostGetDevicePointer_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocManaged_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemset_Negative_InvalidPtr") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemset_Negative_OutOfBoundsSize") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemset_Negative_OutOfBoundsPtr") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemset2D_Negative_InvalidPtr") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemset2D_Negative_InvalidSizes") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemset2D_Negative_OutOfBoundsPtr") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemset3D_Negative_InvalidPtr") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemset3D_Negative_ModifiedPtr") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemset3D_Negative_InvalidSizes") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemset3D_Negative_OutOfBounds") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemPrefetchAsync_NonPageSz") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemcpyFromToSymbol_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipPtrGetAttribute_Simple") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_Basic") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_BasicAlloc") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_BasicTrim") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_BasicReuse") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_Opportunistic") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_Default") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMalloc_ArgumentValidation") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipHostGetDevicePointer_NullCheck") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_ZeroSize_hipMemset") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_ZeroSize_hipMemsetD32") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_ZeroSize_hipMemsetD16") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_ZeroSize_hipMemsetD8") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_PartialSet_1D") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_PartialSet_2D") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_DiffSizes") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MultiThread") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - uint") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - int4") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - ushort") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - short2") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - char") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - char4") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - float2") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - float4") # Failed
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
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - char") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - uchar2") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - short4") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - float2") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - float4") # Failed
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
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipFreeNegativeDev") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipFreeNegativeHost") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipFreeNegativeArray") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipFreeDoubleDevice") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipFreeDoubleHost") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipFreeDoubleArray") # Subprocess aborted
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
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_HMM_OverSubscriptionTst") # Timeout
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetFlags_Negative") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamDestroy_Negative_DoubleDestroy") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamDestroy_Negative_NullStream") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamSynchronize_UninitializedStream") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipStreamAddCallback_StrmSyncTiming") # Timeout
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipEvent") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipEventRecord") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipEventIpc") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipEventSynchronize_Default_Positive") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipEventSynchronize_NoEventRecord_Positive") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetName_NegTst") # Failed
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
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipInit_Negative") # Failed
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
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipCreateTextureObject_ArgValidation") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipCreateTextureObject_LinearResource") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipCreateTextureObject_Pitch2DResource") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGetChannelDesc_CreateAndGet") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - float") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - int") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - unsigned char") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - int16_t") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - char") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - unsigned int") # SEGFAULT
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_tex1Dfetch_CheckModes") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj1DCheckRGBAModes - array") # Timeout
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj1DCheckSRGBAModes - array") # Timeout
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj1DCheckRGBAModes - buffer") # Timeout
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj1DCheckSRGBAModes - buffer") # Timeout
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj2DCheckRGBAModes") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj2DCheckSRGBAModes") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipLaunchBounds_With_maxThreadsPerBlock_Check") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipLaunchBounds_With_maxThreadsPerBlock_blocksPerCU_Check") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMultiThreadStreams2") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipClassKernel_Value") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGetLastError_Positive_Basic") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGetLastError_Positive_Threaded") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipPeekAtLastError_Positive_Basic") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipPeekAtLastError_Positive_Threaded") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "fp16") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipEvent") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocPitch_KernelLaunch - int") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocPitch_KernelLaunch - float") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMallocPitch_KernelLaunch - double") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddHostNode_ClonedGraphwithHostNode") # Failed


# iGPU OpenCL Unit Test Failures
list(APPEND IGPU_OPENCL_FAILED_TESTS "cuda-simpleCallback") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "cuda-qrng") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___fmaf_ieee_rd_float") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___fmaf_ieee_rn_float") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___fmaf_ieee_ru_float") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___fmaf_ieee_rz_float") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___funnelshift_l_int") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___funnelshift_lc_int") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___funnelshift_r_int") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___funnelshift_rc_int") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddEmptyNode_NegTest") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddDependencies_NegTest") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddEventRecordNode_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddEventWaitNode_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraph_BasicFunctional") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraph_SimpleGraphWithKernel") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNode_BasicFunctional") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphClone_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphInstantiateWithFlags_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphInstantiateWithFlags_DependencyGraph") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddHostNode_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddHostNode_ClonedGraphwithHostNode") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeFromSymbol_GlobalMemory") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeFromSymbol_GlobalConstMemory") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeFromSymbol_GlobalMemoryWithKernel") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphChildGraphNodeGetGraph_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphNodeFindInClone_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecHostNodeSetParams_ClonedGraphwithHostNode") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecHostNodeSetParams_BasicFunc") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeToSymbol_GlobalMemory") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeToSymbol_GlobalConstMemory") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeToSymbol_MemcpyToSymbolNodeWithKernel") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemsetNodeSetParams_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemsetNodeSetParams_Functional") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsToSymbol_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsToSymbol_Functional") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphDestroyNode_Negative") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphDestroyNode_DestroyDependencyNode") # SEGFAULT
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
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsFromSymbol_Functional") # Failed
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
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_Flags - int") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_Flags - float") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_Flags - double") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_Negative - int") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_Negative - float") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_Negative - double") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipHostUnregister_MemoryNotAccessableAfterUnregister") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipHostUnregister_NullPtr") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipHostUnregister_NotRegisteredPointer") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipHostUnregister_AlreadyUnregisteredPointer") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3D_ValidatePitch") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemAllocPitch_ValidatePitch") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3D_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMallocPitch_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemAllocPitch_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipHostGetFlags_flagCombos") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipHostGetFlags_DifferentThreads") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipHostGetFlags_InvalidArgs") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipHostGetDevicePointer_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMallocManaged_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemset_Negative_InvalidPtr") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemset_Negative_OutOfBoundsSize") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemset_Negative_OutOfBoundsPtr") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemset2D_Negative_InvalidPtr") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemset2D_Negative_InvalidSizes") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemset2D_Negative_OutOfBoundsPtr") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemset3D_Negative_InvalidPtr") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemset3D_Negative_ModifiedPtr") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemset3D_Negative_InvalidSizes") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemset3D_Negative_OutOfBounds") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemPrefetchAsync_NonPageSz") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemcpyFromToSymbol_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipPtrGetAttribute_Simple") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_Basic") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_BasicAlloc") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_BasicTrim") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_BasicReuse") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_Opportunistic") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_Default") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc_ArgumentValidation") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipHostGetDevicePointer_NullCheck") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_ZeroSize_hipMemset") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_ZeroSize_hipMemsetD32") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_ZeroSize_hipMemsetD16") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_ZeroSize_hipMemsetD8") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_PartialSet_1D") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_PartialSet_2D") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_DiffSizes") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MultiThread") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - uint") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - int4") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - ushort") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - short2") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - char") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - char4") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - float2") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - float4") # Failed
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
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - char") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - uchar2") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - short4") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - float2") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - float4") # Failed
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
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipFreeNegativeDev") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipFreeNegativeHost") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipFreeNegativeArray") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipFreeDoubleDevice") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipFreeDoubleHost") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipFreeDoubleArray") # Subprocess aborted
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
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_HMM_OverSubscriptionTst") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetFlags_Negative") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamDestroy_Negative_DoubleDestroy") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamDestroy_Negative_NullStream") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamSynchronize_UninitializedStream") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipEvent") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipEventRecord") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipEventIpc") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipEventSynchronize_Default_Positive") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipEventSynchronize_NoEventRecord_Positive") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetName_NegTst") # Failed
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
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipInit_Negative") # Failed
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
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipCreateTextureObject_ArgValidation") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipCreateTextureObject_LinearResource") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipCreateTextureObject_Pitch2DResource") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGetChannelDesc_CreateAndGet") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - float") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - int") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - unsigned char") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - int16_t") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - char") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - unsigned int") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_tex1Dfetch_CheckModes") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj1DCheckRGBAModes - array") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj1DCheckSRGBAModes - array") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj1DCheckRGBAModes - buffer") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj1DCheckSRGBAModes - buffer") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj2DCheckRGBAModes") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj2DCheckSRGBAModes") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipLaunchBounds_With_maxThreadsPerBlock_Check") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipLaunchBounds_With_maxThreadsPerBlock_blocksPerCU_Check") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGetLastError_Positive_Basic") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGetLastError_Positive_Threaded") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipPeekAtLastError_Positive_Basic") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipPeekAtLastError_Positive_Threaded") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "hipMultiThreadAddCallback") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamAddCallback_MultipleThreads") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamAddCallback_StrmSyncTiming") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "hipStreamSemantics") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "cuda-simpleCallback") # SEGFAULT
list(APPEND IGPU_OPENCL_FAILED_TESTS "stream") # SEGFAULT

# dGPU OpenCL Unit Test Failures
list(APPEND DGPU_OPENCL_FAILED_TESTS "hipTestDeviceSymbol") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "FloydWarshall") # Timeout
list(APPEND DGPU_OPENCL_FAILED_TESTS "cuda-asyncAPI") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "cuda-matrixMul") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "cuda-simpleCallback") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "cuda-qrng") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___fmaf_ieee_rd_float") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___fmaf_ieee_rn_float") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___fmaf_ieee_ru_float") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___fmaf_ieee_rz_float") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___funnelshift_l_int") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___funnelshift_lc_int") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___funnelshift_r_int") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest___funnelshift_rc_int") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddEmptyNode_NegTest") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddDependencies_NegTest") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddEventRecordNode_MultipleRun") # Timeout
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddEventRecordNode_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddEventWaitNode_MultipleRun") # Timeout
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddEventWaitNode_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraph_BasicFunctional") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraph_SimpleGraphWithKernel") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNode_BasicFunctional") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphClone_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphInstantiateWithFlags_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphInstantiateWithFlags_DependencyGraph") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddHostNode_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddHostNode_ClonedGraphwithHostNode") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeFromSymbol_GlobalMemory") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeFromSymbol_GlobalConstMemory") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeFromSymbol_GlobalMemoryWithKernel") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphChildGraphNodeGetGraph_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphNodeFindInClone_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecHostNodeSetParams_ClonedGraphwithHostNode") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecHostNodeSetParams_BasicFunc") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeToSymbol_GlobalMemory") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeToSymbol_GlobalConstMemory") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeToSymbol_MemcpyToSymbolNodeWithKernel") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemsetNodeSetParams_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphExecMemsetNodeSetParams_Functional") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsToSymbol_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsToSymbol_Functional") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphDestroyNode_Negative") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphDestroyNode_DestroyDependencyNode") # SEGFAULT
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
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsFromSymbol_Functional") # Failed
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
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemcpy2DToArray_PinnedMemSameGPU") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemcpy2DToArrayAsync_PinnedHostMemSameGpu") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemcpy2D_H2D-D2D-D2H - int") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemcpy2D_H2D-D2D-D2H - float") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemcpy2D_H2D-D2D-D2H - double") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemcpy2DAsync_Host&PinnedMem - int") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemcpy2DAsync_Host&PinnedMem - float") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemcpy2DAsync_Host&PinnedMem - double") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemcpy2DFromArray_PinnedMemSameGPU") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemcpy2DFromArrayAsync_PinnedHostMemSameGpu") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemcpyAtoH_Basic - char") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemcpyAtoH_Basic - int") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemcpyAtoH_Basic - float") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemcpyHtoA_Basic - char") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemcpyHtoA_Basic - int") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemcpyHtoA_Basic - float") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - int") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - float") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - double") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_Flags - int") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_Flags - float") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_Flags - double") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_Negative - int") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_Negative - float") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_Negative - double") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostUnregister_MemoryNotAccessableAfterUnregister") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostUnregister_NullPtr") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostUnregister_NotRegisteredPointer") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostUnregister_AlreadyUnregisteredPointer") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3D_ValidatePitch") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemAllocPitch_ValidatePitch") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3D_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMallocPitch_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemAllocPitch_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostGetFlags_flagCombos") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostGetFlags_DifferentThreads") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostGetFlags_InvalidArgs") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostGetDevicePointer_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMallocManaged_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemset_Negative_InvalidPtr") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemset_Negative_OutOfBoundsSize") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemset_Negative_OutOfBoundsPtr") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemset2D_Negative_InvalidPtr") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemset2D_Negative_InvalidSizes") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemset2D_Negative_OutOfBoundsPtr") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemset3D_Negative_InvalidPtr") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemset3D_Negative_ModifiedPtr") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemset3D_Negative_InvalidSizes") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemset3D_Negative_OutOfBounds") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemPrefetchAsync_NonPageSz") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemcpyFromToSymbol_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipPtrGetAttribute_Simple") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_Basic") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_BasicAlloc") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_BasicTrim") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_BasicReuse") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_Opportunistic") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemPoolApi_Default") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc_ArgumentValidation") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc_LoopRegressionAllocFreeCycles") # Timeout
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc_Multithreaded_MultiGPU") # Timeout
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostGetDevicePointer_NullCheck") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemcpyAsync_H2H-H2D-D2H-H2PinMem - char") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemcpyAsync_H2H-H2D-D2H-H2PinMem - int") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemcpyAsync_H2H-H2D-D2H-H2PinMem - float") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemcpyAsync_H2H-H2D-D2H-H2PinMem - double") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_SmallSize_hipMemset") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_SmallSize_hipMemsetD32") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_SmallSize_hipMemsetD16") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_SmallSize_hipMemsetD8") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_ZeroSize_hipMemset") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_ZeroSize_hipMemsetD32") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_ZeroSize_hipMemsetD16") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_ZeroSize_hipMemsetD8") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_PartialSet_1D") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_PartialSet_2D") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_DiffSizes") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MultiThread") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - uint") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - int4") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - ushort") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - short2") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - char") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - char4") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - float2") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - float4") # Failed
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
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - char") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - uchar2") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - short4") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - float2") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - float4") # Failed
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
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipFreeNegativeDev") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipFreeNegativeHost") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipFreeNegativeArray") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipFreeDoubleDevice") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipFreeDoubleHost") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipFreeDoubleArray") # Subprocess aborted
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
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_HMM_OverSubscriptionTst") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamGetFlags_Negative") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMultiStream_multimeDevice") # Timeout
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamAddCallback_MultipleThreads") # Timeout
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamDestroy_Negative_DoubleDestroy") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamDestroy_Negative_NullStream") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamSynchronize_UninitializedStream") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamAddCallback_StrmSyncTiming") # Timeout
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipEventRecord") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipEventIpc") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipEventSynchronize_Default_Positive") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipEventSynchronize_NoEventRecord_Positive") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceGetName_NegTst") # Failed
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
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipInit_Negative") # Failed
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
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipCreateTextureObject_ArgValidation") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipCreateTextureObject_LinearResource") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipCreateTextureObject_Pitch2DResource") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTextureFetch_vector") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj2D_Check") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGetChannelDesc_CreateAndGet") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - float") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - int") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - unsigned char") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - int16_t") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - char") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - unsigned int") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipCreateTextureObject_tex1DfetchVerification") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_tex1Dfetch_CheckModes") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj1DCheckModes") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj2DCheckModes") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj1DCheckRGBAModes - array") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj1DCheckSRGBAModes - array") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj1DCheckRGBAModes - buffer") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj1DCheckSRGBAModes - buffer") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj2DCheckRGBAModes") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj2DCheckSRGBAModes") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamPerThread_Basic") # Timeout
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamPerThread_MemcpyAsync") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemFaultStackAllocation_Check") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipLaunchBounds_With_maxThreadsPerBlock_Check") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipLaunchBounds_With_maxThreadsPerBlock_blocksPerCU_Check") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMultiThreadStreams1_AsyncSync") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMultiThreadStreams1_AsyncAsync") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMultiThreadStreams1_AsyncSame") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMultiThreadStreams2") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGetLastError_Positive_Basic") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGetLastError_Positive_Threaded") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipPeekAtLastError_Positive_Basic") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipPeekAtLastError_Positive_Threaded") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "stream") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetAsync_VerifyExecutionWithKernel") # Timeout
list(APPEND DGPU_OPENCL_FAILED_TESTS "hipTestDeviceSymbol") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "cuda-simpleCallback") # SEGFAULT
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipEvent") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "hipStreamSemantics") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_PartialSet_3D") # Failed

# # dGPU Level Zero Unit Test Failures
list(APPEND DGPU_LEVEL0_FAILED_TESTS "hipDynamicShared") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "hipDynamicShared2") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "cuda-template") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "cuda-clock") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "cuda-simpleTemplates") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "cuda-simpleCallback") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "cuda-dwtHaar1D") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_deviceFunctions_CompileTest___fmaf_ieee_rd_float") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_deviceFunctions_CompileTest___fmaf_ieee_rn_float") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_deviceFunctions_CompileTest___fmaf_ieee_ru_float") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_deviceFunctions_CompileTest___fmaf_ieee_rz_float") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_deviceFunctions_CompileTest___funnelshift_l_int") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_deviceFunctions_CompileTest___funnelshift_lc_int") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_deviceFunctions_CompileTest___funnelshift_r_int") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_deviceFunctions_CompileTest___funnelshift_rc_int") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddEmptyNode_NegTest") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddDependencies_NegTest") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddEventRecordNode_MultipleRun") # Timeout
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddEventRecordNode_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddEventWaitNode_MultipleRun") # Timeout
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddEventWaitNode_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraph_BasicFunctional") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraph_SimpleGraphWithKernel") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddMemcpyNode_BasicFunctional") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphClone_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphInstantiateWithFlags_Negative") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphInstantiateWithFlags_DependencyGraph") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddHostNode_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddHostNode_ClonedGraphwithHostNode") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeFromSymbol_GlobalMemory") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeFromSymbol_GlobalConstMemory") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeFromSymbol_GlobalMemoryWithKernel") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphChildGraphNodeGetGraph_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphNodeFindInClone_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecHostNodeSetParams_ClonedGraphwithHostNode") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecHostNodeSetParams_BasicFunc") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeToSymbol_GlobalMemory") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeToSymbol_GlobalConstMemory") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeToSymbol_MemcpyToSymbolNodeWithKernel") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecMemsetNodeSetParams_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecMemsetNodeSetParams_Functional") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsToSymbol_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsToSymbol_Functional") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphDestroyNode_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphDestroyNode_DestroyDependencyNode") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphGetNodes_Functional") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphGetNodes_CapturedStream") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphGetNodes_ParamValidation") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphGetRootNodes_Functional") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphGetRootNodes_CapturedStream") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphGetRootNodes_ParamValidation") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphHostNodeSetParams_Negative") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddMemcpyNode1D_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_OrgGraphAsChildGraph") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_CloneChildGraph") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_MultipleChildNodes") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_SingleChildNode") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphNodeGetType_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams1D_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams1D_Functional") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphGetEdges_Functionality") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphGetEdges_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphRemoveDependencies_Func_StrmCapture") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphRemoveDependencies_ChangeComputeFunc") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphRemoveDependencies_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphInstantiate_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecUpdate_Negative_Basic") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecUpdate_Negative_TypeChange") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecUpdate_Functional") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecEventRecordNodeSetEvent_Functional") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecEventRecordNodeSetEvent_VerifyEventNotChanged") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecEventRecordNodeSetEvent_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphEventWaitNodeSetEvent_Negative") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphMemsetNodeGetParams_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphMemsetNodeSetParams_InvalidParams") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsFromSymbol_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsFromSymbol_Functional") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphEventRecordNodeGetEvent_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphEventRecordNodeSetEvent_Negative") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphEventWaitNodeGetEvent_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams_Functional") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_BasicFunctional") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_hipStreamPerThread") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_Basic") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_defaultflag") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_blockingflag") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_diffflags") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_diffprio") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_ColligatedStrmCapture_diffprio") # Timeout
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_multiplestrms") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_ColligatedStrmCapture_func") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Global") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_ThreadLocal") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Relaxed") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_CapturingFromWithinStrms") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_DetectingInvalidCapture") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_CapturingMultGraphsFrom1Strm") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_EndingCapturewhenCaptureInProgress") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_nestedStreamCapture") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_streamReuse") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_captureComplexGraph") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_captureEmptyStreams") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamIsCapturing_Negative") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamIsCapturing_Functional_Basic") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamIsCapturing_Functional") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamIsCapturing_hipStreamPerThread") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_BasicFunctional") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_hipStreamPerThread") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_UniqueID") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_ArgValidation") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamEndCapture_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamEndCapture_Thread_Negative") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsFromSymbol_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsFromSymbol_Functional") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecEventWaitNodeSetEvent_SetAndVerifyMemory") # Timeout
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecEventWaitNodeSetEvent_VerifyEventNotChanged") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecEventWaitNodeSetEvent_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddMemsetNode_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddKernelNode_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphMemcpyNodeGetParams_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphMemcpyNodeGetParams_Functional") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParams_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphKernelNodeGetParams_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphKernelNodeSetParams_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphKernelNodeSetParams_Functional") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphKernelNodeGetSetParams_Functional") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecKernelNodeSetParams_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecKernelNodeSetParams_Functional") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphLaunch_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParams1D_Negative") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsToSymbol_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsToSymbol_Functional") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphNodeGetDependentNodes_Functional") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphNodeGetDependentNodes_ParamValidation") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphNodeGetDependencies_Functional") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphNodeGetDependencies_ParamValidation") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphHostNodeGetParams_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecChildGraphNodeSetParams_Negative") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecChildGraphNodeSetParams_BasicFunc") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecChildGraphNodeSetParams_ChildTopology") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_BasicFunctional") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_hipStreamPerThread") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_UniqueID") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_ParamValidation") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_1") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_2") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_3") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_4") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipUserObjectCreate_Negative") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipUserObjectRelease_Negative") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipUserObjectRetain_Negative") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipUserObj_Negative_Test") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphRetainUserObject_Functional_1") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphRetainUserObject_Functional_2") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphRetainUserObject_Negative") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphReleaseUserObject_Negative") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphRetainUserObject_Negative_Basic") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphRetainUserObject_Negative_Null_Object") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - int") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - float") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - double") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipHostRegister_Memcpy - double") # Timeout
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipHostRegister_Flags - int") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipHostRegister_Flags - float") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipHostRegister_Flags - double") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipHostRegister_Negative - int") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipHostRegister_Negative - float") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipHostRegister_Negative - double") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipHostUnregister_MemoryNotAccessableAfterUnregister") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipHostUnregister_NullPtr") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipHostUnregister_NotRegisteredPointer") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipHostUnregister_AlreadyUnregisteredPointer") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3D_ValidatePitch") # Bus error
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemAllocPitch_ValidatePitch") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3D_Negative") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocPitch_Negative") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemAllocPitch_Negative") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipHostGetFlags_flagCombos") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipHostGetFlags_DifferentThreads") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipHostGetFlags_InvalidArgs") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipHostGetDevicePointer_Negative") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocManaged_Negative") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemset_Negative_InvalidPtr") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemset_Negative_OutOfBoundsSize") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemset_Negative_OutOfBoundsPtr") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemset2D_Negative_InvalidPtr") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemset2D_Negative_InvalidSizes") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemset2D_Negative_OutOfBoundsPtr") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemset3D_Negative_InvalidPtr") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemset3D_Negative_ModifiedPtr") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemset3D_Negative_InvalidSizes") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemset3D_Negative_OutOfBounds") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemset_SetMemoryWithOffset") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemsetAsync_SetMemoryWithOffset") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemPrefetchAsync_NonPageSz") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemcpyFromToSymbol_Negative") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipPtrGetAttribute_Simple") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemPoolApi_Basic") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemPoolApi_BasicAlloc") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemPoolApi_BasicTrim") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemPoolApi_BasicReuse") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemPoolApi_Opportunistic") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemPoolApi_Default") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc_ArgumentValidation") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipHostGetDevicePointer_NullCheck") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemsetFunctional_ZeroSize_hipMemset") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemsetFunctional_ZeroSize_hipMemsetD32") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemsetFunctional_ZeroSize_hipMemsetD16") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemsetFunctional_ZeroSize_hipMemsetD8") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemsetFunctional_PartialSet_1D") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemsetFunctional_PartialSet_2D") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_DiffSizes") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_MultiThread") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - uint") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - int4") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - ushort") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - short2") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - char") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - char4") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - float2") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - float4") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_Negative_DifferentChannelSizes") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_Negative_NullArrayPtr") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_Negative_NullDescPtr") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_Negative_BadFlags") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_Negative_8bitFloat - float") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_Negative_8bitFloat - float2") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_Negative_8bitFloat - float4") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_Negative_BadNumberOfBits") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_Negative_3ChannelElement") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_Negative_ChannelAfterZeroChannel") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_Negative_InvalidChannelFormat") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_Negative_NumericLimit") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_DiffSizes") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_MultiThread") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_NullArrayPtr") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_NullDescPtr") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_ZeroHeight") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_InvalidFlags") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_BadChannelLayout") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_8BitFloat") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_DifferentChannelSizes") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_BadChannelSize") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_NumericLimit") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - char") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - uchar2") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - short4") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - float2") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - float4") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDrvMemcpy3D_MultipleDataTypes - uint8_t") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDrvMemcpy3D_MultipleDataTypes - int") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDrvMemcpy3D_MultipleDataTypes - float") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDrvMemcpy3D_HosttoDevice") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDrvMemcpy3D_ExtentValidation") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - uint8_t") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - int") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - float") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_HosttoDevice") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_ExtentValidation") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipPointerGetAttribute_MemoryTypes") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipPointerGetAttribute_KernelUpdation") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipPointerGetAttribute_BufferID") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipPointerGetAttribute_MappedMem") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipPointerGetAttribute_Negative") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDrvPtrGetAttributes_Negative") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDrvPtrGetAttributes_Functional") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemGetInfo_DifferentMallocSmall") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemGetInfo_ParaSmall") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemGetInfo_ParaNonDiv") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemGetInfo_ParaMultiSmall") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemGetInfo_Negative") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeNegativeDev") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeNegativeHost") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeNegativeArray") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeDoubleDevice") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeDoubleHost") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeDoubleArray") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeMultiTDev - char") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeMultiTDev - int") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeMultiTDev - float2") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeMultiTDev - float4") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeMultiTHost - char") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeMultiTHost - int") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeMultiTHost - float2") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeMultiTHost - float4") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeMultiTArray - char") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeMultiTArray - int") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeMultiTArray - float2") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeMultiTArray - float4") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemsetASyncMulti") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int8_t") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int16_t") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemsetDASyncMulti - uint32_t") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemset2DASyncMulti") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipMemset3DASyncMulti") # Bus error
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_HMM_OverSubscriptionTst") # Timeout
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamGetFlags_Negative") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamAddCallback_MultipleThreads") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamDestroy_Negative_DoubleDestroy") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamDestroy_Negative_NullStream") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamSynchronize_UninitializedStream") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamAddCallback_StrmSyncTiming") # Timeout
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipEventRecord") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipEventIpc") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipEventSynchronize_Default_Positive") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipEventSynchronize_NoEventRecord_Positive") # Timeout
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetName_NegTst") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetPCIBusId_Negative_PartialFill") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetCacheConfig_Positive_Default") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetCacheConfig_Positive_Basic") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetCacheConfig_Positive_Threaded") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_HipDeviceGetCacheConfig_Negative_Parameters") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceSynchronize_Positive_Nullstream") # Timeout
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceSynchronize_Functional") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceTotalMem_NegTst") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGetSetDeviceFlags_NullptrFlag") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGetSetDeviceFlags_InvalidFlag") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGetSetDeviceFlags_ValidFlag") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGetSetDeviceFlags_SetThenGet") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGetSetDeviceFlags_Threaded") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGetDeviceFlags_Positive_Context") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGetSetDevice_MultiThreaded") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipSetGetDevice_Negative") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetUuid_Positive") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetUuid_Negative") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetDefaultMemPool_Positive_Basic") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetDefaultMemPool_Negative_Parameters") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceSetLimit_SetGet") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Positive_Default") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Positive_Basic") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Positive_Threaded") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Negative_Parameters") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceReset_Positive_Basic") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceReset_Positive_Threaded") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceSetMemPool_Positive_Basic") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceSetMemPool_Negative_Parameters") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetMemPool_Positive_Default") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetMemPool_Positive_Basic") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetMemPool_Positive_Threaded") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetMemPool_Negative_Parameters") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipInit_Negative") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipDriverGetVersion_Negative") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipIpcOpenMemHandle_Negative_Open_In_Creating_Process") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipIpcOpenMemHandle_Negative_Open_In_Two_Contexts_Same_Device") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipIpcGetMemHandle_Positive_Unique_Handles_Separate_Allocations") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipIpcGetMemHandle_Positive_Unique_Handles_Reused_Memory") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipIpcGetMemHandle_Negative_Handle_For_Freed_Memory") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipIpcGetMemHandle_Negative_Out_Of_Bound_Pointer") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipIpcCloseMemHandle_Positive_Reference_Counting") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipIpcCloseMemHandle_Negative_Close_In_Originating_Process") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_printf_flags") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_printf_specifier") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipCreateTextureObject_ArgValidation") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipCreateTextureObject_LinearResource") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipCreateTextureObject_Pitch2DResource") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGetChannelDesc_CreateAndGet") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - float") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - int") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - unsigned char") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - int16_t") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - char") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - unsigned int") # SEGFAULT
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_tex1Dfetch_CheckModes") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipTextureObj1DCheckRGBAModes - array") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipTextureObj1DCheckSRGBAModes - array") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipTextureObj1DCheckRGBAModes - buffer") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipTextureObj1DCheckSRGBAModes - buffer") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipTextureObj2DCheckRGBAModes") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipTextureObj2DCheckSRGBAModes") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipLaunchBounds_With_maxThreadsPerBlock_Check") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipLaunchBounds_With_maxThreadsPerBlock_blocksPerCU_Check") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGetLastError_Positive_Basic") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipGetLastError_Positive_Threaded") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipPeekAtLastError_Positive_Basic") # Failed
list(APPEND DGPU_LEVEL0_FAILED_TESTS "Unit_hipPeekAtLastError_Positive_Threaded") # Subprocess aborted
list(APPEND DGPU_LEVEL0_FAILED_TESTS "hipDynamicShared") # SEGFAULT

# iGPU Level Zero Unit Test Failures
list(APPEND IGPU_LEVEL0_FAILED_TESTS "cuda-template") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "cuda-clock") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "cuda-simpleTemplates") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "cuda-simpleCallback") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "cuda-dwtHaar1D") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_deviceFunctions_CompileTest___fmaf_ieee_rd_float") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_deviceFunctions_CompileTest___fmaf_ieee_rn_float") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_deviceFunctions_CompileTest___fmaf_ieee_ru_float") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_deviceFunctions_CompileTest___fmaf_ieee_rz_float") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_deviceFunctions_CompileTest___funnelshift_l_int") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_deviceFunctions_CompileTest___funnelshift_lc_int") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_deviceFunctions_CompileTest___funnelshift_r_int") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_deviceFunctions_CompileTest___funnelshift_rc_int") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddEmptyNode_NegTest") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddDependencies_NegTest") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddEventRecordNode_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddEventWaitNode_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraph_BasicFunctional") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraph_SimpleGraphWithKernel") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddMemcpyNode_BasicFunctional") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphClone_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphInstantiateWithFlags_Negative") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphInstantiateWithFlags_DependencyGraph") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddHostNode_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddHostNode_ClonedGraphwithHostNode") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeFromSymbol_GlobalMemory") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeFromSymbol_GlobalConstMemory") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeFromSymbol_GlobalMemoryWithKernel") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphChildGraphNodeGetGraph_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphNodeFindInClone_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecHostNodeSetParams_ClonedGraphwithHostNode") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecHostNodeSetParams_BasicFunc") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeToSymbol_GlobalMemory") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeToSymbol_GlobalConstMemory") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddMemcpyNodeToSymbol_MemcpyToSymbolNodeWithKernel") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecMemsetNodeSetParams_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecMemsetNodeSetParams_Functional") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsToSymbol_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsToSymbol_Functional") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphDestroyNode_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphDestroyNode_DestroyDependencyNode") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphGetNodes_Functional") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphGetNodes_CapturedStream") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphGetNodes_ParamValidation") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphGetRootNodes_Functional") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphGetRootNodes_CapturedStream") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphGetRootNodes_ParamValidation") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphHostNodeSetParams_Negative") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddMemcpyNode1D_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_OrgGraphAsChildGraph") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_CloneChildGraph") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_MultipleChildNodes") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddChildGraphNode_SingleChildNode") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphNodeGetType_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams1D_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams1D_Functional") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphGetEdges_Functionality") # Timeout
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphGetEdges_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphRemoveDependencies_Func_StrmCapture") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphRemoveDependencies_ChangeComputeFunc") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphRemoveDependencies_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphInstantiate_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecUpdate_Negative_Basic") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecUpdate_Negative_TypeChange") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecUpdate_Functional") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecEventRecordNodeSetEvent_Functional") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecEventRecordNodeSetEvent_VerifyEventNotChanged") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecEventRecordNodeSetEvent_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphEventWaitNodeSetEvent_Negative") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphMemsetNodeGetParams_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphMemsetNodeSetParams_InvalidParams") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsFromSymbol_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsFromSymbol_Functional") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphEventRecordNodeGetEvent_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphEventRecordNodeSetEvent_Negative") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphEventWaitNodeGetEvent_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParams_Functional") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_hipStreamPerThread") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_Basic") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_defaultflag") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_blockingflag") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_diffflags") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_InterStrmEventSync_diffprio") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_multiplestrms") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_ColligatedStrmCapture_func") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Global") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_ThreadLocal") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Relaxed") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_CapturingFromWithinStrms") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_DetectingInvalidCapture") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_CapturingMultGraphsFrom1Strm") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_EndingCapturewhenCaptureInProgress") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_nestedStreamCapture") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_streamReuse") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_captureComplexGraph") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamBeginCapture_captureEmptyStreams") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamIsCapturing_Negative") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamIsCapturing_Functional_Basic") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamIsCapturing_Functional") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamIsCapturing_hipStreamPerThread") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_BasicFunctional") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_hipStreamPerThread") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_UniqueID") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_ArgValidation") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamEndCapture_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamEndCapture_Thread_Negative") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsFromSymbol_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParamsFromSymbol_Functional") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecEventWaitNodeSetEvent_SetAndVerifyMemory") # Timeout
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecEventWaitNodeSetEvent_VerifyEventNotChanged") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecEventWaitNodeSetEvent_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddMemsetNode_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphAddKernelNode_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphMemcpyNodeGetParams_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphMemcpyNodeGetParams_Functional") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParams_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphKernelNodeGetParams_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphKernelNodeSetParams_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphKernelNodeSetParams_Functional") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphKernelNodeGetSetParams_Functional") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecKernelNodeSetParams_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecKernelNodeSetParams_Functional") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphLaunch_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphMemcpyNodeSetParams1D_Negative") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsToSymbol_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecMemcpyNodeSetParamsToSymbol_Functional") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphNodeGetDependentNodes_Functional") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphNodeGetDependentNodes_ParamValidation") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphNodeGetDependencies_Functional") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphNodeGetDependencies_ParamValidation") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphHostNodeGetParams_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecChildGraphNodeSetParams_Negative") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecChildGraphNodeSetParams_BasicFunc") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphExecChildGraphNodeSetParams_ChildTopology") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_BasicFunctional") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_hipStreamPerThread") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_UniqueID") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamGetCaptureInfo_v2_ParamValidation") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_1") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_2") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_3") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipUserObjectCreate_Functional_4") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipUserObjectCreate_Negative") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipUserObjectRelease_Negative") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipUserObjectRetain_Negative") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipUserObj_Negative_Test") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphRetainUserObject_Functional_1") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphRetainUserObject_Functional_2") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphRetainUserObject_Negative") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphReleaseUserObject_Negative") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphRetainUserObject_Negative_Basic") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGraphRetainUserObject_Negative_Null_Object") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - int") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - float") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - double") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipHostRegister_Flags - int") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipHostRegister_Flags - float") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipHostRegister_Flags - double") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipHostRegister_Negative - int") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipHostRegister_Negative - float") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipHostRegister_Negative - double") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipHostUnregister_MemoryNotAccessableAfterUnregister") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipHostUnregister_NullPtr") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipHostUnregister_NotRegisteredPointer") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipHostUnregister_AlreadyUnregisteredPointer") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3D_ValidatePitch") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemAllocPitch_ValidatePitch") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3D_Negative") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocPitch_Negative") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemAllocPitch_Negative") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipHostGetFlags_flagCombos") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipHostGetFlags_DifferentThreads") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipHostGetFlags_InvalidArgs") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipHostGetDevicePointer_Negative") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocManaged_Negative") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemset_Negative_InvalidPtr") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemset_Negative_OutOfBoundsSize") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemset_Negative_OutOfBoundsPtr") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemset2D_Negative_InvalidPtr") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemset2D_Negative_InvalidSizes") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemset2D_Negative_OutOfBoundsPtr") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemset3D_Negative_InvalidPtr") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemset3D_Negative_ModifiedPtr") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemset3D_Negative_InvalidSizes") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemset3D_Negative_OutOfBounds") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemset_SetMemoryWithOffset") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemsetAsync_SetMemoryWithOffset") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemPrefetchAsync_NonPageSz") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemcpyFromToSymbol_Negative") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipPtrGetAttribute_Simple") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemPoolApi_Basic") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemPoolApi_BasicAlloc") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemPoolApi_BasicTrim") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemPoolApi_BasicReuse") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemPoolApi_Opportunistic") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemPoolApi_Default") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc_ArgumentValidation") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipHostGetDevicePointer_NullCheck") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemsetFunctional_ZeroSize_hipMemset") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemsetFunctional_ZeroSize_hipMemsetD32") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemsetFunctional_ZeroSize_hipMemsetD16") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemsetFunctional_ZeroSize_hipMemsetD8") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemsetFunctional_PartialSet_1D") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemsetFunctional_PartialSet_2D") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_DiffSizes") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_MultiThread") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - uint") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - int4") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - ushort") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - short2") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - char") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - char4") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - float2") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_MaxTexture_Default - float4") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_Negative_DifferentChannelSizes") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_Negative_NullArrayPtr") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_Negative_NullDescPtr") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_Negative_BadFlags") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_Negative_8bitFloat - float") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_Negative_8bitFloat - float2") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_Negative_8bitFloat - float4") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_Negative_BadNumberOfBits") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_Negative_3ChannelElement") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_Negative_ChannelAfterZeroChannel") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_Negative_InvalidChannelFormat") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMallocArray_Negative_NumericLimit") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_DiffSizes") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_MultiThread") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_NullArrayPtr") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_NullDescPtr") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_ZeroHeight") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_InvalidFlags") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_BadChannelLayout") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_8BitFloat") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_DifferentChannelSizes") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_BadChannelSize") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_NumericLimit") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - char") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - uchar2") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - short4") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - float2") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - float4") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDrvMemcpy3D_MultipleDataTypes - uint8_t") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDrvMemcpy3D_MultipleDataTypes - int") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDrvMemcpy3D_MultipleDataTypes - float") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDrvMemcpy3D_HosttoDevice") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDrvMemcpy3D_ExtentValidation") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - uint8_t") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - int") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - float") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_HosttoDevice") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDrvMemcpy3DAsync_ExtentValidation") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipPointerGetAttribute_MemoryTypes") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipPointerGetAttribute_KernelUpdation") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipPointerGetAttribute_BufferID") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipPointerGetAttribute_MappedMem") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipPointerGetAttribute_Negative") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDrvPtrGetAttributes_Negative") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDrvPtrGetAttributes_Functional") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemGetInfo_DifferentMallocSmall") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemGetInfo_ParaSmall") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemGetInfo_ParaNonDiv") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemGetInfo_ParaMultiSmall") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemGetInfo_Negative") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeNegativeDev") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeNegativeHost") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeNegativeArray") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeDoubleDevice") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeDoubleHost") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeDoubleArray") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeMultiTDev - char") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeMultiTDev - int") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeMultiTDev - float2") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeMultiTDev - float4") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeMultiTHost - char") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeMultiTHost - int") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeMultiTHost - float2") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeMultiTHost - float4") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeMultiTArray - char") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeMultiTArray - int") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeMultiTArray - float2") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipFreeMultiTArray - float4") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemsetASyncMulti") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int8_t") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int16_t") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemsetDASyncMulti - uint32_t") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemset2DASyncMulti") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipMemset3DASyncMulti") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_HMM_OverSubscriptionTst") # Timeout
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamGetFlags_Negative") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamAddCallback_MultipleThreads") # Timeout
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamDestroy_Negative_DoubleDestroy") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamDestroy_Negative_NullStream") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamSynchronize_UninitializedStream") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipStreamAddCallback_StrmSyncTiming") # Timeout
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipEventRecord") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipEventIpc") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipEventSynchronize_Default_Positive") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipEventSynchronize_NoEventRecord_Positive") # Timeout
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetName_NegTst") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetPCIBusId_Negative_PartialFill") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetCacheConfig_Positive_Default") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetCacheConfig_Positive_Basic") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetCacheConfig_Positive_Threaded") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_HipDeviceGetCacheConfig_Negative_Parameters") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceSynchronize_Positive_Nullstream") # Timeout
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceSynchronize_Functional") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceTotalMem_NegTst") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGetSetDeviceFlags_NullptrFlag") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGetSetDeviceFlags_InvalidFlag") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGetSetDeviceFlags_ValidFlag") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGetSetDeviceFlags_SetThenGet") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGetSetDeviceFlags_Threaded") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGetDeviceFlags_Positive_Context") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGetSetDevice_MultiThreaded") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipSetGetDevice_Negative") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetUuid_Positive") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetUuid_Negative") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetDefaultMemPool_Positive_Basic") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetDefaultMemPool_Negative_Parameters") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceSetLimit_SetGet") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Positive_Default") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Positive_Basic") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Positive_Threaded") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetSharedMemConfig_Negative_Parameters") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceReset_Positive_Basic") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceReset_Positive_Threaded") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceSetMemPool_Positive_Basic") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceSetMemPool_Negative_Parameters") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetMemPool_Positive_Default") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetMemPool_Positive_Basic") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetMemPool_Positive_Threaded") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDeviceGetMemPool_Negative_Parameters") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipInit_Negative") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipDriverGetVersion_Negative") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipIpcOpenMemHandle_Negative_Open_In_Creating_Process") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipIpcOpenMemHandle_Negative_Open_In_Two_Contexts_Same_Device") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipIpcGetMemHandle_Positive_Unique_Handles_Separate_Allocations") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipIpcGetMemHandle_Positive_Unique_Handles_Reused_Memory") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipIpcGetMemHandle_Negative_Handle_For_Freed_Memory") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipIpcGetMemHandle_Negative_Out_Of_Bound_Pointer") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipIpcCloseMemHandle_Positive_Reference_Counting") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipIpcCloseMemHandle_Negative_Close_In_Originating_Process") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_printf_flags") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_printf_specifier") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipCreateTextureObject_ArgValidation") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipCreateTextureObject_LinearResource") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipCreateTextureObject_Pitch2DResource") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGetChannelDesc_CreateAndGet") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - float") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - int") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - unsigned char") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - int16_t") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - char") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - unsigned int") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_tex1Dfetch_CheckModes") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipTextureObj1DCheckRGBAModes - array") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipTextureObj1DCheckSRGBAModes - array") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipTextureObj1DCheckRGBAModes - buffer") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipTextureObj1DCheckSRGBAModes - buffer") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipTextureObj2DCheckRGBAModes") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipTextureObj2DCheckSRGBAModes") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipLaunchBounds_With_maxThreadsPerBlock_Check") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipLaunchBounds_With_maxThreadsPerBlock_blocksPerCU_Check") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGetLastError_Positive_Basic") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipGetLastError_Positive_Threaded") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipPeekAtLastError_Positive_Basic") # Failed
list(APPEND IGPU_LEVEL0_FAILED_TESTS "Unit_hipPeekAtLastError_Positive_Threaded") # Subprocess aborted
list(APPEND IGPU_LEVEL0_FAILED_TESTS "hipDynamicShared") # SEGFAULT
list(APPEND IGPU_LEVEL0_FAILED_TESTS "hipDynamicShared2") # SEGFAULT

list(APPEND ALL_FAILED_TESTS ${DGPU_OPENCL_FAILED_TESTS})
list(APPEND ALL_FAILED_TESTS ${IGPU_OPENCL_FAILED_TESTS})
list(APPEND ALL_FAILED_TESTS ${CPU_OPENCL_FAILED_TESTS})
list(APPEND ALL_FAILED_TESTS ${DGPU_LEVEL0_FAILED_TESTS})
list(APPEND ALL_FAILED_TESTS ${IGPU_LEVEL0_FAILED_TESTS})

list(REMOVE_DUPLICATES ALL_FAILED_TESTS)

string(REGEX REPLACE ";" "\$|" DGPU_OPENCL_FAILED_TESTS_STR "${DGPU_OPENCL_FAILED_TESTS}")
string(REGEX REPLACE ";" "\$|" IGPU_OPENCL_FAILED_TESTS_STR "${IGPU_OPENCL_FAILED_TESTS}")
string(REGEX REPLACE ";" "\$|"  CPU_OPENCL_FAILED_TESTS_STR "${CPU_OPENCL_FAILED_TESTS}")
string(REGEX REPLACE ";" "\$|" DGPU_LEVEL0_FAILED_TESTS_STR "${DGPU_LEVEL0_FAILED_TESTS}")
string(REGEX REPLACE ";" "\$|" IGPU_LEVEL0_FAILED_TESTS_STR "${IGPU_LEVEL0_FAILED_TESTS}")
string(REGEX REPLACE ";" "\$|" ALL_FAILED_TESTS_STR "${ALL_FAILED_TESTS}")

string(CONCAT DGPU_OPENCL_FAILED_TESTS_STR ${DGPU_OPENCL_FAILED_TESTS_STR} "\$|")
string(CONCAT IGPU_OPENCL_FAILED_TESTS_STR ${IGPU_OPENCL_FAILED_TESTS_STR} "\$|")
string(CONCAT CPU_OPENCL_FAILED_TESTS_STR ${CPU_OPENCL_FAILED_TESTS_STR} "\$|")
string(CONCAT DGPU_LEVEL0_FAILED_TESTS_STR ${DGPU_LEVEL0_FAILED_TESTS_STR} "\$|")
string(CONCAT IGPU_LEVEL0_FAILED_TESTS_STR ${IGPU_LEVEL0_FAILED_TESTS_STR} "\$|")
string(CONCAT ALL_FAILED_TESTS_STR ${ALL_FAILED_TESTS_STR} "\$|")

FILE(WRITE "${CMAKE_BINARY_DIR}/test_lists/dgpu_opencl_failed_tests.txt" "\"${DGPU_OPENCL_FAILED_TESTS_STR}\"")
FILE(WRITE "${CMAKE_BINARY_DIR}/test_lists/igpu_opencl_failed_tests.txt" "\"${IGPU_OPENCL_FAILED_TESTS_STR}\"")
FILE(WRITE "${CMAKE_BINARY_DIR}/test_lists/cpu_opencl_failed_tests.txt" "\"${CPU_OPENCL_FAILED_TESTS_STR}\"")
FILE(WRITE "${CMAKE_BINARY_DIR}/test_lists/dgpu_level0_failed_tests.txt" "\"${DGPU_LEVEL0_FAILED_TESTS_STR}\"")
FILE(WRITE "${CMAKE_BINARY_DIR}/test_lists/igpu_level0_failed_tests.txt" "\"${IGPU_LEVEL0_FAILED_TESTS_STR}\"")
FILE(WRITE "${CMAKE_BINARY_DIR}/test_lists/all_failed_tests.txt" "\"${ALL_FAILED_TESTS_STR}\"")

set(EXCLUDED_TESTS "${ALL_FAILED_TESTS_STR}$")

# TODO fix-254 how do I make these read from the environment?
# MULTI_TESTS_REPEAT=33 make multi_tests
# Preferably without an additional reconfigure. Every way I tried escaping ${MULTI_TESTS_REPEAT} results in something undesirable like \${MULTI_TESTS_REPEAT}
set(FLAKY_TESTS_REPEAT 100)
set(MULTI_TESTS_REPEAT 10)
set(PARALLEL_TESTS 1)

set(TEST_OPTIONS -j ${PARALLEL_TESTS} --timeout 120 --output-on-failure)
add_custom_target(flaky_tests COMMAND ${CMAKE_CTEST_COMMAND} ${TEST_OPTIONS} -R ${FLAKY_TESTS} --repeat until-fail:${FLAKY_TESTS_REPEAT} USES_TERMINAL VERBATIM)
add_custom_target(multi_tests COMMAND ${CMAKE_CTEST_COMMAND} ${TEST_OPTIONS} -R "[Aa]sync|[Mm]ulti[Tt]hread|[Mm]ulti[Ss]tream|[Tt]hread|[Ss]tream" --repeat until-fail:${MULTI_TESTS_REPEAT} USES_TERMINAL VERBATIM)

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} ${TEST_OPTIONS} -E ${EXCLUDED_TESTS} VERBATIM)