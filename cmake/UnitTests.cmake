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

list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2float_rd_float") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2float_rn_float") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2float_ru_float") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2float_rz_float") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2hiint_int") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2int_rd_int") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2int_rn_int") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2int_ru_int") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2int_rz_int") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2ll_rd_longlong") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2ll_rn_longlong") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2ll_ru_longlong") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2ll_rz_longlong") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2loint_int") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2uint_rd_unsigned") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2uint_rn_unsigned") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2uint_ru_unsigned") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2uint_rz_unsigned") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2ull_rd_ulonglong") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2ull_rn_ulonglong") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2ull_ru_ulonglong") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___double2ull_rz_ulonglong") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2int_rd_int") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2int_rn_int") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2int_ru_int") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2int_rz_int") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2ll_rd_longlong") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2ll_rn_longlong") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2ll_ru_longlong") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2ll_rz_longlong") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2uint_rd_unsigned") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2uint_rn_unsigned") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2uint_ru_unsigned") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2uint_rz_unsigned") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2ull_rd_ulonglong") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2ull_rn_ulonglong") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2ull_ru_ulonglong") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___float2ull_rz_ulonglong") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___hiloint2double_double") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___int2double_rn_double") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___int2float_rd_float") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___int2float_rn_float") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___int2float_ru_float") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___int2float_rz_float") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ll2double_rd_double") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ll2double_rn_double") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ll2double_ru_double") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ll2double_rz_double") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ll2float_rd_float") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ll2float_rn_float") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ll2float_ru_float") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ll2float_rz_float") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___uint2double_rn_double") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___uint2float_rd_float") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___uint2float_rn_float") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___uint2float_ru_float") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___uint2float_rz_float") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ull2double_rd_double") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ull2double_rn_double") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ull2double_ru_double") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ull2double_rz_double") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ull2float_rd_float") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ull2float_rn_float") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ull2float_ru_float") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_deviceFunctions_CompileTest___ull2float_rz_float") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphAddEmptyNode_NegTest") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphAddDependencies_NegTest") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphAddEventRecordNode_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphAddEventWaitNode_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraph_BasicFunctional") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphAddMemcpyNode_BasicFunctional") # SEGFAULT|Failed|SEGFAULT|Failed|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphClone_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphInstantiateWithFlags_Negative") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphInstantiateWithFlags_DependencyGraph") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphAddHostNode_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphAddHostNode_ClonedGraphwithHostNode") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphChildGraphNodeGetGraph_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphNodeFindInClone_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphExecHostNodeSetParams_ClonedGraphwithHostNode") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphExecHostNodeSetParams_BasicFunc") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphExecMemsetNodeSetParams_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphExecMemsetNodeSetParams_Functional") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphMemcpyNodeSetParamsToSymbol_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphDestroyNode_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphDestroyNode_DestroyDependencyNode") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphGetNodes_Functional") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphGetNodes_CapturedStream") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphGetNodes_ParamValidation") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphGetRootNodes_Functional") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphGetRootNodes_CapturedStream") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphGetRootNodes_ParamValidation") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphHostNodeSetParams_Negative") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphAddMemcpyNode1D_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphAddChildGraphNode_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphAddChildGraphNode_OrgGraphAsChildGraph") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphAddChildGraphNode_CloneChildGraph") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphAddChildGraphNode_MultipleChildNodes") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphAddChildGraphNode_SingleChildNode") # SEGFAULT|Failed|Failed|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphNodeGetType_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphExecMemcpyNodeSetParams1D_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphExecMemcpyNodeSetParams1D_Functional") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphGetEdges_Functionality") # Subprocess aborted|Timeout|Failed|Timeout|Timeout|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphGetEdges_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphRemoveDependencies_Func_StrmCapture") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphRemoveDependencies_ChangeComputeFunc") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphRemoveDependencies_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphInstantiate_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphExecUpdate_Negative_Basic") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphExecUpdate_Negative_TypeChange") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphExecUpdate_Functional") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphExecEventRecordNodeSetEvent_Functional") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphExecEventRecordNodeSetEvent_VerifyEventNotChanged") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphExecEventRecordNodeSetEvent_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphEventWaitNodeSetEvent_Negative") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphMemsetNodeGetParams_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphMemsetNodeSetParams_InvalidParams") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphExecMemcpyNodeSetParamsFromSymbol_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphExecMemcpyNodeSetParamsFromSymbol_Functional") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphEventRecordNodeGetEvent_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphEventRecordNodeSetEvent_Negative") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphEventWaitNodeGetEvent_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphExecMemcpyNodeSetParams_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphExecMemcpyNodeSetParams_Functional") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamBeginCapture_hipStreamPerThread") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamBeginCapture_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamBeginCapture_Basic") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamBeginCapture_InterStrmEventSync_defaultflag") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamBeginCapture_InterStrmEventSync_blockingflag") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamBeginCapture_InterStrmEventSync_diffflags") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamBeginCapture_InterStrmEventSync_diffprio") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamBeginCapture_multiplestrms") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamBeginCapture_CapturingFromWithinStrms") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamBeginCapture_DetectingInvalidCapture") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamBeginCapture_CapturingMultGraphsFrom1Strm") # Timeout|Timeout|Timeout|Timeout|Timeout|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamBeginCapture_EndingCapturewhenCaptureInProgress") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamBeginCapture_nestedStreamCapture") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamBeginCapture_streamReuse") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamBeginCapture_captureComplexGraph") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamBeginCapture_captureEmptyStreams") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamIsCapturing_Negative") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamIsCapturing_Functional_Basic") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamIsCapturing_Functional") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamIsCapturing_hipStreamPerThread") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamGetCaptureInfo_BasicFunctional") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamGetCaptureInfo_hipStreamPerThread") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamGetCaptureInfo_UniqueID") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamGetCaptureInfo_ArgValidation") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamEndCapture_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamEndCapture_Thread_Negative") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphMemcpyNodeSetParamsFromSymbol_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphExecEventWaitNodeSetEvent_SetAndVerifyMemory") # Subprocess aborted|Subprocess aborted|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphExecEventWaitNodeSetEvent_VerifyEventNotChanged") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphExecEventWaitNodeSetEvent_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphAddMemsetNode_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphAddKernelNode_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphMemcpyNodeGetParams_Negative") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphMemcpyNodeGetParams_Functional") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphMemcpyNodeSetParams_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphMemcpyNodeSetParams_Functional") # Failed|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphKernelNodeGetParams_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphKernelNodeSetParams_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphKernelNodeSetParams_Functional") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphKernelNodeGetSetParams_Functional") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphExecKernelNodeSetParams_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphExecKernelNodeSetParams_Functional") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphLaunch_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphMemcpyNodeSetParams1D_Negative") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphExecMemcpyNodeSetParamsToSymbol_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphExecMemcpyNodeSetParamsToSymbol_Functional") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphNodeGetDependentNodes_Functional") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphNodeGetDependentNodes_ParamValidation") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphNodeGetDependencies_Functional") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphNodeGetDependencies_ParamValidation") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphHostNodeGetParams_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphExecChildGraphNodeSetParams_Negative") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphExecChildGraphNodeSetParams_BasicFunc") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphExecChildGraphNodeSetParams_ChildTopology") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamGetCaptureInfo_v2_BasicFunctional") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamGetCaptureInfo_v2_hipStreamPerThread") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamGetCaptureInfo_v2_UniqueID") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamGetCaptureInfo_v2_ParamValidation") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipUserObjectCreate_Functional_1") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipUserObjectCreate_Functional_2") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipUserObjectCreate_Functional_3") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipUserObjectCreate_Functional_4") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipUserObjectCreate_Negative") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipUserObjectRelease_Negative") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipUserObjectRetain_Negative") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipUserObj_Negative_Test") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphRetainUserObject_Functional_1") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphRetainUserObject_Functional_2") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphRetainUserObject_Negative") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphReleaseUserObject_Negative") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphRetainUserObject_Negative_Basic") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGraphRetainUserObject_Negative_Null_Object") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - int") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - float") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - double") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMalloc3D_ValidatePitch") # Failed|Timeout|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMemAllocPitch_ValidatePitch") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMalloc3D_Negative") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMemAllocPitch_Negative") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMallocPitch_KernelLaunch - int") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMallocPitch_KernelLaunch - float") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMallocPitch_KernelLaunch - double") # Failed|Failed|Failed|Failed|SEGFAULT|
list(APPEND FAILING_FOR_ALL "Unit_hipHostGetFlags_flagCombos") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipHostGetFlags_DifferentThreads") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipHostGetFlags_InvalidArgs") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipHostGetDevicePointer_Negative") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMallocManaged_Negative") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMemset_Negative_InvalidPtr") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMemset_Negative_OutOfBoundsSize") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMemset_Negative_OutOfBoundsPtr") # Failed|Failed|Failed|Failed|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipMemset3D_Negative_InvalidPtr") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMemset3D_Negative_ModifiedPtr") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMemset3D_Negative_InvalidSizes") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMemset3D_Negative_OutOfBounds") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMemPrefetchAsync_NonPageSz") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipPtrGetAttribute_Simple") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMemPoolApi_Basic") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMemPoolApi_BasicAlloc") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMemPoolApi_BasicTrim") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMemPoolApi_BasicReuse") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMemPoolApi_Opportunistic") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMemPoolApi_Default") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMalloc_ArgumentValidation") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipHostGetDevicePointer_NullCheck") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_DiffSizes") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_MultiThread") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_MaxTexture_Default - uint") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_MaxTexture_Default - int4") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_MaxTexture_Default - ushort") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_MaxTexture_Default - short2") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_MaxTexture_Default - char") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_MaxTexture_Default - char4") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_MaxTexture_Default - float2") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_MaxTexture_Default - float4") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_Negative_DifferentChannelSizes") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_Negative_NullArrayPtr") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_Negative_NullDescPtr") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_Negative_BadFlags") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_Negative_8bitFloat - float") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_Negative_8bitFloat - float2") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_Negative_8bitFloat - float4") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_Negative_BadNumberOfBits") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_Negative_3ChannelElement") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_Negative_ChannelAfterZeroChannel") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_Negative_InvalidChannelFormat") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMallocArray_Negative_NumericLimit") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMalloc3DArray_DiffSizes") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipMalloc3DArray_MultiThread") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipMalloc3DArray_Negative_NullArrayPtr") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMalloc3DArray_Negative_NullDescPtr") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMalloc3DArray_Negative_ZeroHeight") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMalloc3DArray_Negative_InvalidFlags") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMalloc3DArray_Negative_BadChannelLayout") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMalloc3DArray_Negative_8BitFloat") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMalloc3DArray_Negative_DifferentChannelSizes") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMalloc3DArray_Negative_BadChannelSize") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMalloc3DArray_Negative_NumericLimit") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - char") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - uchar2") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - short4") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - float2") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMalloc3DArray_Negative_Non2DTextureGather - float4") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDrvMemcpy3D_MultipleDataTypes - uint8_t") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDrvMemcpy3D_MultipleDataTypes - int") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDrvMemcpy3D_MultipleDataTypes - float") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDrvMemcpy3D_HosttoDevice") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDrvMemcpy3D_ExtentValidation") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - uint8_t") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - int") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDrvMemcpy3DAsync_MultipleDataTypes - float") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDrvMemcpy3DAsync_HosttoDevice") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDrvMemcpy3DAsync_ExtentValidation") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipPointerGetAttribute_MemoryTypes") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipPointerGetAttribute_KernelUpdation") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipPointerGetAttribute_BufferID") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipPointerGetAttribute_MappedMem") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipPointerGetAttribute_Negative") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDrvPtrGetAttributes_Negative") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDrvPtrGetAttributes_Functional") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMemGetInfo_DifferentMallocSmall") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMemGetInfo_ParaSmall") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipMemGetInfo_ParaNonDiv") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipMemGetInfo_ParaMultiSmall") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipMemGetInfo_Negative") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipFreeMultiTDev - char") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipFreeMultiTDev - int") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipFreeMultiTDev - float2") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipFreeMultiTDev - float4") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipFreeMultiTHost - char") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipFreeMultiTHost - int") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipFreeMultiTHost - float2") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipFreeMultiTHost - float4") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipFreeMultiTArray - char") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipFreeMultiTArray - int") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipFreeMultiTArray - float2") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipFreeMultiTArray - float4") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipMemset2DASyncMulti") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMemset3DASyncMulti") # SEGFAULT|Timeout|Timeout|Timeout|Failed|
list(APPEND FAILING_FOR_ALL "Unit_HMM_OverSubscriptionTst") # Timeout|Timeout|Timeout|Timeout|Timeout|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamGetFlags_Negative") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipMultiStream_multimeDevice") # Timeout|Timeout|Timeout|Timeout|Timeout|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamDestroy_Negative_DoubleDestroy") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamDestroy_Negative_NullStream") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamSynchronize_UninitializedStream") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamAddCallback_StrmSyncTiming") # Subprocess aborted|Subprocess aborted|Timeout|Timeout|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipEventIpc") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipEventSynchronize_Default_Positive") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipEventSynchronize_NoEventRecord_Positive") # Failed|Failed|Timeout|Timeout|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDeviceGetPCIBusId_Negative_PartialFill") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDeviceGetCacheConfig_Positive_Default") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDeviceGetCacheConfig_Positive_Basic") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDeviceGetCacheConfig_Positive_Threaded") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_HipDeviceGetCacheConfig_Negative_Parameters") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDeviceTotalMem_NegTst") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGetSetDeviceFlags_NullptrFlag") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGetSetDeviceFlags_InvalidFlag") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGetSetDeviceFlags_ValidFlag") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGetSetDeviceFlags_SetThenGet") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGetSetDeviceFlags_Threaded") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipGetDeviceFlags_Positive_Context") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGetSetDevice_MultiThreaded") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipSetGetDevice_Negative") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDeviceGetUuid_Positive") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDeviceGetUuid_Negative") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDeviceGetDefaultMemPool_Positive_Basic") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDeviceGetDefaultMemPool_Negative_Parameters") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDeviceSetLimit_SetGet") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDeviceGetSharedMemConfig_Positive_Default") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDeviceGetSharedMemConfig_Positive_Basic") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDeviceGetSharedMemConfig_Positive_Threaded") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipDeviceGetSharedMemConfig_Negative_Parameters") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDeviceReset_Positive_Basic") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDeviceReset_Positive_Threaded") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDeviceSetMemPool_Positive_Basic") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDeviceSetMemPool_Negative_Parameters") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDeviceGetMemPool_Positive_Default") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDeviceGetMemPool_Positive_Basic") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDeviceGetMemPool_Positive_Threaded") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDeviceGetMemPool_Negative_Parameters") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipDriverGetVersion_Negative") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipIpcOpenMemHandle_Negative_Open_In_Creating_Process") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipIpcOpenMemHandle_Negative_Open_In_Two_Contexts_Same_Device") # Failed|Failed|Timeout|Timeout|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipIpcGetMemHandle_Positive_Unique_Handles_Separate_Allocations") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipIpcGetMemHandle_Positive_Unique_Handles_Reused_Memory") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipIpcGetMemHandle_Negative_Handle_For_Freed_Memory") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipIpcGetMemHandle_Negative_Out_Of_Bound_Pointer") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipIpcCloseMemHandle_Positive_Reference_Counting") # Failed|Failed|Timeout|Timeout|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipIpcCloseMemHandle_Negative_Close_In_Originating_Process") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_printf_flags") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_printf_specifier") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipCreateTextureObject_ArgValidation") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipCreateTextureObject_LinearResource") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipCreateTextureObject_Pitch2DResource") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipGetChannelDesc_CreateAndGet") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipTextureObj1DCheckRGBAModes - array") # Failed|Failed|Failed|Failed|Timeout|
list(APPEND FAILING_FOR_ALL "Unit_hipTextureObj1DCheckSRGBAModes - array") # Failed|Failed|Failed|Failed|Timeout|
list(APPEND FAILING_FOR_ALL "Unit_hipTextureObj1DCheckRGBAModes - buffer") # Failed|Failed|Failed|Failed|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipTextureObj1DCheckSRGBAModes - buffer") # Failed|Failed|Failed|Failed|Timeout|
list(APPEND FAILING_FOR_ALL "Unit_hipTextureObj2DCheckRGBAModes") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipTextureObj2DCheckSRGBAModes") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipStreamPerThread_Basic") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipLaunchBounds_With_maxThreadsPerBlock_Check") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipLaunchBounds_With_maxThreadsPerBlock_blocksPerCU_Check") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGetLastError_Positive_Basic") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipGetLastError_Positive_Threaded") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "Unit_hipPeekAtLastError_Positive_Basic") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "Unit_hipPeekAtLastError_Positive_Threaded") # Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|Subprocess aborted|
list(APPEND FAILING_FOR_ALL "deviceMallocCompile") # Failed|Failed|Failed|Failed|Failed|
list(APPEND FAILING_FOR_ALL "hipStreamSemantics") # Timeout|Failed|Failed|Timeout|Failed|
list(APPEND FAILING_FOR_ALL "cuda-simpleCallback") # SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|SEGFAULT|



list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddEventRecordNode_MultipleRun") # Timeout
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddEventWaitNode_MultipleRun") # Timeout
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_BasicFunctional") # Timeout
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Relaxed") # Timeout
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_Memcpy - int") # Timeout
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_Memcpy - double") # Timeout
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemset_SetMemoryWithOffset") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetAsync_QueueJobsMultithreaded") # Timeout
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemcpy_MultiThreadWithSerialization") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_SmallSize_hipMemset") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_SmallSize_hipMemsetD32") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_SmallSize_hipMemsetD16") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_SmallSize_hipMemsetD8") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetASyncMulti") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int8_t") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int16_t") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - uint32_t") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamAddCallback_MultipleThreads") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipStreamAddCallback_ParamTst_Positive") # Timeout
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipEvent") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceSynchronize_Positive_Nullstream") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceSynchronize_Functional") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTextureFetch_vector") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj2D_Check") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - float") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - int") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - unsigned char") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - int16_t") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - char") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - unsigned int") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipCreateTextureObject_tex1DfetchVerification") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_tex1Dfetch_CheckModes") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj1DCheckModes") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj2DCheckModes") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMemFaultStackAllocation_Check") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMultiThreadStreams1_AsyncSync") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMultiThreadStreams1_AsyncAsync") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMultiThreadStreams1_AsyncSame") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "Unit_hipMultiThreadStreams2") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "hipMultiThreadAddCallback") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "BitonicSort") # Timeout
list(APPEND DGPU_OPENCL_FAILED_TESTS "FloydWarshall") # Timeout
list(APPEND DGPU_OPENCL_FAILED_TESTS "cuda-matrixMul") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "cuda-bandwidthTest") # Subprocess aborted
list(APPEND DGPU_OPENCL_FAILED_TESTS "cuda-convolutionSeparable") # Failed
list(APPEND DGPU_OPENCL_FAILED_TESTS "cuda-blackscholes") # Timeout
list(APPEND DGPU_OPENCL_FAILED_TESTS "cuda-scan") # Timeout
list(APPEND DGPU_OPENCL_FAILED_TESTS "cuda-FDTD3d") # Timeout



list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddEventRecordNode_MultipleRun") # Timeout
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipGraphAddEventWaitNode_MultipleRun") # Timeout
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_BasicFunctional") # Timeout
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_ColligatedStrmCapture_func") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Global") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_ThreadLocal") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Relaxed") # Timeout
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_Memcpy - int") # Timeout
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipHostRegister_Memcpy - double") # Timeout
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemset_SetMemoryWithOffset") # Timeout
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetAsync_SetMemoryWithOffset") # Timeout
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetAsync_QueueJobsMultithreaded") # Timeout
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemset3D_MemsetWithExtent") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemset3DAsync_ConcurrencyMthread") # Timeout
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemset2DAsync_MultiThread") # Timeout
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc_LoopRegressionAllocFreeCycles") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMalloc_Multithreaded_MultiGPU") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_PartialSet_1D") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_ZeroSize_3D") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetFunctional_PartialSet_3D") # Timeout
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetASyncMulti") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int8_t") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int16_t") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - uint32_t") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipHostMalloc_CoherentAccess") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMultiStream_sameDevice") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamAddCallback_MultipleThreads") # Timeout
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipStreamCreate_MultistreamBasicFunctionalities") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipEvent") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipEventRecord") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceSynchronize_Positive_Nullstream") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipDeviceSynchronize_Functional") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipTextureObj2DCheckModes") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMultiThreadStreams1_AsyncSync") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMultiThreadStreams1_AsyncAsync") # Timeout
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMultiThreadStreams1_AsyncSame") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipMultiThreadStreams2") # Timeout
list(APPEND IGPU_OPENCL_FAILED_TESTS "Unit_hipClassKernel_Overload_Override") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "TestHiprtcCPPKernels") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "TestRDCWithMultipleHipccCmds") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "TestStlFunctions") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "TestStlFunctionsDouble") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "TestAtomics") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "fp16") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "MatrixMultiply") # Timeout
list(APPEND IGPU_OPENCL_FAILED_TESTS "hip_async_binomial") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "BinomialOption") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "BitonicSort") # Timeout
list(APPEND IGPU_OPENCL_FAILED_TESTS "DCT") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "dwtHaar1D") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "FastWalshTransform") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "FloydWarshall") # Timeout
list(APPEND IGPU_OPENCL_FAILED_TESTS "graphMatrixMultiply") # Timeout
list(APPEND IGPU_OPENCL_FAILED_TESTS "cuda-matrixMul") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "cuda-bandwidthTest") # Subprocess aborted
list(APPEND IGPU_OPENCL_FAILED_TESTS "cuda-convolutionSeparable") # Timeout
list(APPEND IGPU_OPENCL_FAILED_TESTS "cuda-histogram") # Timeout
list(APPEND IGPU_OPENCL_FAILED_TESTS "cuda-binomialoptions") # Timeout
list(APPEND IGPU_OPENCL_FAILED_TESTS "cuda-blackscholes") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "cuda-qrng") # Timeout
list(APPEND IGPU_OPENCL_FAILED_TESTS "cuda-mergesort") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "cuda-scan") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "cuda-sortnet") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "cuda-FDTD3d") # Timeout
list(APPEND IGPU_OPENCL_FAILED_TESTS "cuda-reduction") # Failed
list(APPEND IGPU_OPENCL_FAILED_TESTS "cuda-fastwalsh") # Failed



list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipGraphAddEventRecordNode_MultipleRun") # Timeout
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipGraphAddEventWaitNode_MultipleRun") # Timeout
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipStreamBeginCapture_BasicFunctional") # Timeout
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipStreamBeginCapture_ColligatedStrmCapture_func") # Failed
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Global") # Failed
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_ThreadLocal") # Failed
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Relaxed") # Failed
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipHostRegister_Memcpy - int") # Timeout
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipHostRegister_Memcpy - double") # Timeout
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMemset_SetMemoryWithOffset") # Timeout
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMemsetAsync_SetMemoryWithOffset") # Timeout
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMemset3D_MemsetWithExtent") # Timeout
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMemset3DAsync_ConcurrencyMthread") # Timeout
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMemset2DAsync_MultiThread") # Timeout
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMemsetFunctional_PartialSet_3D") # Timeout
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMemsetASyncMulti") # Failed
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int8_t") # Failed
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int16_t") # Failed
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - uint32_t") # Failed
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipStreamAddCallback_MultipleThreads") # Subprocess aborted
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "TestAssert") # Failed
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "TestAssertFail") # Failed
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "abort") # Failed
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "hipMultiThreadAddCallback") # Timeout
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "hip_async_binomial") # Timeout
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "BitonicSort") # Timeout
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "FloydWarshall") # Timeout
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "cuda-bandwidthTest") # Timeout
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS "cuda-blackscholes") # Timeout



list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipGraphAddEventRecordNode_MultipleRun") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipGraphAddEventWaitNode_MultipleRun") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipStreamBeginCapture_BasicFunctional") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipStreamBeginCapture_ColligatedStrmCapture_func") # Failed
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Global") # Failed
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_ThreadLocal") # Failed
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Relaxed") # Failed
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipHostRegister_Memcpy - int") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipHostRegister_Memcpy - double") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMemset_SetMemoryWithOffset") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMemsetAsync_SetMemoryWithOffset") # Failed
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMemsetAsync_QueueJobsMultithreaded") # Subprocess aborted
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMemset3D_MemsetWithExtent") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMemset3DAsync_MemsetWithExtent") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMemset3DAsync_ConcurrencyMthread") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMemset2DAsync_WithKernel") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMemset2DAsync_MultiThread") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMemcpy_MultiThreadWithSerialization") # Subprocess aborted
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMemsetFunctional_ZeroSize_3D") # Failed
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMemsetFunctional_PartialSet_3D") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMemsetASyncMulti") # Failed
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int8_t") # Failed
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int16_t") # Failed
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - uint32_t") # Failed
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMultiStream_sameDevice") # Failed
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipStreamAddCallback_MultipleThreads") # Subprocess aborted
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipStreamCreate_MultistreamBasicFunctionalities") # Failed
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipEvent") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipEventRecord") # Failed
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipDeviceSynchronize_Positive_Nullstream") # Failed
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipDeviceSynchronize_Functional") # Failed
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipTextureFetch_vector") # Failed
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipTextureObj2DCheckModes") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMemFaultStackAllocation_Check") # Failed
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMultiThreadStreams1_AsyncSync") # Failed
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMultiThreadStreams1_AsyncAsync") # Subprocess aborted
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMultiThreadStreams1_AsyncSame") # Subprocess aborted
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "Unit_hipMultiThreadStreams2") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "TestWholeProgramCompilation") # Failed
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "TestAssert") # Failed
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "TestAssertFail") # Failed
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "abort") # Failed
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "fp16") # Subprocess aborted
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "MatrixMultiply") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "hipKernelLaunchIsNonBlocking") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "hipMultiThreadAddCallback") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "hip_async_binomial") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "BinomialOption") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "BitonicSort") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "FastWalshTransform") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "FloydWarshall") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "SimpleConvolution") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "graphMatrixMultiply") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "cuda-matrixMul") # Failed
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "cuda-bandwidthTest") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "cuda-convolutionSeparable") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "cuda-histogram") # Failed
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "cuda-binomialoptions") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "cuda-blackscholes") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "cuda-qrng") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "cuda-scan") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "cuda-sortnet") # Failed
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "cuda-FDTD3d") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "cuda-reduction") # Timeout
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS "cuda-fastwalsh") # Timeout



list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipGraphAddEventRecordNode_MultipleRun") # Timeout
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipGraphAddEventWaitNode_MultipleRun") # Timeout
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipStreamBeginCapture_BasicFunctional") # Timeout
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipStreamBeginCapture_ColligatedStrmCapture_func") # Failed
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Global") # Failed
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_ThreadLocal") # Failed
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Relaxed") # Failed
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipHostRegister_Memcpy - int") # Timeout
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipHostRegister_Memcpy - double") # Timeout
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemset_SetMemoryWithOffset") # Timeout
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemsetAsync_SetMemoryWithOffset") # Timeout
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemset3D_MemsetWithExtent") # Timeout
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemset3DAsync_ConcurrencyMthread") # Timeout
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemset2DAsync_MultiThread") # Timeout
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemsetFunctional_PartialSet_3D") # Timeout
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemsetASyncMulti") # Failed
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int8_t") # Failed
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int16_t") # Failed
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - uint32_t") # Failed
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipStreamAddCallback_MultipleThreads") # Subprocess aborted
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "TestAssert") # Failed
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "TestAssertFail") # Failed
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "abort") # Failed
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "hipMultiThreadAddCallback") # Timeout
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "hip_async_binomial") # Timeout
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "BitonicSort") # Timeout
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "FloydWarshall") # Timeout
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "cuda-bandwidthTest") # Timeout
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS "cuda-blackscholes") # Timeout



list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipGraphAddEventRecordNode_MultipleRun") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipGraphAddEventWaitNode_MultipleRun") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipStreamBeginCapture_BasicFunctional") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipStreamBeginCapture_ColligatedStrmCapture_func") # Failed
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Global") # Failed
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_ThreadLocal") # Failed
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipStreamBeginCapture_Multithreaded_Relaxed") # Failed
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipHostRegister_Memcpy - int") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipHostRegister_Memcpy - double") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemset_SetMemoryWithOffset") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemsetAsync_SetMemoryWithOffset") # Failed
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemsetAsync_QueueJobsMultithreaded") # Subprocess aborted
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemset3D_MemsetWithExtent") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemset3DAsync_MemsetWithExtent") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemset3DAsync_ConcurrencyMthread") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemset2DAsync_WithKernel") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemset2DAsync_MultiThread") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemcpy_MultiThreadWithSerialization") # Subprocess aborted
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemsetFunctional_ZeroSize_3D") # Failed
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemsetFunctional_PartialSet_3D") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemsetASyncMulti") # Failed
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int8_t") # Failed
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int16_t") # Failed
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - uint32_t") # Failed
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMultiStream_sameDevice") # Failed
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipStreamAddCallback_MultipleThreads") # Subprocess aborted
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipStreamCreate_MultistreamBasicFunctionalities") # Failed
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipEvent") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipEventRecord") # Failed
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipDeviceSynchronize_Positive_Nullstream") # Failed
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipDeviceSynchronize_Functional") # Failed
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipTextureFetch_vector") # Failed
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipTextureObj2DCheckModes") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMemFaultStackAllocation_Check") # Failed
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMultiThreadStreams1_AsyncSync") # Failed
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMultiThreadStreams1_AsyncAsync") # Subprocess aborted
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMultiThreadStreams1_AsyncSame") # Subprocess aborted
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "Unit_hipMultiThreadStreams2") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "TestWholeProgramCompilation") # Failed
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "TestAssert") # Failed
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "TestAssertFail") # Failed
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "abort") # Failed
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "fp16") # Subprocess aborted
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "MatrixMultiply") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "hipKernelLaunchIsNonBlocking") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "hipMultiThreadAddCallback") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "hip_async_binomial") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "BinomialOption") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "BitonicSort") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "FastWalshTransform") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "FloydWarshall") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "SimpleConvolution") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "graphMatrixMultiply") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "cuda-matrixMul") # Failed
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "cuda-bandwidthTest") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "cuda-convolutionSeparable") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "cuda-histogram") # Failed
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "cuda-binomialoptions") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "cuda-blackscholes") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "cuda-qrng") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "cuda-scan") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "cuda-sortnet") # Failed
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "cuda-FDTD3d") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "cuda-reduction") # Timeout
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS "cuda-fastwalsh") # Timeout



list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDeviceSynchronize_Positive_Nullstream") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipDeviceSynchronize_Functional") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipMultiThreadStreams2") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "Unit_hipClassKernel_Value") # Subprocess aborted
list(APPEND CPU_OPENCL_FAILED_TESTS "2d_shuffle") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "hipDynamicShared2") # Failed
list(APPEND CPU_OPENCL_FAILED_TESTS "cuda-reduction") # Timeout



list(APPEND CPU_POCL_FAILED_TESTS "Unit_deviceFunctions_CompileTest_atomicAdd_usigned_int") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemcpyWithStream_TestkindDefault") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemcpyAsync_KernelLaunch - float") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemcpyAsync_hipMultiMemcpyMultiThread - int") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemcpyAsync_hipMultiMemcpyMultiThread - double") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemsetASyncMulti") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int8_t") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - int16_t") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMemsetDASyncMulti - uint32_t") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDeviceSynchronize_Positive_Nullstream") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipDeviceSynchronize_Functional") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - int16_t") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipTexObjPitch_texture2D - unsigned int") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "Unit_hipMultiThreadStreams2") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "ABM_AddKernel_MultiTypeMultiSize - float") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "hipcc-TestHipComplexInclude") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "hipcc-TestHipccAcceptCcFiles") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "hipcc-TestHipccAcceptCppFiles") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "hipcc-Test513Regression") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "TestHipccNeedsDashO") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "hipcc-TestHipVersion") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "TestHipccHalfConversions") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "TestHipccHalfOperators") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "TestHipccCompileAndLink") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "TestHipccCompileThenLink") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "TestHipcc621") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "TestHipccAcceptCFiles") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "TestRDCWithSingleHipccCmd") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "TestRDCWithMultipleHipccCmds") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "TestWholeProgramCompilation") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "TestHipccFp16Include") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "TestHipccMultiSource") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "TestAssert") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "TestAssertFail") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "TestForgottenModuleUnload") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "TestIndirectCall") # Subprocess aborted
list(APPEND CPU_POCL_FAILED_TESTS "TestRuntimeWarnings") # Failed
list(APPEND CPU_POCL_FAILED_TESTS "abort") # Failed

list(APPEND DGPU_OPENCL_FAILED_TESTS ${FAILING_FOR_ALL})
list(APPEND IGPU_OPENCL_FAILED_TESTS ${FAILING_FOR_ALL})
list(APPEND CPU_OPENCL_FAILED_TESTS ${FAILING_FOR_ALL})
list(APPEND DGPU_LEVEL0_RCL_FAILED_TESTS ${FAILING_FOR_ALL} ${DGPU_LEVEL0_BASE_FAILED_TESTS} ${DGPU_LEVEL0_RCL_FAILED_TESTS})
list(APPEND DGPU_LEVEL0_ICL_FAILED_TESTS ${FAILING_FOR_ALL} ${DGPU_LEVEL0_BASE_FAILED_TESTS} ${DGPU_LEVEL0_ICL_FAILED_TESTS})
list(APPEND IGPU_LEVEL0_RCL_FAILED_TESTS ${FAILING_FOR_ALL} ${IGPU_LEVEL0_BASE_FAILED_TESTS} ${IGPU_LEVEL0_RCL_FAILED_TESTS})
list(APPEND IGPU_LEVEL0_ICL_FAILED_TESTS ${FAILING_FOR_ALL} ${IGPU_LEVEL0_BASE_FAILED_TESTS} ${IGPU_LEVEL0_ICL_FAILED_TESTS})
list(APPEND CPU_POCL_FAILED_TESTS ${FAILING_FOR_ALL})

string(REGEX REPLACE ";" "\$|" DGPU_OPENCL_FAILED_TESTS_STR "${DGPU_OPENCL_FAILED_TESTS}")
string(REGEX REPLACE ";" "\$|" IGPU_OPENCL_FAILED_TESTS_STR "${IGPU_OPENCL_FAILED_TESTS}")
string(REGEX REPLACE ";" "\$|"  CPU_OPENCL_FAILED_TESTS_STR "${CPU_OPENCL_FAILED_TESTS}")
string(REGEX REPLACE ";" "\$|" DGPU_LEVEL0_RCL_FAILED_TESTS_STR "${DGPU_LEVEL0_RCL_FAILED_TESTS}")
string(REGEX REPLACE ";" "\$|" DGPU_LEVEL0_ICL_FAILED_TESTS_STR "${DGPU_LEVEL0_ICL_FAILED_TESTS}")
string(REGEX REPLACE ";" "\$|" IGPU_LEVEL0_RCL_FAILED_TESTS_STR "${IGPU_LEVEL0_RCL_FAILED_TESTS}")
string(REGEX REPLACE ";" "\$|" IGPU_LEVEL0_ICL_FAILED_TESTS_STR "${IGPU_LEVEL0_ICL_FAILED_TESTS}")
string(REGEX REPLACE ";" "\$|"    CPU_POCL_FAILED_TESTS_STR "${CPU_POCL_FAILED_TESTS}")
string(REGEX REPLACE ";" "\$|" ALL_FAILED_TESTS_STR "${FAILING_FOR_ALL}")
string(CONCAT DGPU_OPENCL_FAILED_TESTS_STR ${DGPU_OPENCL_FAILED_TESTS_STR} "\$|")
string(CONCAT IGPU_OPENCL_FAILED_TESTS_STR ${IGPU_OPENCL_FAILED_TESTS_STR} "\$|")
string(CONCAT CPU_OPENCL_FAILED_TESTS_STR ${CPU_OPENCL_FAILED_TESTS_STR} "\$|")
string(CONCAT DGPU_LEVEL0_RCL_FAILED_TESTS_STR ${DGPU_LEVEL0_RCL_FAILED_TESTS_STR} "\$|")
string(CONCAT DGPU_LEVEL0_ICL_FAILED_TESTS_STR ${DGPU_LEVEL0_ICL_FAILED_TESTS_STR} "\$|")
string(CONCAT IGPU_LEVEL0_RCL_FAILED_TESTS_STR ${IGPU_LEVEL0_RCL_FAILED_TESTS_STR} "\$|")
string(CONCAT IGPU_LEVEL0_ICL_FAILED_TESTS_STR ${IGPU_LEVEL0_ICL_FAILED_TESTS_STR} "\$|")
string(CONCAT CPU_POCL_FAILED_TESTS_STR ${CPU_POCL_FAILED_TESTS_STR} "\$|")
string(CONCAT ALL_FAILED_TESTS_STR ${ALL_FAILED_TESTS_STR} "\$|")

FILE(WRITE "${CMAKE_BINARY_DIR}/test_lists/dgpu_opencl_failed_tests.txt" "\"${DGPU_OPENCL_FAILED_TESTS_STR}\"")
FILE(WRITE "${CMAKE_BINARY_DIR}/test_lists/igpu_opencl_failed_tests.txt" "\"${IGPU_OPENCL_FAILED_TESTS_STR}\"")
FILE(WRITE "${CMAKE_BINARY_DIR}/test_lists/cpu_opencl_failed_tests.txt" "\"${CPU_OPENCL_FAILED_TESTS_STR}\"")
FILE(WRITE "${CMAKE_BINARY_DIR}/test_lists/dgpu_level0_failed_reg_tests.txt" "\"${DGPU_LEVEL0_RCL_FAILED_TESTS_STR}\"")
FILE(WRITE "${CMAKE_BINARY_DIR}/test_lists/dgpu_level0_failed_imm_tests.txt" "\"${DGPU_LEVEL0_ICL_FAILED_TESTS_STR}\"")
FILE(WRITE "${CMAKE_BINARY_DIR}/test_lists/igpu_level0_failed_reg_tests.txt" "\"${IGPU_LEVEL0_RCL_FAILED_TESTS_STR}\"")
FILE(WRITE "${CMAKE_BINARY_DIR}/test_lists/igpu_level0_failed_imm_tests.txt" "\"${IGPU_LEVEL0_ICL_FAILED_TESTS_STR}\"")
FILE(WRITE "${CMAKE_BINARY_DIR}/test_lists/cpu_pocl_failed_tests.txt" "\"${CPU_POCL_FAILED_TESTS_STR}\"")
FILE(WRITE "${CMAKE_BINARY_DIR}/test_lists/all_failed_tests.txt" "\"${ALL_FAILED_TESTS_STR}\"")
