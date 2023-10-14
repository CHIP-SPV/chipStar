# usage: add_cuda_sample_args(NAME test-name SUBDIR test-dir SOURCES source1 source2 ARGS command-arg1 arg2 arg3)
function(add_cuda_sample_args)
  set(options)
  set(oneValueArgs NAME SUBDIR)
  set(multiValueArgs SOURCES ARGS)
  cmake_parse_arguments(CUDA "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
  set(SOURCES)
  foreach(SOURCE IN LISTS CUDA_SOURCES)
    list(APPEND SOURCES "${CUDA_SUBDIR}/${SOURCE}")
  endforeach()

  set(EXEC_NAME ${CUDA_NAME})
  set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
  add_executable("${EXEC_NAME}" ${SOURCES})

  target_compile_options("${EXEC_NAME}" PRIVATE ${OFFLOAD_ARCH_STR})

  set_target_properties("${EXEC_NAME}" PROPERTIES CXX_STANDARD_REQUIRED ON)

  target_link_libraries("${EXEC_NAME}" CHIP deviceInternal)
  target_include_directories("${EXEC_NAME}"
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Common ${CUDA_SUBDIR})
  install(TARGETS "${EXEC_NAME}"
    RUNTIME DESTINATION "${CHIP_SAMPLE_BINDIR}/cuda_samples")

  add_test(NAME "${CUDA_NAME}"
    COMMAND "${CMAKE_CURRENT_BINARY_DIR}/${EXEC_NAME}" ${CUDA_ARGS})
  set_property(TEST "${CUDA_NAME}" PROPERTY LABELS "internal;cuda")

endfunction()

# ARGN = additional source files
function(add_cuda_sample TEST_NAME SUBDIR MAIN_SOURCE)
  add_cuda_sample_args(NAME "${TEST_NAME}"
                       SUBDIR "${SUBDIR}"
                       SOURCES ${MAIN_SOURCE} ${ARGN})
endfunction()

############################
## 0_Simple
add_cuda_sample(cuda-asyncAPI 0_Simple/asyncAPI asyncAPI.cu)
add_cuda_sample(cuda-lambda 0_Simple/lambda lambda.cu)
add_cuda_sample(cuda-matrixMul 0_Simple/matrixMul matrixMul.cu)
add_cuda_sample(cuda-template 0_Simple/template template.cu template_cpu.cpp)
add_cuda_sample(cuda-vectorAdd 0_Simple/vectorAdd vectorAdd.cu)

add_cuda_sample(cuda-clock 0_Simple/clock clock.cu)
add_cuda_sample(cuda-cppIntegration 0_Simple/cppIntegration cppIntegration.cu  cppIntegration_gold.cpp  main.cpp)

add_cuda_sample(cuda-simplePrintf 0_Simple/simplePrintf  simplePrintf.cu)
add_cuda_sample(cuda-simpleAtomicIntrinsics 0_Simple/simpleAtomicIntrinsics  simpleAtomicIntrinsics_cpu.cpp  simpleAtomicIntrinsics.cu)
add_cuda_sample(cuda-simpleTemplates 0_Simple/simpleTemplates  simpleTemplates.cu)
add_cuda_sample(cuda-simpleCallback 0_Simple/simpleCallback  multithreading.cpp   simpleCallback.cu)

############################
## 1_Utilities
add_cuda_sample(cuda-bandwidthTest 1_Utilities/bandwidthTest bandwidthTest.cu)
add_cuda_sample(cuda-deviceQuery 1_Utilities/deviceQuery deviceQuery.cpp)

############################
## 2_Graphics

add_cuda_sample(cuda-convolutionSeparable 2_Graphics/convolutionSeparable convolutionSeparable.cu  convolutionSeparable_gold.cpp  main.cpp)

#### works after editing sources: threadblock size 512 -> 256
add_cuda_sample_args(NAME "cuda-dwtHaar1D"
                     SUBDIR "2_Graphics/dwtHaar1D"
                     SOURCES "dwtHaar1D.cu"
                     ARGS "--signal=${CMAKE_CURRENT_SOURCE_DIR}/2_Graphics/dwtHaar1D/data/signal.dat"
                     "--gold=${CMAKE_CURRENT_SOURCE_DIR}/2_Graphics/dwtHaar1D/data/regression.gold.dat"
                     "--result=${CMAKE_BINARY_DIR}/dwtHaar1D_result.dat"
                     )

add_cuda_sample(cuda-histogram  2_Graphics/histogram  histogram_gold.cpp  main.cpp  histogram256.cu  histogram64.cu)

############################

## 4_Finance
add_cuda_sample(cuda-binomialoptions  4_Finance/binomialOptions binomialOptions.cpp  binomialOptions_gold.cpp  binomialOptions_kernel.cu)
add_cuda_sample(cuda-blackscholes  4_Finance/BlackScholes BlackScholes.cu  BlackScholes_gold.cpp)
add_cuda_sample(cuda-qrng  4_Finance/quasirandomGenerator quasirandomGenerator.cpp  quasirandomGenerator_gold.cpp  quasirandomGenerator_kernel.cu)

############################

## 6_Advanced
add_cuda_sample(cuda-mergesort  6_Advanced/mergeSort bitonic.cu  main.cpp  mergeSort.cu  mergeSort_host.cpp  mergeSort_validate.cpp)
add_cuda_sample(cuda-scalarprod  6_Advanced/scalarProd scalarProd_cpu.cpp  scalarProd.cu)
add_cuda_sample(cuda-scan  6_Advanced/scan scan.cu  main.cpp scan_gold.cpp)
add_cuda_sample(cuda-sortnet  6_Advanced/sortingNetworks bitonicSort.cu  main.cpp  oddEvenMergeSort.cu  sortingNetworks_validate.cpp)
add_cuda_sample(cuda-FDTD3d  6_Advanced/FDTD3d FDTD3dReference.cpp FDTD3dGPU.cu FDTD3d.cpp)
add_cuda_sample(cuda-sobolqrng  4_Finance/SobolQRNG sobol.cpp  sobol_gold.cpp  sobol_gpu.cu  sobol_primitives.cpp)
add_cuda_sample(cuda-reduction  6_Advanced/reduction reduction.cpp  reduction_kernel.cu)
add_cuda_sample(cuda-fastwalsh  6_Advanced/fastWalshTransform fastWalshTransform.cu  fastWalshTransform_gold.cpp)

############################

# if(NOT CHIP_ENABLE_FAILING_TESTS)
#   ##### when run, sometimes fails sometimes passes TODO investigate
#   set_tests_properties("cuda-simpleCallback" PROPERTIES DISABLED 1 LABELS "")
# endif()

if(CHIP_ENABLE_UNCOMPILABLE_TESTS)
  # cppOverload: DISABLED: Due to the following error:
  #
  # ../cppOverload.cu:138:11: error: assigning to 'void (*)(const int *, int *, int)' from incompatible type '<overloaded function type>'
  #  func1 = simple_kernel;
  #          ^~~~~~~~~~~~~
  #
  # The correct overload is not compatible for some unknown reason. It might be
  # a Clang issue which only appear in HIP-SPIRV path.
  #
  add_cuda_sample(cuda-cppOverload 0_Simple/cppOverload cppOverload.cu)

  ### hipErrorNotSupported (Called a function which is not implemented) in /home/michal/0/source/chip-spv-test/src/CHIPBindings.cc:995:hipSetDeviceFlags
  add_cuda_sample(cuda-simpleStreams 0_Simple/simpleStreams  simpleStreams.cu)

  #### use of undeclared identifier 'all', 'any'
  add_cuda_sample(cuda-simpleVoteIntrinsics 0_Simple/simpleVoteIntrinsics  simpleVoteIntrinsics.cu)

  #### use of undeclared identifier 'CU_MEMHOSTALLOC_DEVICEMAP'
  add_cuda_sample(cuda-simpleZeroCopy 0_Simple/simpleZeroCopy  simpleZeroCopy.cu)

  ##### requires assert (different from abort)
  add_cuda_sample(cuda-simpleAssert 0_Simple/simpleAssert  simpleAssert.cu)

  #### requires function pointers
  add_cuda_sample(cuda-simpleSeparateCompilation 0_Simple/simpleSeparateCompilation  simpleDeviceLibrary.cu  simpleSeparateCompilation.cu)

  #### requires surfaces
  add_cuda_sample(cuda-simpleSurfaceWrite 0_Simple/simpleSurfaceWrite  simpleSurfaceWrite.cu)

  #### requires textures
  add_cuda_sample(cuda-simpleTexture 0_Simple/simpleTexture  simpleTexture.cu)

  #### requires texture support: "Don't know how to lower this texture use case."
  add_cuda_sample(cuda-convolutionTexture  2_Graphics/convolutionTexture  convolutionTexture.cu  convolutionTexture_gold.cpp  main.cpp)

  #### requires texture support: "Don't know how to lower this texture use case."
  add_cuda_sample(cuda-dct8x8  2_Graphics/dct8x8  BmpUtil.cpp dct8x8.cu  DCT8x8_Gold.cpp )

  #### hipErrorLaunchFailure (chip_kernels is empty for this device)
  ## llvm translator BUG: produces invalid SPIR-V:
  ## 1st operand of Capability: operand FPFastMathModeINTEL(5837) requires one of these extensions: SPV_INTEL_fp_fast_math_mode
  add_cuda_sample(cuda-dxtc  2_Graphics/dxtc  dxtc.cu)

  #### requires texture support: "Don't know how to lower this texture use case."
  add_cuda_sample(cuda-HSOpticalFlow  2_Graphics/HSOpticalFlow  flowCUDA.cu flowGold.cpp  main.cpp)

  #### requires texture support: "Don't know how to lower this texture use case."
  add_cuda_sample(cuda-stereoDisparity  2_Graphics/stereoDisparity  stereoDisparity.cu)

  # 6_Advanced/shfl_scan/shfl_integral_image.cuh:31:33: error: use of undeclared identifier 'warpSize'
  # 6_Advanced/shfl_scan/shfl_integral_image.cuh:83:17: error: no matching function for call to '__shfl_up'
  # error: no matching function for call to '__shfl_xor'
  add_cuda_sample(cuda-shflscan  6_Advanced/shfl_scan shfl_scan.cu)

  # requires function pointers:    kernel = &copySharedMem;
  add_cuda_sample(cuda-transpose 6_Advanced/transpose transpose.cu)

endif()

