# Installs chipStar to a scratch prefix and builds a CUDAToolkit consumer
# with that prefix's cucc, once through the toolchain file and once with only
# CMAKE_CUDA_COMPILER.
set(PREFIX "${WORK_DIR}/prefix")
file(REMOVE_RECURSE "${WORK_DIR}")
execute_process(COMMAND ${CMAKE_COMMAND} --install "${BUILD_DIR}" --prefix "${PREFIX}"
  OUTPUT_QUIET RESULT_VARIABLE RC)
if(NOT RC EQUAL 0)
  message(FATAL_ERROR "install failed")
endif()

function(run_mode MODE)
  set(B "${WORK_DIR}/${MODE}")
  execute_process(COMMAND ${CMAKE_COMMAND} -G "${GENERATOR}" -S "${CONSUMER_DIR}" -B "${B}"
    ${ARGN} RESULT_VARIABLE RC)
  if(NOT RC EQUAL 0)
    message(FATAL_ERROR "${MODE}: configure failed")
  endif()
  execute_process(COMMAND ${CMAKE_COMMAND} --build "${B}" RESULT_VARIABLE RC)
  if(NOT RC EQUAL 0)
    message(FATAL_ERROR "${MODE}: build failed")
  endif()
  execute_process(COMMAND env -u LD_LIBRARY_PATH "${B}/t"
    OUTPUT_VARIABLE OUT RESULT_VARIABLE RC)
  if(NOT RC EQUAL 0 OR NOT OUT MATCHES "PASSED")
    message(FATAL_ERROR "${MODE}: run failed: ${OUT}")
  endif()
endfunction()

run_mode(toolchain
  -DCMAKE_TOOLCHAIN_FILE=${PREFIX}/share/cmake/chipstar/cuda-toolchain.cmake
  -DCMAKE_CUDA_COMPILER=${PREFIX}/bin/cucc)
# The toolchain file must also keep a system CUDA toolkit's libcudart out.
file(STRINGS "${WORK_DIR}/toolchain/CMakeCache.txt" CUDART REGEX "^CUDA_CUDART:")
if(NOT CUDART MATCHES "=${PREFIX}/")
  message(FATAL_ERROR "toolchain: cudart is not from the chipStar prefix: ${CUDART}")
endif()

run_mode(plain -DCMAKE_CUDA_COMPILER=${PREFIX}/bin/cucc)
