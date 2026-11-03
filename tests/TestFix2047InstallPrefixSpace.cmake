# hipcc and cucc must compile and link from an install prefix containing a space.
set(WORK "${WORK_DIR}")
file(REMOVE_RECURSE "${WORK}")
set(PREFIX "${WORK}/with space")
execute_process(COMMAND ${CMAKE_COMMAND} --install "${BUILD_DIR}" --prefix "${PREFIX}"
  OUTPUT_VARIABLE OUT ERROR_VARIABLE OUT RESULT_VARIABLE RC)
if(NOT RC EQUAL 0)
  message(FATAL_ERROR "install failed: ${OUT}")
endif()

set(SRC "#include <hip/hip_runtime.h>\n__global__ void k(int *p) { *p = 1; }\nint main() { int *p = nullptr; k<<<1, 1>>>(p); return 0; }\n")
file(WRITE "${WORK}/t.hip" "${SRC}")
file(WRITE "${WORK}/t.cu" "${SRC}")

set(DRIVERS hipcc cucc)
foreach(DRIVER ${DRIVERS})
  foreach(EXT hip cu)
    execute_process(COMMAND "${PREFIX}/bin/${DRIVER}" -c "${WORK}/t.${EXT}" -o "${WORK}/${DRIVER}-${EXT}.o"
      OUTPUT_VARIABLE OUT ERROR_VARIABLE OUT RESULT_VARIABLE RC)
    if(NOT RC EQUAL 0)
      message(FATAL_ERROR "${DRIVER} compile of t.${EXT} failed: ${OUT}")
    endif()
    execute_process(COMMAND "${PREFIX}/bin/${DRIVER}" "${WORK}/${DRIVER}-${EXT}.o" -o "${WORK}/${DRIVER}-${EXT}"
      OUTPUT_VARIABLE OUT ERROR_VARIABLE OUT RESULT_VARIABLE RC)
    if(NOT RC EQUAL 0)
      message(FATAL_ERROR "${DRIVER} link of t.${EXT} failed: ${OUT}")
    endif()
  endforeach()
endforeach()
