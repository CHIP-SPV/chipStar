# Reproduces #2205: find_package(SPIRV-Tools) succeeded on a package that only
# exports SPIRV-Tools-static and SPIRV-Tools-shared, so linking the name
# SPIRV-Tools became a bare -lSPIRV-Tools with no include directories.
#
# Invoked as:
#   cmake -DCHIPSTAR_SOURCE_DIR=<chipStar source> -DLLVM_CONFIG_BIN=<llvm-config>
#         -DGENERATOR=<generator> [-DOpenCL_LIBRARY=.. -DOpenCL_INCLUDE_DIR=..
#         -DLevelZero_LIBRARY=.. -DCHIP_USE_NATIVE_VULKAN_SPIRV=.. -DCMAKE_MAKE_PROGRAM=..]
#         -P TestFix2205SpirvToolsTarget.cmake

if(NOT CHIPSTAR_SOURCE_DIR)
  message(FATAL_ERROR "CHIPSTAR_SOURCE_DIR not set")
endif()
if(NOT GENERATOR)
  set(GENERATOR "Ninja")
endif()
if(NOT CMAKE_MAKE_PROGRAM)
  set(CMAKE_MAKE_PROGRAM ninja)
endif()

if(NOT EXISTS "${LLVM_CONFIG_BIN}")
  message(STATUS "HIP_SKIP_THIS_TEST: no llvm-config at '${LLVM_CONFIG_BIN}'")
  return()
endif()
if(NOT GENERATOR STREQUAL "Ninja")
  message(STATUS "HIP_SKIP_THIS_TEST: needs Ninja to read the link and compile commands")
  return()
endif()

string(RANDOM LENGTH 12 ALPHABET "abcdefghijklmnopqrstuvwxyz0123456789" SUFFIX)
set(SCRATCH_DIR "$ENV{TMPDIR}")
if(NOT SCRATCH_DIR)
  set(SCRATCH_DIR "/tmp")
endif()
set(SCRATCH_DIR "${SCRATCH_DIR}/chipstar-spirv-tools-target-${SUFFIX}")
file(REMOVE_RECURSE "${SCRATCH_DIR}")
file(MAKE_DIRECTORY "${SCRATCH_DIR}")

# Fake package defining only SPIRV-Tools-static, with no plain SPIRV-Tools target.
set(PREFIX "${SCRATCH_DIR}/fake-spirv-tools")
set(BUILD "${SCRATCH_DIR}/build")
file(WRITE "${PREFIX}/lib/libSPIRV-Tools.a" "")
file(MAKE_DIRECTORY "${PREFIX}/include")
file(WRITE "${PREFIX}/lib/cmake/SPIRV-Tools/SPIRV-ToolsConfig.cmake"
  "add_library(SPIRV-Tools-static STATIC IMPORTED)\n"
  "set_target_properties(SPIRV-Tools-static PROPERTIES\n"
  "  IMPORTED_LOCATION \"${PREFIX}/lib/libSPIRV-Tools.a\"\n"
  "  INTERFACE_INCLUDE_DIRECTORIES \"${PREFIX}/include\")\n"
  "if(TARGET SPIRV-Tools)\n"
  "  get_target_property(SPIRV-Tools_INCLUDE_DIRS SPIRV-Tools INTERFACE_INCLUDE_DIRECTORIES)\n"
  "endif()\n")

set(ARGS -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_CONFIG_BIN=${LLVM_CONFIG_BIN}
         -DCHIP_BUILD_TESTS=OFF -DCHIP_BUILD_SHARED_LIBS=OFF
         -DCMAKE_PREFIX_PATH=${PREFIX})
# Forward the outer backend configuration; leave a variable undefined when empty.
foreach(VAR OpenCL_LIBRARY OpenCL_INCLUDE_DIR LevelZero_LIBRARY
            CHIP_USE_NATIVE_VULKAN_SPIRV CMAKE_MAKE_PROGRAM)
  if(${VAR})
    list(APPEND ARGS -D${VAR}=${${VAR}})
  endif()
endforeach()
execute_process(
  COMMAND ${CMAKE_COMMAND} ${ARGS} -S "${CHIPSTAR_SOURCE_DIR}" -B "${BUILD}"
  RESULT_VARIABLE RC OUTPUT_VARIABLE OUT ERROR_VARIABLE ERR)
if(NOT RC EQUAL 0)
  message(FATAL_ERROR "configure failed (exit ${RC})\n${OUT}\n${ERR}\nscratch tree kept at ${SCRATCH_DIR}")
endif()

# The scratch prefix is unique, so match it plus the file name instead of a
# normalized path.
set(FAILURES "")
function(commands_must_contain TARGET NEEDLE WHAT)
  execute_process(
    COMMAND ${CMAKE_MAKE_PROGRAM} -C "${BUILD}" -t commands ${TARGET}
    RESULT_VARIABLE RC OUTPUT_VARIABLE CMDS ERROR_VARIABLE CERR)
  if(NOT RC EQUAL 0)
    set(FAILURES "${FAILURES}\n  ninja -t commands ${TARGET} failed: ${CERR}" PARENT_SCOPE)
  elseif(NOT CMDS MATCHES "${NEEDLE}")
    set(FAILURES "${FAILURES}\n  ${TARGET}: ${WHAT}" PARENT_SCOPE)
  endif()
endfunction()

commands_must_contain(spirv-extractor "${PREFIX}[^ ]*/lib/libSPIRV-Tools\\.a"
  "does not link the package's libSPIRV-Tools.a")
commands_must_contain(spirv-extractor "(-I|-isystem )${PREFIX}[^ ]*/include"
  "does not compile with the package's include directory")
commands_must_contain(CHIP "(-I|-isystem )${PREFIX}[^ ]*/include"
  "does not compile with the package's include directory")

file(GLOB_RECURSE EXPORTS "${BUILD}/CMakeFiles/Export/*CHIPTargets.cmake")
if(NOT EXPORTS)
  set(FAILURES "${FAILURES}\n  no exported CHIPTargets.cmake under ${BUILD}")
else()
  file(READ "${EXPORTS}" EXPORTED)
  if(EXPORTED MATCHES "SPIRV-Tools-static")
    set(FAILURES "${FAILURES}\n  CHIPTargets.cmake mentions SPIRV-Tools-static")
  endif()
  if(NOT EXPORTED MATCHES "\\$<LINK_ONLY:SPIRV-Tools>")
    set(FAILURES "${FAILURES}\n  CHIPTargets.cmake lacks $<LINK_ONLY:SPIRV-Tools>")
  endif()
endif()

if(FAILURES)
  message(FATAL_ERROR "SPIRV-Tools target problems:${FAILURES}\nscratch tree kept at ${SCRATCH_DIR}")
endif()

file(REMOVE_RECURSE "${SCRATCH_DIR}")
message(STATUS "TestFix2205SpirvToolsTarget passed")
