# Reproduces #2209: a static CHIP exports -lSPIRV-Tools but neither installs
# the archive nor exports a search path for it. Only generated files are
# inspected; nothing is built.
#
# Invoked as:
#   cmake -DCHIPSTAR_SOURCE_DIR=<chipStar source> -DLLVM_CONFIG_BIN=<llvm-config>
#         -DGENERATOR=<generator> [-DOpenCL_LIBRARY=.. -DOpenCL_INCLUDE_DIR=..
#         -DLevelZero_LIBRARY=.. -DCMAKE_MAKE_PROGRAM=.. -DLLVM_SPIRV=..
#         -DCHIP_LLVM_USE_INTERGRATED_SPIRV=..]
#         -P TestFix2209StaticSpirvToolsExport.cmake
# CHIP_USE_NATIVE_VULKAN_SPIRV=ON only skips the test; it is not forwarded.

if(NOT CHIPSTAR_SOURCE_DIR)
  message(FATAL_ERROR "CHIPSTAR_SOURCE_DIR not set")
endif()
if(NOT GENERATOR)
  set(GENERATOR "Ninja")
endif()

if(NOT EXISTS "${LLVM_CONFIG_BIN}")
  message(STATUS "HIP_SKIP_THIS_TEST: no llvm-config at '${LLVM_CONFIG_BIN}'")
  return()
endif()
if(CHIP_USE_NATIVE_VULKAN_SPIRV)
  message(STATUS "HIP_SKIP_THIS_TEST: the nested configure would refetch VulkanMemoryAllocator")
  return()
endif()

string(RANDOM LENGTH 12 ALPHABET "abcdefghijklmnopqrstuvwxyz0123456789" SUFFIX)
set(SCRATCH_DIR "$ENV{TMPDIR}")
if(NOT SCRATCH_DIR)
  set(SCRATCH_DIR "/tmp")
endif()
set(SCRATCH_DIR "${SCRATCH_DIR}/chipstar-static-spirv-tools-${SUFFIX}")
cmake_path(NORMAL_PATH SCRATCH_DIR)
file(REMOVE_RECURSE "${SCRATCH_DIR}")
file(MAKE_DIRECTORY "${SCRATCH_DIR}")

# Fake package defining only SPIRV-Tools-static, with no plain SPIRV-Tools target.
set(PREFIX "${SCRATCH_DIR}/fake-spirv-tools")
file(WRITE "${PREFIX}/lib/libSPIRV-Tools.a" "")
file(MAKE_DIRECTORY "${PREFIX}/include")
file(WRITE "${PREFIX}/lib/cmake/SPIRV-Tools/SPIRV-ToolsConfig.cmake"
  "add_library(SPIRV-Tools-static STATIC IMPORTED)\n"
  "set_target_properties(SPIRV-Tools-static PROPERTIES\n"
  "  IMPORTED_LOCATION \"${PREFIX}/lib/libSPIRV-Tools.a\"\n"
  "  INTERFACE_INCLUDE_DIRECTORIES \"${PREFIX}/include\")\n")

# Configure a static CHIP with EXTRA args; the archive it must install is ARCHIVE.
function(check_case NAME ARCHIVE)
  set(PROBLEMS "")
  set(BUILD "${SCRATCH_DIR}/build-${NAME}")
  set(ARGS -G ${GENERATOR} -DCMAKE_BUILD_TYPE=Release -DLLVM_CONFIG_BIN=${LLVM_CONFIG_BIN}
           -DCHIP_BUILD_TESTS=OFF -DCHIP_BUILD_SHARED_LIBS=OFF ${ARGN})
  foreach(VAR OpenCL_LIBRARY OpenCL_INCLUDE_DIR LevelZero_LIBRARY CMAKE_MAKE_PROGRAM
              LLVM_SPIRV CHIP_LLVM_USE_INTERGRATED_SPIRV)
    if(DEFINED ${VAR})
      list(APPEND ARGS -D${VAR}=${${VAR}})
    endif()
  endforeach()
  execute_process(
    COMMAND ${CMAKE_COMMAND} ${ARGS} -S "${CHIPSTAR_SOURCE_DIR}" -B "${BUILD}"
    RESULT_VARIABLE RC OUTPUT_VARIABLE OUT ERROR_VARIABLE ERR)
  if(NOT RC EQUAL 0)
    message(FATAL_ERROR "${NAME}: configure failed (exit ${RC})\n${OUT}\n${ERR}\nscratch tree kept at ${SCRATCH_DIR}")
  endif()

  file(READ "${BUILD}/cmake_install.cmake" INSTALL_SCRIPT)
  string(FIND "${INSTALL_SCRIPT}" "${ARCHIVE}" POS)
  if(POS EQUAL -1)
    set(PROBLEMS "${PROBLEMS}\n  ${NAME}: cmake_install.cmake does not install ${ARCHIVE}")
  endif()

  file(GLOB_RECURSE EXPORTS "${BUILD}/CMakeFiles/Export/*CHIPTargets.cmake"
                            "${BUILD}/CMakeFiles/Export/*hip-targets.cmake")
  list(LENGTH EXPORTS N)
  if(NOT N EQUAL 2)
    set(PROBLEMS "${PROBLEMS}\n  ${NAME}: expected CHIPTargets.cmake and hip-targets.cmake, found: ${EXPORTS}")
  endif()
  foreach(EXPORT_FILE ${EXPORTS})
    file(READ "${EXPORT_FILE}" EXPORTED)
    file(STRINGS "${EXPORT_FILE}" DIRS_LINE REGEX "INTERFACE_LINK_DIRECTORIES")
    string(FIND "${DIRS_LINE}" "\${_IMPORT_PREFIX}/lib" POS)
    if(POS EQUAL -1)
      set(PROBLEMS "${PROBLEMS}\n  ${NAME}: ${EXPORT_FILE} INTERFACE_LINK_DIRECTORIES lacks \${_IMPORT_PREFIX}/lib: ${DIRS_LINE}")
    endif()
    string(FIND "${EXPORTED}" "$<LINK_ONLY:SPIRV-Tools>" POS)
    if(POS EQUAL -1)
      set(PROBLEMS "${PROBLEMS}\n  ${NAME}: ${EXPORT_FILE} lost $<LINK_ONLY:SPIRV-Tools>")
    endif()
  endforeach()
  set(FAILURES "${FAILURES}${PROBLEMS}" PARENT_SCOPE)
endfunction()

set(FAILURES "")
check_case(package "${PREFIX}/lib/libSPIRV-Tools.a" -DCMAKE_PREFIX_PATH=${PREFIX})
check_case(source "external/spirv-tools/lib/libSPIRV-Tools.a" -DCMAKE_DISABLE_FIND_PACKAGE_SPIRV-Tools=ON)

if(FAILURES)
  message(FATAL_ERROR "static SPIRV-Tools export problems:${FAILURES}\nscratch tree kept at ${SCRATCH_DIR}")
endif()

file(REMOVE_RECURSE "${SCRATCH_DIR}")
message(STATUS "TestFix2209StaticSpirvToolsExport passed")
