function(add_hip_test MAIN_SOURCE)
  get_filename_component(EXEC_NAME ${MAIN_SOURCE} NAME_WLE)
  set_source_files_properties(${MAIN_SOURCE} PROPERTIES LANGUAGE CXX)
  add_executable("${EXEC_NAME}" EXCLUDE_FROM_ALL ${MAIN_SOURCE})
  set_target_properties("${EXEC_NAME}" PROPERTIES CXX_STANDARD_REQUIRED ON)

  target_link_libraries("${EXEC_NAME}" CHIP deviceInternal)
  target_include_directories("${EXEC_NAME}"
    PUBLIC ${CMAKE_SOURCE_DIR}/HIP/include ${CMAKE_SOURCE_DIR}/include)

  add_test(NAME ${EXEC_NAME} COMMAND ${SKIP_DOUBLE_TESTS} ${CMAKE_CURRENT_BINARY_DIR}/${EXEC_NAME})
  set_tests_properties("${EXEC_NAME}" PROPERTIES
    SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST")
  add_dependencies(build_tests "${EXEC_NAME}")
endfunction()

add_subdirectory(compiler)
add_subdirectory(cuda)
add_subdirectory(devicelib)
add_subdirectory(fromLibCeed)
add_subdirectory(hiprtc)
add_subdirectory(post-install)
add_subdirectory(runtime)
add_subdirectory(libraries)
add_subdirectory(benchmarks)
add_subdirectory(regression)
add_subdirectory(hipcc-verify)

# Reproducer for CHIP-SPV/chipStar#1488. clang loads lib/libLLVMHipSpvPasses.so
# as a pass plugin (through --hip-path) while compiling HIP device code, so
# every object compiled from a HIP source has to be ordered after that link.
# The walk follows order-only inputs, which is where the Ninja generator puts
# target-level dependencies. One probe per registration path: the Test* glob
# in runtime/, add_hip_test(), and the catch suite, whose own CMake tree
# compiles with hipcc.bin and never links CHIP.
if(CMAKE_GENERATOR MATCHES "Ninja")
  find_package(Python3 COMPONENTS Interpreter)
endif()
if(CMAKE_GENERATOR MATCHES "Ninja" AND Python3_Interpreter_FOUND)
  set(BUILD_ORDER_PROBES
    $<TARGET_OBJECTS:TestKernelArgs>
    $<TARGET_OBJECTS:TestFixUmul64hi>)
  if(TARGET ErrorHandlingTest)
    list(APPEND BUILD_ORDER_PROBES $<TARGET_OBJECTS:ErrorHandlingTest>)
  endif()
  add_test(NAME TestFix1488PassPluginBuildOrder
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_build_order.py
      --ninja ${CMAKE_MAKE_PROGRAM} --build-dir ${CMAKE_BINARY_DIR}
      --required $<TARGET_FILE:LLVMHipPasses> ${BUILD_ORDER_PROBES})
endif()

# Reproduce #1486: the SPIRV-Tools ExternalProject named its compilers by bare
# name, so a PATH that differs between two builds made cmake wipe the
# external's cache and `ninja install` wrote to /usr/local. Script test:
# configures the module on its own and drives the generated configure step.
add_test(NAME TestFix1486SpirvToolsInstallPrefix
  COMMAND ${CMAKE_COMMAND}
    -DCHIPSTAR_SOURCE_DIR=${CMAKE_SOURCE_DIR}
    -DGENERATOR=${CMAKE_GENERATOR}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/TestFix1486SpirvToolsInstallPrefix.cmake)
set_tests_properties(TestFix1486SpirvToolsInstallPrefix PROPERTIES TIMEOUT 300)
