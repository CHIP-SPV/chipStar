add_custom_command(
  OUTPUT saxpy.o DEPFILE ${CMAKE_CURRENT_BINARY_DIR}/saxpy.o.d DEPENDS ${CMAKE_SOURCE_DIR}/samples/ccompat/saxpy.hip
  ${HIP_DEVICE_TOOLCHAIN_STAMP}
  COMMAND ${CMAKE_BINARY_DIR}/bin/hipcc
  ${CMAKE_SOURCE_DIR}/samples/ccompat/saxpy.hip -c -g -pthread
  -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_SOURCE_DIR}/HIP/include
  -MD -MF ${CMAKE_CURRENT_BINARY_DIR}/saxpy.o.d -MT ${CMAKE_CURRENT_BINARY_DIR}/saxpy.o)

add_custom_command(
  OUTPUT ccompat.o DEPFILE ${CMAKE_CURRENT_BINARY_DIR}/ccompat.o.d DEPENDS ${CMAKE_SOURCE_DIR}/samples/ccompat/ccompat.c
  ${HIP_DEVICE_TOOLCHAIN_STAMP}
  COMMAND ${CMAKE_BINARY_DIR}/bin/hipcc
  ${CMAKE_SOURCE_DIR}/samples/ccompat/ccompat.c -c -g -pthread
  -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_SOURCE_DIR}/HIP/include
  -MD -MF ${CMAKE_CURRENT_BINARY_DIR}/ccompat.o.d -MT ${CMAKE_CURRENT_BINARY_DIR}/ccompat.o)

add_custom_command(
OUTPUT ccompat DEPENDS ccompat.o saxpy.o $<TARGET_FILE:CHIP> $<TARGET_FILE:ocml_host_math_funcs>
COMMAND ${CMAKE_BINARY_DIR}/bin/hipcc 
./ccompat.o ./saxpy.o -o ccompat  -g -pthread)

add_custom_target(ccompatBuild ALL
  DEPENDS ccompat
)

add_dependencies(ccompatBuild hipcc.bin CHIP)
add_test(NAME ccompat COMMAND ${CMAKE_BINARY_DIR}/samples/ccompat/ccompat)

# Reproduce #2057: ccompat.o must rebuild when hipcc or .hipInfo changes.
if(CMAKE_GENERATOR MATCHES "Ninja")
  add_test(NAME TestFix2057CcompatToolchainStamp
    COMMAND ${CMAKE_MAKE_PROGRAM} -C ${CMAKE_BINARY_DIR} -t query samples/ccompat/ccompat.o)
  set_tests_properties(TestFix2057CcompatToolchainStamp PROPERTIES
    PASS_REGULAR_EXPRESSION "\n    (\\| )?hip_device_toolchain\\.stamp\n")
endif()
