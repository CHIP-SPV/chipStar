
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/BC")

#Ugly fix for interactions between clang13+ and igc
if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13)
	set(CLANG_CL_NO_STDINC_FLAG "")
else ()
	set(CLANG_CL_NO_STDINC_FLAG "-cl-no-stdinc")
endif ()

if(NOT DEFINED LLVM_VERSION)
  message(FATAL "Could not determine LLVM version.")
endif()

if("${LLVM_VERSION}" VERSION_LESS 14.0)
  # Definitions for pre-upstreamed HIP-Clang.
  set(BC_TRIPLE "spir64-unknown-unknown")
  set(BC_FILE "kernellib.bc")
  set(BC_DESTINATION share)
else()
  # HIP-Clang automatically looks for hipspv-<triple>.bc in
  # <bc-destination> directory in HIP compilation mode.
  set(BC_TRIPLE "spirv64")
  set(BC_FILE "hipspv-${BC_TRIPLE}.bc")
  set(BC_DESTINATION lib/hip-device-lib)
endif()

# Support function(s) for printf().
add_custom_command( OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/BC/printf_support.bc"
                    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/_cl_print_str.cl"
        COMMAND "${CMAKE_CXX_COMPILER}"
        "${CLANG_CL_NO_STDINC_FLAG}" -Xclang -finclude-default-header
        -O2 -x cl -cl-std=CL2.0
        --target=spir64-unknown-unknown -emit-llvm
        -o "${CMAKE_CURRENT_BINARY_DIR}/BC/printf_support.bc"
        -c "${CMAKE_CURRENT_SOURCE_DIR}/_cl_print_str.cl"
        COMMENT "Building printf_support.bc"
        VERBATIM)
list(APPEND DEPEND_LIST "${CMAKE_CURRENT_BINARY_DIR}/BC/printf_support.bc")

add_custom_command( OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/BC/texture.bc"
                    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/texture.cl"
        COMMAND "${CMAKE_CXX_COMPILER}"
        "${CLANG_CL_NO_STDINC_FLAG}" -Xclang -finclude-default-header
        -O2 -x cl -cl-std=CL2.0
        --target=${BC_TRIPLE} -emit-llvm
        -o "${CMAKE_CURRENT_BINARY_DIR}/BC/texture.bc"
        -c "${CMAKE_CURRENT_SOURCE_DIR}/texture.cl"
        COMMENT "Building texture.bc"
        VERBATIM)
list(APPEND DEPEND_LIST "${CMAKE_CURRENT_BINARY_DIR}/BC/texture.bc")

add_custom_command(
        OUTPUT "${CMAKE_BINARY_DIR}/${BC_DESTINATION}/${BC_FILE}"
        DEPENDS ${DEPEND_LIST}
        COMMAND ${CMAKE_COMMAND} -E make_directory
        "${CMAKE_BINARY_DIR}/${BC_DESTINATION}"
        COMMAND "${LLVM_LINK}"
        -o "${CMAKE_BINARY_DIR}/${BC_DESTINATION}/${BC_FILE}"
        ${DEPEND_LIST}
        COMMENT "Linking device library bitcode '${BC_FILE}'"
        VERBATIM)

add_custom_target("kernellib_bc"
  DEPENDS 
  "${CMAKE_BINARY_DIR}/${BC_DESTINATION}/${BC_FILE}"
  opencl ocml ockl
  )

install(FILES "${CMAKE_BINARY_DIR}/${BC_DESTINATION}/${BC_FILE}"
  DESTINATION ${BC_DESTINATION})

