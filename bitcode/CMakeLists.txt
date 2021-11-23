
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/BC")

#Ugly fix for interactions between clang13+ and igc
if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13)
	set(CLANG_CL_NO_STDINC_FLAG "")
else ()
	set(CLANG_CL_NO_STDINC_FLAG "-cl-no-stdinc")
endif ()

add_custom_command( OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/BC/mathlib.bc"
                    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mathlib.cl"
        COMMAND "${CMAKE_CXX_COMPILER}"
        "${CLANG_CL_NO_STDINC_FLAG}" -Xclang -finclude-default-header
        -O2 -x cl -cl-std=CL2.0
        --target=spir64-unknown-unknown -emit-llvm
        -o "${CMAKE_CURRENT_BINARY_DIR}/BC/mathlib.bc"
        -c "${CMAKE_CURRENT_SOURCE_DIR}/mathlib.cl"
        COMMENT "Building mathlib.bc"
        VERBATIM)
set(DEPEND_LIST "${CMAKE_CURRENT_BINARY_DIR}/BC/mathlib.bc")

#add_custom_command( OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/BC/mathlib.bc"
#                    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mathlib.bc"
#                    COMMAND "${CMAKE_COMMAND}" -E copy
#                    "${CMAKE_CURRENT_SOURCE_DIR}/mathlib.bc"
#                    "${CMAKE_CURRENT_BINARY_DIR}/BC/mathlib.bc"
#                    COMMENT "Copying mathlib.bc"
#                    VERBATIM)
#set(DEPEND_LIST "${CMAKE_CURRENT_BINARY_DIR}/BC/mathlib.bc")

# kernellib sources

set(SOURCES erfcinvD erfcinvF erfcxD erfcxF erfinvD erfinvF i0D i0F i1D i1F j0D j0F j1D j1F ncdfD ncdfF ncdfinvD ncdfinvF nearbyintD nearbyintF rcbrtD rcbrtF rhypotF rhypotD rlen3D rlen3F rlen4D rlen4F scalbD scalbF scalbnD scalbnF tables y0D y0F y1D y1F)
foreach(SOURCE IN LISTS SOURCES)
  add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/BC/${SOURCE}.bc"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/OCML/${SOURCE}.cl"
        COMMAND "${CMAKE_CXX_COMPILER}"
        "${CLANG_CL_NO_STDINC_FLAG}" -Xclang -finclude-default-header
        -O2 -pthread -x cl -cl-std=CL2.0
        --target=spir64-unknown-unknown -emit-llvm
        -o "${CMAKE_CURRENT_BINARY_DIR}/BC/${SOURCE}.bc"
        -c "${CMAKE_CURRENT_SOURCE_DIR}/OCML/${SOURCE}.cl"
        COMMENT "Building ${SOURCE}.bc"
        VERBATIM)
  list(APPEND DEPEND_LIST "${CMAKE_CURRENT_BINARY_DIR}/BC/${SOURCE}.bc")
endforeach()

# kernellib

add_custom_command(
        OUTPUT "${CMAKE_BINARY_DIR}/kernellib.bc"
        DEPENDS ${DEPEND_LIST}
        COMMAND "${LLVM_LINK}"
        -o "${CMAKE_BINARY_DIR}/kernellib.bc"
        ${DEPEND_LIST}
        COMMENT "Linking kernellib.bc"
        VERBATIM)

add_custom_target("kernellib_bc" DEPENDS "${CMAKE_BINARY_DIR}/kernellib.bc")

install(FILES "${CMAKE_BINARY_DIR}/kernellib.bc" DESTINATION share)

