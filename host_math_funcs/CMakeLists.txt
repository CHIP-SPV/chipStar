message(STATUS "Building host math functions")
set(ROCm-Device-Libs_SRC_DIR ${CMAKE_SOURCE_DIR}/bitcode/ROCm-Device-Libs)
set(OCML_SRC_DIR ${ROCm-Device-Libs_SRC_DIR}/ocml/src)
set(OCML_INCLUDE_DIR ${ROCm-Device-Libs_SRC_DIR}/ocml/include)
set(OCML_LIB_DIR ${ROCm-Device-Libs_SRC_DIR}/ocml/lib)

# Add include directories for OCML
include_directories(
  ${OCML_INCLUDE_DIR}
  ${ROCm-Device-Libs_SRC_DIR}/irif/inc
  ${ROCm-Device-Libs_SRC_DIR}/oclc/inc
  ${ROCm-Device-Libs_SRC_DIR}/ocml/inc
  ${CMAKE_CURRENT_SOURCE_DIR}
)


# Create host_math_funcs directory in build
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/host_math_funcs)

# Copy ncdfF.cl to the build directory

# cospi			OpenCL in chipStar (extern)
# cospif			OpenCL in chipStar (calling cospi)
configure_file(${OCML_SRC_DIR}/trigredF.h ${CMAKE_BINARY_DIR}/host_math_funcs/trigredF.h COPYONLY)
configure_file(${OCML_SRC_DIR}/sincospiredF.cl ${CMAKE_BINARY_DIR}/host_math_funcs/sincospiredF.cl COPYONLY)
configure_file(${OCML_SRC_DIR}/tanpiredF.cl ${CMAKE_BINARY_DIR}/host_math_funcs/tanpiredF.cl COPYONLY)
configure_file(${OCML_SRC_DIR}/trigpiredF.h ${CMAKE_BINARY_DIR}/host_math_funcs/trigpiredF.h COPYONLY)
configure_file(${OCML_SRC_DIR}/trigpiredF.cl ${CMAKE_BINARY_DIR}/host_math_funcs/trigpiredF.cl COPYONLY)
configure_file(${OCML_SRC_DIR}/cospiF.cl ${CMAKE_BINARY_DIR}/host_math_funcs/cospiF.cl COPYONLY)

configure_file(${OCML_SRC_DIR}/trigpiredD.h ${CMAKE_BINARY_DIR}/host_math_funcs/trigpiredD.h COPYONLY)
configure_file(${OCML_SRC_DIR}/sincospiredD.cl ${CMAKE_BINARY_DIR}/host_math_funcs/sincospiredD.cl COPYONLY)
configure_file(${OCML_SRC_DIR}/tanpiredD.cl ${CMAKE_BINARY_DIR}/host_math_funcs/tanpiredD.cl COPYONLY)
configure_file(${OCML_SRC_DIR}/trigpiredD.cl ${CMAKE_BINARY_DIR}/host_math_funcs/trigpiredD.cl COPYONLY)
configure_file(${OCML_SRC_DIR}/cospiD.cl ${CMAKE_BINARY_DIR}/host_math_funcs/cospiD.cl COPYONLY)

configure_file(${OCML_SRC_DIR}/erfcD.cl ${CMAKE_BINARY_DIR}/host_math_funcs/erfcD.cl COPYONLY)
configure_file(${OCML_SRC_DIR}/erfcF.cl ${CMAKE_BINARY_DIR}/host_math_funcs/erfcF.cl COPYONLY)

configure_file(${OCML_SRC_DIR}/expD_base.h ${CMAKE_BINARY_DIR}/host_math_funcs/expD_base.h COPYONLY)
configure_file(${OCML_SRC_DIR}/expF_base.h ${CMAKE_BINARY_DIR}/host_math_funcs/expF_base.h COPYONLY)
configure_file(${OCML_SRC_DIR}/logD_base.h ${CMAKE_BINARY_DIR}/host_math_funcs/logD_base.h COPYONLY)
configure_file(${OCML_SRC_DIR}/logF_base.h ${CMAKE_BINARY_DIR}/host_math_funcs/logF_base.h COPYONLY)
configure_file(${OCML_SRC_DIR}/logD.cl ${CMAKE_BINARY_DIR}/host_math_funcs/logD.cl COPYONLY)
configure_file(${OCML_SRC_DIR}/logF.cl ${CMAKE_BINARY_DIR}/host_math_funcs/logF.cl COPYONLY)


configure_file(${OCML_SRC_DIR}/expD.cl ${CMAKE_BINARY_DIR}/host_math_funcs/expD.cl COPYONLY)
configure_file(${OCML_SRC_DIR}/expF.cl ${CMAKE_BINARY_DIR}/host_math_funcs/expF.cl COPYONLY)

# erfcinv			OCML in chipStar (__ocml_erfcinv_f64)
configure_file(${OCML_SRC_DIR}/erfcinvD.cl ${CMAKE_BINARY_DIR}/host_math_funcs/erfcinvD.cl COPYONLY)

# # erfcinvf		OCML in chipStar (__ocml_erfcinv_f32)
configure_file(${OCML_SRC_DIR}/erfcinvF.cl ${CMAKE_BINARY_DIR}/host_math_funcs/erfcinvF.cl COPYONLY)

# # erfcx			OCML in chipStar (__ocml_erfcx_f64)
configure_file(${OCML_SRC_DIR}/erfcxD.cl ${CMAKE_BINARY_DIR}/host_math_funcs/erfcxD.cl COPYONLY)

# # erfcxf			OCML in chipStar (__ocml_erfcx_f32)
configure_file(${OCML_SRC_DIR}/erfcxF.cl ${CMAKE_BINARY_DIR}/host_math_funcs/erfcxF.cl COPYONLY)

# # erfinv			OCML in chipStar (__ocml_erfinv_f64)
configure_file(${OCML_SRC_DIR}/erfinvD.cl ${CMAKE_BINARY_DIR}/host_math_funcs/erfinvD.cl COPYONLY)

# # erfinvf			OCML in chipStar (__ocml_erfinv_f32)
configure_file(${OCML_SRC_DIR}/erfinvF.cl ${CMAKE_BINARY_DIR}/host_math_funcs/erfinvF.cl COPYONLY)

# # normcdf			OCML in chipStar (__ocml_ncdf_f64)
configure_file(${OCML_SRC_DIR}/ncdfD.cl ${CMAKE_BINARY_DIR}/host_math_funcs/ncdfD.cl COPYONLY)

# # normcdff		OCML in chipStar (__ocml_ncdf_f32)
configure_file(${OCML_SRC_DIR}/ncdfF.cl ${CMAKE_BINARY_DIR}/host_math_funcs/ncdfF.cl COPYONLY)

# # normcdfinv		OCML in chipStar (__ocml_ncdfinv_f64)
configure_file(${OCML_SRC_DIR}/ncdfinvD.cl ${CMAKE_BINARY_DIR}/host_math_funcs/ncdfinvD.cl COPYONLY)

# # normcdfinvf		OCML in chipStar (__ocml_ncdfinv_f32)
configure_file(${OCML_SRC_DIR}/ncdfinvF.cl ${CMAKE_BINARY_DIR}/host_math_funcs/ncdfinvF.cl COPYONLY)

# # rcbrt			OCML in chipStar (__ocml_rcbrt_f64)
configure_file(${OCML_SRC_DIR}/rcbrtD.cl ${CMAKE_BINARY_DIR}/host_math_funcs/rcbrtD.cl COPYONLY)

# # rcbrtf			OCML in chipStar (__ocml_rcbrt_f32)
configure_file(${OCML_SRC_DIR}/rcbrtF.cl ${CMAKE_BINARY_DIR}/host_math_funcs/rcbrtF.cl COPYONLY)

# # sincospi		OCML in chipStar (__ocml_sincospi_f64)
configure_file(${OCML_SRC_DIR}/sincospiD.cl ${CMAKE_BINARY_DIR}/host_math_funcs/sincospiD.cl COPYONLY)

# # sincospif		OCML in chipStar (__ocml_sincospi_f32)
configure_file(${OCML_SRC_DIR}/sincospiF.cl ${CMAKE_BINARY_DIR}/host_math_funcs/sincospiF.cl COPYONLY)

# # sinpi			OpenCL in chipStar (extern)
configure_file(${OCML_SRC_DIR}/sinpiD.cl ${CMAKE_BINARY_DIR}/host_math_funcs/sinpiD.cl COPYONLY)

# # sinpif			OpenCL in chipStar (call sinpi)
configure_file(${OCML_SRC_DIR}/sinpiF.cl ${CMAKE_BINARY_DIR}/host_math_funcs/sinpiF.cl COPYONLY)

# rsqrt			OCML in chipStar (__ocml_rsqrt_f64)
configure_file(${OCML_SRC_DIR}/rsqrtD.cl ${CMAKE_BINARY_DIR}/host_math_funcs/rsqrtD.cl COPYONLY)

# rsqrtf			OCML in chipStar (__ocml_rsqrt_f32)
configure_file(${OCML_SRC_DIR}/rsqrtF.cl ${CMAKE_BINARY_DIR}/host_math_funcs/rsqrtF.cl COPYONLY)

# llmax			calling max() in chipStar
# llmin			calling min() in chipStar
# ullmax			calling max() in chipStar
# ullmin			calling min() in chipStar
# umax			calling max() in chipStar
# umin			calling min() in chipStar

# signbit			OCML in chipStar (__ocml_signbit_f64)
configure_file(${OCML_SRC_DIR}/signbitD.cl ${CMAKE_BINARY_DIR}/host_math_funcs/signbitD.cl COPYONLY)
configure_file(${OCML_SRC_DIR}/signbitF.cl ${CMAKE_BINARY_DIR}/host_math_funcs/signbitF.cl COPYONLY)


# Gather all .cl sources from the OCML source directory
file(GLOB OCML_CL_SOURCES ${CMAKE_BINARY_DIR}/host_math_funcs/*.cl)

# Print the list of gathered .cl sources for debugging
message(STATUS "OCML CL Sources: ${OCML_CL_SOURCES}")

# Set the language for the target
set_source_files_properties(${OCML_CL_SOURCES} PROPERTIES LANGUAGE C)

# Add the library target
add_library(ocml_host_math_funcs STATIC ${OCML_CL_SOURCES})
set_target_properties(ocml_host_math_funcs PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
# Set the linker language explicitly
set_target_properties(ocml_host_math_funcs PROPERTIES LINKER_LANGUAGE C)

# Add ocml_host_math_funcs to the export set
install(TARGETS ocml_host_math_funcs EXPORT hip-targets)
install(TARGETS ocml_host_math_funcs EXPORT CHIPTargets)