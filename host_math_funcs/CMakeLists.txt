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
file(COPY ${OCML_SRC_DIR}/trigredF.h DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)
file(COPY ${OCML_SRC_DIR}/sincospiredF.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)
file(COPY ${OCML_SRC_DIR}/tanpiredF.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)
file(COPY ${OCML_SRC_DIR}/trigpiredF.h DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)
file(COPY ${OCML_SRC_DIR}/trigpiredF.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)
file(COPY ${OCML_SRC_DIR}/cospiF.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)

file(COPY ${OCML_SRC_DIR}/trigpiredD.h DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)
file(COPY ${OCML_SRC_DIR}/sincospiredD.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)
file(COPY ${OCML_SRC_DIR}/tanpiredD.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)
file(COPY ${OCML_SRC_DIR}/trigpiredD.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)
file(COPY ${OCML_SRC_DIR}/cospiD.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)

file(COPY ${OCML_SRC_DIR}/erfcD.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)
file(COPY ${OCML_SRC_DIR}/erfcF.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)

file(COPY ${OCML_SRC_DIR}/expD_base.h DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)
file(COPY ${OCML_SRC_DIR}/expF_base.h DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)
file(COPY ${OCML_SRC_DIR}/logD_base.h DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)
file(COPY ${OCML_SRC_DIR}/logF_base.h DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)
file(COPY ${OCML_SRC_DIR}/logD.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)
file(COPY ${OCML_SRC_DIR}/logF.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)


file(COPY ${OCML_SRC_DIR}/expD.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)
file(COPY ${OCML_SRC_DIR}/expF.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)

# erfcinv			OCML in chipStar (__ocml_erfcinv_f64)
file(COPY ${OCML_SRC_DIR}/erfcinvD.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)

# # erfcinvf		OCML in chipStar (__ocml_erfcinv_f32)
file(COPY ${OCML_SRC_DIR}/erfcinvF.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)

# # erfcx			OCML in chipStar (__ocml_erfcx_f64)
file(COPY ${OCML_SRC_DIR}/erfcxD.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)

# # erfcxf			OCML in chipStar (__ocml_erfcx_f32)
file(COPY ${OCML_SRC_DIR}/erfcxF.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)

# # erfinv			OCML in chipStar (__ocml_erfinv_f64)
file(COPY ${OCML_SRC_DIR}/erfinvD.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)

# # erfinvf			OCML in chipStar (__ocml_erfinv_f32)
file(COPY ${OCML_SRC_DIR}/erfinvF.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)

# # normcdf			OCML in chipStar (__ocml_ncdf_f64)
file(COPY ${OCML_SRC_DIR}/ncdfD.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)

# # normcdff		OCML in chipStar (__ocml_ncdf_f32)
file(COPY ${OCML_SRC_DIR}/ncdfF.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)

# # normcdfinv		OCML in chipStar (__ocml_ncdfinv_f64)
file(COPY ${OCML_SRC_DIR}/ncdfinvD.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)

# # normcdfinvf		OCML in chipStar (__ocml_ncdfinv_f32)
file(COPY ${OCML_SRC_DIR}/ncdfinvF.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)

# # rcbrt			OCML in chipStar (__ocml_rcbrt_f64)
file(COPY ${OCML_SRC_DIR}/rcbrtD.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)

# # rcbrtf			OCML in chipStar (__ocml_rcbrt_f32)
file(COPY ${OCML_SRC_DIR}/rcbrtF.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)

# # sincospi		OCML in chipStar (__ocml_sincospi_f64)
file(COPY ${OCML_SRC_DIR}/sincospiD.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)

# # sincospif		OCML in chipStar (__ocml_sincospi_f32)
file(COPY ${OCML_SRC_DIR}/sincospiF.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)

# # sinpi			OpenCL in chipStar (extern)
file(COPY ${OCML_SRC_DIR}/sinpiD.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)

# # sinpif			OpenCL in chipStar (call sinpi)
file(COPY ${OCML_SRC_DIR}/sinpiF.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)

# llmax			calling max() in chipStar
# llmin			calling min() in chipStar
# ullmax			calling max() in chipStar
# ullmin			calling min() in chipStar
# umax			calling max() in chipStar
# umin			calling min() in chipStar

# signbit			OCML in chipStar (__ocml_signbit_f64)
file(COPY ${OCML_SRC_DIR}/signbitD.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)
file(COPY ${OCML_SRC_DIR}/signbitF.cl DESTINATION ${CMAKE_BINARY_DIR}/host_math_funcs)


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