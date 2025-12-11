# hip-lang-config.cmake
# chipStar's hip-lang package - provides compatibility with ROCm's hip-lang interface
# This package is loaded by CMake's HIP language support during language enablement

# chipStar does not provide a separate device runtime library
# The device runtime is handled via the hip package targets (hip::device, hip::host)
set(_CMAKE_HIP_DEVICE_RUNTIME_TARGET "" CACHE INTERNAL "HIP device runtime target for chipStar")

# Set hip_DIR so users can find_package(hip) after project()
# This makes chipStar work the same way as ROCm without requiring CMAKE_PREFIX_PATH
get_filename_component(_hip_lang_dir "${CMAKE_CURRENT_LIST_DIR}" DIRECTORY)
get_filename_component(_chipstar_root "${_hip_lang_dir}" DIRECTORY)
set(hip_DIR "${_chipstar_root}/cmake/hip" CACHE PATH "Path to hip package config file")
unset(_hip_lang_dir)
unset(_chipstar_root)

# Mark package as found
set(hip-lang_FOUND TRUE)

