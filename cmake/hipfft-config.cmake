# hipfft-config.cmake
# CMake find-package config for hipFFT installed by chipStar
#
# Provides the imported target:
#   hip::hipfft  - links libhipfft.so and sets include path
#
# Works with:
#   find_package(hipfft CONFIG)
#   target_link_libraries(myapp hip::hipfft)

cmake_minimum_required(VERSION 3.5)

if(TARGET hip::hipfft)
  return()
endif()

get_filename_component(_HIPFFT_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}" DIRECTORY)  # lib/cmake
get_filename_component(_HIPFFT_CMAKE_DIR "${_HIPFFT_CMAKE_DIR}" DIRECTORY)       # lib
get_filename_component(_HIPFFT_PREFIX "${_HIPFFT_CMAKE_DIR}" DIRECTORY)           # <prefix>

find_library(_HIPFFT_LIBRARY
  NAMES hipfft
  HINTS "${_HIPFFT_PREFIX}/lib"
  NO_DEFAULT_PATH
)

find_path(_HIPFFT_INCLUDE_DIR
  NAMES hipfft/hipfft.h hipfft.h
  HINTS "${_HIPFFT_PREFIX}/include"
  NO_DEFAULT_PATH
)

if(NOT _HIPFFT_LIBRARY)
  set(hipfft_FOUND FALSE)
  if(hipfft_FIND_REQUIRED)
    message(FATAL_ERROR "hipfft: libhipfft.so not found under ${_HIPFFT_PREFIX}/lib. "
      "Install H4I-HipFFT or chipFFT via install_chipstar.py first.")
  endif()
  return()
endif()

add_library(hip::hipfft SHARED IMPORTED)
set_target_properties(hip::hipfft PROPERTIES
  IMPORTED_LOCATION "${_HIPFFT_LIBRARY}"
)
if(_HIPFFT_INCLUDE_DIR)
  set_target_properties(hip::hipfft PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_HIPFFT_INCLUDE_DIR}"
  )
endif()

set(hipfft_FOUND TRUE)
set(hipfft_LIBRARIES hip::hipfft)
set(hipfft_INCLUDE_DIRS "${_HIPFFT_INCLUDE_DIR}")

unset(_HIPFFT_PREFIX)
unset(_HIPFFT_CMAKE_DIR)
