cmake_minimum_required(VERSION 3.4.3 FATAL_ERROR)
cmake_policy(VERSION 3.4)

# For building on top of HIP-COMMON
add_compile_options(-D__HIP_PLATFORM_AMD__)

option(ABORT_IF_NOT_IMPLEMENTED OFF)
option(LAZY_JIT OFF)
option(CACHE_KERNELS OFF)

if (ABORT_IF_NOT_IMPLEMENTED)
  add_compile_options(-DCHIP_ABORT_ON_UNIMPL)
endif()

project(CHIP-SPV
        VERSION 0.1.0
        DESCRIPTION "HIP implementation over any backend that supports SPIR-V"
        LANGUAGES C CXX)
set(CMAKE_CXX_FLAGS "-Wno-tautological-constant-compare -Wno-return-type")
set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard to use.")
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "/opt/CHIP-SPV" CACHE PATH "Install path prefix" FORCE)
endif()
message(STATUS "CHIP-SPV will be installed to: ${CMAKE_INSTALL_PREFIX}")


 if(UNIX AND ((CMAKE_CXX_COMPILER_ID MATCHES "[Cc]lang") OR (CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")))
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0.0)
     message(FATAL_ERROR "this project requires clang >= 8.0")
   endif()
 else()
   message(FATAL_ERROR "this project must be compiled with clang. CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")
endif()

# Find OpenCL
if(NOT DEFINED OpenCL_LIBRARY_PATH)
  message(STATUS "-DOpenCL_LIBRARY_PATH was not used, checking LD_LIBRARY_PATH for OpenCL")
  find_library(OpenCL_LIBRARY 
    NAMES OpenCL 
    PATHS ENV LD_LIBRARY_PATH
    NOCACHE)
  if(${OpenCL_LIBRARY} STREQUAL OpenCL_LIBRARY-NOTFOUND)
    message(FATAL_ERROR "OpenCL library not found in LD_LIBRARY_PATH")
  else()
    message(STATUS "OpenCL Found: ${OpenCL_LIBRARY}")
  endif()
else()
  message(STATUS "-DOpenCL_LIBRARY_PATH was set. Looking for OpenCL.so")
  find_library(OpenCL_LIBRARY 
    NAMES OpenCL 
    PATHS ${OpenCL_LIBRARY_PATH}
    NOCACHE)
  if(${OpenCL_LIBRARY} STREQUAL OpenCL_LIBRARY-NOTFOUND)
    message(FATAL_ERROR "OpenCL library not found in OpenCL_LIBRARY_PATH")
  else()
    message(STATUS "OpenCL Found: ${OpenCL_LIBRARY}")
  endif()
endif()

# TODO : Check OpenCL version

# Setup Testing
find_package(Catch2 REQUIRED)
include(CTest)
enable_testing()

add_compile_options(-Wno-format-extra-args)

include_directories(src backend include . HIP/include ${OpenCL_INCLUDE_DIR})

find_package(Threads REQUIRED)
set(CHIP_SRC 
    src/spirv.cc 
    src/CHIPDriver.cc 
    src/CHIPBackend.cc 
    src/logging.cc
    src/CHIPBindings.cc
    src/backend/Level0/Level0Backend.cc
    src/backend/OpenCL/CHIPBackendOpenCL.cc
    src/backend/OpenCL/SVMemoryRegion.cc
)
# set(DRIVER_SRC src/main.cc)

# Level0 Runtime
macro(find_library_dynamic libname)
  message(STATUS "\nSearching for ${libname}")
  find_library(${libname}_LIBRARY
    NAMES ${libname}
    PATHS ENV LD_LIBRARY_PATH
  )
  if(${${libname}_LIBRARY} STREQUAL ${libname}_LIBRARY-NOTFOUND)
    message(FATAL_ERROR "${libname} library not found in LD_LIBRARY_PATH")
  else()
    message(STATUS "Found: ${${libname}_LIBRARY}\n")
    list(APPEND Level0_LIBRARIES ${${libname}_LIBRARY})
  endif()
endmacro()

find_library_dynamic(libpi_level_zero.so)
find_library_dynamic(libze_loader.so.1)
find_library_dynamic(libsvml.so)
find_library_dynamic(libirng.so)
find_library_dynamic(libimf.so)
find_library_dynamic(libintlc.so.5)

message(STATUS "Level0_LIBRARIES=${Level0_LIBRARIES}")

add_library(CHIP SHARED ${CHIP_SRC})
target_link_libraries(CHIP ${OpenCL_LIBRARY} ${Level0_LIBRARIES})
# add_executable(driver ${DRIVER_SRC})
#target_link_libraries(driver -stdlib=libstdc++ CHIP Threads::Threads ${OpenCL_LIBARY} ${Level0_LIBRARIES})
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

add_executable(tests tests/TestCHIPDriver.cc tests/TestCHIPBackend.cc tests/TestCHIPBackendOpenCL.cc)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain CHIP ${Level0_LIBRARIES})
# Add to LD_LIBRARY_PATH
#clang++ -pthread -fPIE -O2 -g -std=c++11 `hipcl_config -C` -o binary MatrixMultiply.cpp  -Wl,-rpath,${hipcl_install_prefix}/lib -L${hipcl_install_prefix}/lib -L/home/pvelesko/CHIP/build -lCHIP