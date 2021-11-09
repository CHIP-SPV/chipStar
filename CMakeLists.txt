cmake_minimum_required(VERSION 3.4.3 FATAL_ERROR)
cmake_policy(VERSION 3.4)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ABORT_IF_NOT_IMPLEMENTED OFF)
option(LAZY_JIT OFF)
option(CACHE_KERNELS OFF)
option(VERBOSE "Verbose compilation" OFF)

if(VERBOSE)
  set(CMAKE_VERBOSE_MAKEFILE ON)
  add_compile_options("-v")
endif()

if (ABORT_IF_NOT_IMPLEMENTED)
  add_definitions(-DCHIP_ABORT_ON_UNIMPL)
endif()

#add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_ERROR)
#define SPDLOG_LEVEL_TRACE 0
#define SPDLOG_LEVEL_DEBUG 1
#define SPDLOG_LEVEL_INFO 2
#define SPDLOG_LEVEL_WARN 3
#define SPDLOG_LEVEL_ERROR 4
#define SPDLOG_LEVEL_CRITICAL 5
#define SPDLOG_LEVEL_OFF 6

project(CHIP-SPV
        VERSION 0.1.0
        DESCRIPTION "HIP implementation over any backend that supports SPIR-V"
        LANGUAGES C CXX)
set(CMAKE_CXX_FLAGS "-Wno-tautological-constant-compare -Wno-return-type -Wno-c++20-extensions -Wno-unused-result")
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
    src/spirv_hiprtc.cc 
    src/CHIPDriver.cc 
    src/CHIPBackend.cc 
    src/CHIPBindings.cc
    src/backend/Level0/Level0Backend.cc
    src/backend/OpenCL/CHIPBackendOpenCL.cc
    src/backend/OpenCL/SVMemoryRegion.cc
    src/logging.cc
)
set(DRIVER_SRC src/main.cc)

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

add_library(CHIP STATIC ${CHIP_SRC})

message("\n\nClang Version: ${CMAKE_CXX_COMPILER_VERSION}")
if (${CMAKE_CXX_COMPILER_VERSION} EQUAL 8.0.1)
  set(HIPLZ_DIR "/soft/compilers/hipcl/20210506")
else()
  set(HIPLZ_DIR "/home/pvelesko/HIPLZ")
endif()

message("Using HIPLZ_DIR=${HIPLZ_DIR}" )
set(HIP_PATH HIP)
set(OFFLOAD_ARCH_STR -D__HIP_PLATFORM_SPIRV__= -x hip --target=x86_64-linux-gnu --hip-device-lib-path=${HIPLZ_DIR}/share --hip-device-lib=kernellib.bc --hip-llvm-pass-path=${HIPLZ_DIR}/lib/llvm  )

# For building on top of HIP-COMMON
target_compile_definitions(CHIP PUBLIC -D__HIP_PLATFORM_SPIRV__ -D__HIP_PLATFORM_CLANG__ )
target_compile_options(CHIP PUBLIC ${OFFLOAD_ARCH_STR})

target_link_libraries(CHIP ${OpenCL_LIBRARY} ${Level0_LIBRARIES})
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)


enable_testing()
add_subdirectory(HIP/tests/catch catch)

add_executable(tests tests/TestCHIPDriver.cc tests/TestCHIPBackend.cc tests/TestCHIPBackendOpenCL.cc)
target_compile_definitions(tests PRIVATE -D__HIP_PLATFORM_SPIRV__ -D__HIP_PLATFORM_CLANG__)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain CHIP ${Level0_LIBRARIES})

add_executable(driver ${DRIVER_SRC})
target_compile_options(driver PRIVATE ${OFFLOAD_ARCH_STR})
target_include_directories(driver PRIVATE HIP/include)
target_link_libraries(driver -stdlib=libstdc++ CHIP Threads::Threads ${OpenCL_LIBARY} ${Level0_LIBRARIES})

add_subdirectory(./samples samples)



install(TARGETS CHIP
        EXPORT CHIPTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES ${PROJECT_BINARY_DIR}/libCHIP.so DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/HIP/include DESTINATION . USE_SOURCE_PERMISSIONS)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION . USE_SOURCE_PERMISSIONS)

install(EXPORT CHIPTargets
        FILE CHIPTargets.cmake
        NAMESPACE CHIP::
        DESTINATION  cmake/CHIP)


# Add to LD_LIBRARY_PATH
#pvelesko@jlselogin4:~/CHIP-SPV/samples/0_MatrixMultiply> /soft/compilers/clang-hipcl/13.0-experimental/bin/clang++  -pthread -fPIE -O2 -g -std=c++17  
#-I/home/pvelesko/CHIP-SPV/HIP/include  -I/home/pvelesko/CHIP-SPV/include -D__HIP_PLATFORM_SPIRV__=  
#-x hip --hip-device-lib-path=/soft/compilers/hipcl/20210907-experimental13-debug/share  --hip-device-lib=kernellib.bc 
#--hip-llvm-pass-path=/soft/compilers/hipcl/20210907-experimental13-debug/lib/llvm  -o binary MatrixMultiply.cpp  
#-Wl,-rpath,${hipcl_install_prefix}/lib -L${hipcl_install_prefix}/lib -L/home/pvelesko/CHIP-SPV/build -lCHIP -D__HIP_PLATFORM_CLANG__