# temporary
set(CMAKE_WARN_DEPRECATED OFF)
add_compile_options(-Wno-format-extra-args)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "-Wno-tautological-constant-compare  -Wno-c++20-extensions -Wno-unused-result ")
# end temporary

cmake_minimum_required(VERSION 3.4.3 FATAL_ERROR)
cmake_policy(VERSION 3.4)

project(CHIP-SPV
        VERSION 0.1.0
        DESCRIPTION "HIP implementation over any backend that supports SPIR-V"
        LANGUAGES C CXX)
include(cmake/chip_spv_cmake_macros.cmake)

# Cmake options that can be passed in by the user
option(ABORT_IF_NOT_IMPLEMENTED "Some HIP functionality is not yet implemented in CHIP-SPV. If such a function is called, abort execution via std::abort()" OFF)
option(LAZY_JIT "JIT kernels at the time of the first kernel invocation instead of at the start of the program execution" OFF)
option(CACHE_KERNELS "Save the compiled kernel to disk to speed up program startup" OFF)
option(VERBOSE "Verbose compilation" OFF)
option(BUILD_SHARED_LIBS "Build CHIP-SPV as a shared library" OFF)
#option_str(HIP_SPIRV_DIR "" CACHE PATH "Path to HIP SPIRV directory containing HIP_SPIRV_DIR/share/kernellib.bc and HIP_SPIRV_DIR/lib/llvm")

# Find kernellib.bc and HIP SPIRV LLVM passes
#if("${HIP_SPIRV_DIR}" STREQUAL "")
#  message(FATAL_ERROR "You must set pass -DHIP_SPIRV_DIR=... ")
#endif()
#message("\n\nUsing HIP_SPIRV_DIR=${HIP_SPIRV_DIR}\n\n")

set(HIP_PLATFORM spirv)
set(HIP_COMPILER clang)

if(VERBOSE)
  set(CMAKE_VERBOSE_MAKEFILE ON)
  add_compile_options("-v")
endif()

if (ABORT_IF_NOT_IMPLEMENTED)
  add_definitions(-DCHIP_ABORT_ON_UNIMPL)
endif()

add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard to use.")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "/opt/CHIP-SPV" CACHE PATH "Install path prefix" FORCE)
endif()
message(STATUS "CHIP-SPV will be installed to: ${CMAKE_INSTALL_PREFIX}")

# Compiler CHECK
 if((CMAKE_CXX_COMPILER_ID MATCHES "[Cc]lang") OR (CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM"))
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0.0)
     message(FATAL_ERROR "this project requires clang >= 8.0")
   endif()
 else()
   message(FATAL_ERROR "this project must be compiled with clang. CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")
endif()

if (NOT DEFINED OpenCL_LIBRARY)
  message(STATUS "OpenCL_LIBRARY was not set. Searching for libOpenCL.so in LD_LIBRARY_PATH")
  find_library_dynamic(OpenCL)
endif()

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

find_library_dynamic(libze_loader.so.1)
message(STATUS "\n\nCHIP-SPV_LIBRARIES=${CHIP-SPV_LIBRARIES}\n\n")

# TODO small : read cmake configuration
if (BUILD_SHARED_LIBS)
  message(STATUS "Buiding CHIP-SPV as a shared library")
  add_library(CHIP SHARED ${CHIP_SRC})
  set(CHIP_LIB_NAME "libCHIP.so")
else()
  message(STATUS "Buiding CHIP-SPV as a shared library")
  add_library(CHIP STATIC ${CHIP_SRC})
  set(CHIP_LIB_NAME "libCHIP.a")
endif()

set(HIP_PATH HIP)
set(OFFLOAD_ARCH_STR -D__HIP_PLATFORM_SPIRV__= -x hip --target=x86_64-linux-gnu --hip-device-lib-path=${HIP_SPIRV_DIR}/share --hip-device-lib=kernellib.bc --hip-llvm-pass-path=${HIP_SPIRV_DIR}/lib/llvm  )

# For building on top of HIP-COMMON
target_compile_options(CHIP PUBLIC ${OFFLOAD_ARCH_STR})

target_link_libraries(CHIP ${OpenCL_LIBRARY} ${CHIP-SPV_LIBRARIES})
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)


add_subdirectory(HIP/tests/catch catch)

add_executable(driver ${DRIVER_SRC})
target_compile_options(driver PRIVATE ${OFFLOAD_ARCH_STR})
target_include_directories(driver PRIVATE HIP/include)
target_link_libraries(driver -stdlib=libstdc++ CHIP Threads::Threads ${OpenCL_LIBARY} ${CHIP-SPV_LIBRARIES})

add_subdirectory(./samples samples)

include(cmake/FindLLVM.cmake)
add_subdirectory(llvm_passes)
add_subdirectory(bitcode)

install(TARGETS CHIP
        EXPORT CHIPTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES ${PROJECT_BINARY_DIR}/${CHIP_LIB_NAME} DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/HIP/include DESTINATION . USE_SOURCE_PERMISSIONS)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION . USE_SOURCE_PERMISSIONS)

install(EXPORT CHIPTargets
        FILE CHIPTargets.cmake
        NAMESPACE CHIP::
        DESTINATION  cmake/CHIP)