cmake_minimum_required(VERSION 3.4.3 FATAL_ERROR)
cmake_policy(VERSION 3.4)

project(HIPxx
        VERSION 0.1.0
        DESCRIPTION "HIP implementation over any backend that supports SPIR-V"
        LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ standard to use.")
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "/opt/hipxx" CACHE PATH "Install path prefix" FORCE)
endif()
message(STATUS "HIPxx will be installed to: ${CMAKE_INSTALL_PREFIX}")


 if(UNIX AND (CMAKE_CXX_COMPILER_ID MATCHES "[Cc]lang"))
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0.0)
     message(FATAL_ERROR "this project requires clang >= 8.0")
   endif()
 else()
   message(FATAL_ERROR "this project must be compiled with clang. CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")
endif()

find_package(OpenCL 1.2 REQUIRED)
#set(OpenCL_LIBRARY /soft/libraries/khronos/loader/master-2021.01.04/lib64/libOpenCL.so.1.2)
#set(OpenCL_INCLUDE_DIR /soft/libraries/khronos/headers/master-2021.01.04/include)

# Setup Testing
find_package(Catch2 REQUIRED)
include(CTest)
enable_testing()

add_compile_options(-Wno-format-extra-args)

include_directories(src backend include . ${OpenCL_INCLUDE_DIR})

find_package(Threads REQUIRED)
set(HIPXX_SRC 
    src/HIPxxDriver.cc 
    src/HIPxxBackend.cc 
    src/logging.cc
    src/HIPxxBindings.cc
)
set(DRIVER_SRC src/main.cc)

add_library(HIPxx SHARED ${HIPXX_SRC})
target_link_libraries(HIPxx ${OpenCL_LIBRARY})
add_executable(driver ${DRIVER_SRC})


target_link_libraries(driver -stdlib=libstdc++ HIPxx Threads::Threads ${OpenCL_LIBARY})
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

add_executable(tests tests/TestHIPxxDriver.cc tests/TestHIPxxBackend.cc tests/TestHIPxxBackendOpenCL.cc)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain HIPxx)