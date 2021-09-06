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

include_directories(src backend include . ${OpenCL_INCLUDE_DIR})

find_package(Threads REQUIRED)
set(HIPXX_SRC 
    src/HIPxxDriver.cc 
    src/HIPxxBackend.cc 
    src/logging.cc
    src/HIPxxBindings.cc
    src/backend/OpenCL/HIPxxQueueOpenCL.cc
    src/backend/OpenCL/HIPxxContextOpenCL.cc
    src/backend/OpenCL/HIPxxDeviceOpenCL.cc
    src/backend/OpenCL/HIPxxKernelOpenCL.cc
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

add_library(HIPxx SHARED ${HIPXX_SRC})
target_link_libraries(HIPxx ${OpenCL_LIBRARY} ${Level0_LIBRARIES})
add_executable(driver ${DRIVER_SRC})

target_link_libraries(driver -stdlib=libstdc++ HIPxx Threads::Threads ${OpenCL_LIBARY} ${Level0_LIBRARIES})
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

add_executable(tests tests/TestHIPxxDriver.cc tests/TestHIPxxBackend.cc tests/TestHIPxxBackendOpenCL.cc)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain HIPxx ${Level0_LIBRARIES})
# Add to LD_LIBRARY_PATH
#clang++ -pthread -fPIE -O2 -g -std=c++11 `hipcl_config -C` -o binary MatrixMultiply.cpp  -Wl,-rpath,${hipcl_install_prefix}/lib -L${hipcl_install_prefix}/lib -L/home/pvelesko/HIPxx/build -lHIPxx