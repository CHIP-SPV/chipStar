cmake_minimum_required(VERSION 3.0.0)
project(HIPxx VERSION 0.1.0)

set(CMAKE_CXX_COMPILER clang++)

include(CTest)
enable_testing()

include_directories(./)

set(HIPXX_SRC HIPxxDriver.cc)
set(DRIVER_SRC main.cc)

add_library(HIPxx ${HIPXX_SRC})
add_executable(driver ${DRIVER_SRC})
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)