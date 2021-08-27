cmake_minimum_required(VERSION 3.0.0)
project(HIPxx VERSION 0.1.0)

set(CMAKE_PREFIX_PATH /home/pvelesko/local/catch2)
find_package(Catch2 REQUIRED)
# include(FetchContent)
# FetchContent_Declare(
#   Catch2
#   GIT_REPOSITORY https://github.com/catchorg/Catch2.git
#   GIT_TAG        v3.0.0-preview3
# )

# FetchContent_MakeAvailable(Catch2)


set(CMAKE_CXX_COMPILER /soft/compilers/llvm/release-12.0.1/bin/clang++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

set(OpenCL_LIBRARY /soft/libraries/khronos/loader/master-2021.01.04/lib64/libOpenCL.so.1.2)
set(OpenCL_INCLUDE_DIR /soft/libraries/khronos/headers/master-2021.01.04/include)

include(CTest)
enable_testing()

add_compile_options(-Wno-format-extra-args)

include_directories(src backend include . ${OpenCL_INCLUDE_DIR})

find_package(Threads REQUIRED)
set(HIPXX_SRC src/HIPxxDriver.cc src/HIPxxBackend.cc src/logging.cc)
set(DRIVER_SRC src/main.cc)

add_library(HIPxx ${HIPXX_SRC})
target_link_libraries(HIPxx ${OpenCL_LIBRARY})
add_executable(driver ${DRIVER_SRC})


target_link_libraries(driver -stdlib=libstdc++ HIPxx Threads::Threads ${OpenCL_LIBARY})
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

add_executable(tests tests/TestHIPxxDriver.cc tests/TestHIPxxBackend.cc tests/TestHIPxxBackendOpenCL.cc)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain HIPxx)

#-DOpenCL_LIBRARY=/soft/libraries/khronos/loader/master-2021.01.04/lib64/libOpenCL.so.1.2 -DOpenCL_INCLUDE_DIR=/soft/libraries/khronos/headers/master-2021.01.04/include


