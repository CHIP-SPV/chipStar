cmake_minimum_required(VERSION 3.0.0)
project(HIPxx VERSION 0.1.0)

set(CMAKE_CXX_COMPILER /soft/compilers/llvm/release-12.0.1/bin/clang++)
set(CXX_STANDARD 17)


include(CTest)
enable_testing()

add_compile_options(-std=c++17)
add_compile_options(-Wno-format-extra-args)

include_directories(src backend include)

find_package(Threads REQUIRED)
set(HIPXX_SRC src/HIPxxDriver.cc src/HIPxxBackend.cc)
set(DRIVER_SRC src/main.cc)

add_library(HIPxx ${HIPXX_SRC})
add_executable(driver ${DRIVER_SRC})


target_link_libraries(driver -stdlib=libstdc++ HIPxx Threads::Threads)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)