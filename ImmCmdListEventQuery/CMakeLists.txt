cmake_minimum_required(VERSION 3.0.0)
project(Reproducer VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE Debug)
set(OFFLOAD_TARGETS "icllp") # ocloc compile --help to get list of supported targets

find_library(Level0_LIBRARY ze_loader REQUIRED PATHS ENV LD_LIBRARY_PATH)

add_executable(driver main.cpp)
target_link_libraries(driver ${Level0_LIBRARY})

add_custom_command( OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/SlowKernel.spv"
                    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/SlowKernel.cl"
        COMMAND ocloc compile 
        -file "${CMAKE_CURRENT_SOURCE_DIR}/SlowKernel.cl"
        -device ${OFFLOAD_TARGETS}
        -output_no_suffix
        COMMENT "Building SlowKernel.spv"
        VERBATIM)

add_custom_target(Kernel DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/SlowKernel.spv")
add_dependencies(driver Kernel)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
