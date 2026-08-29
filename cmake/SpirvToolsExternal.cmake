# Builds SPIRV-Tools from source as an ExternalProject when find_package did
# not find an installed copy, and exposes it as the imported target
# SPIRV-Tools rooted at ${CMAKE_BINARY_DIR}/external/spirv-tools.

# Download and build SPIRV-Tools
include(ExternalProject)

set(SPIRV_TOOLS_VERSION "main")
set(SPIRV_TOOLS_INSTALL_DIR "${CMAKE_BINARY_DIR}/external/spirv-tools")

# Create the include directory before it's referenced
file(MAKE_DIRECTORY "${SPIRV_TOOLS_INSTALL_DIR}/include")
file(MAKE_DIRECTORY "${SPIRV_TOOLS_INSTALL_DIR}/lib")

set(CHIPSTAR_C_COMPILER gcc)
set(CHIPSTAR_CXX_COMPILER g++)
message(STATUS "CHIPSTAR_C_COMPILER: ${CHIPSTAR_C_COMPILER}")
message(STATUS "CHIPSTAR_CXX_COMPILER: ${CHIPSTAR_CXX_COMPILER}")
ExternalProject_Add(SPIRV-Tools-External
  GIT_REPOSITORY   https://github.com/CHIP-SPV/SPIRV-Tools.git
  GIT_TAG          ${SPIRV_TOOLS_VERSION}
  GIT_SHALLOW      TRUE
  SOURCE_SUBDIR    "."
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${SPIRV_TOOLS_INSTALL_DIR}
    -DSPIRV_SKIP_TESTS=ON
    -DCMAKE_C_COMPILER=${CHIPSTAR_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CHIPSTAR_CXX_COMPILER}
    -DCMAKE_INSTALL_LIBDIR=lib
    -DSPIRV_TOOLS_INSTALL_HEADERS=ON
    -DSPIRV_TOOLS_BUILD_STATIC=ON
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  UPDATE_COMMAND   ${CMAKE_COMMAND} -E chdir <SOURCE_DIR> python3 utils/git-sync-deps
  LOG_DOWNLOAD ON
  LOG_UPDATE ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
  BUILD_BYPRODUCTS 
    ${SPIRV_TOOLS_INSTALL_DIR}/lib/libSPIRV-Tools.a
    ${SPIRV_TOOLS_INSTALL_DIR}/lib/libSPIRV-Tools-opt.a
    ${SPIRV_TOOLS_INSTALL_DIR}/lib/libSPIRV-Tools-link.a
    ${SPIRV_TOOLS_INSTALL_DIR}/lib/libSPIRV-Tools-reduce.a
)

# Create an imported target for SPIRV-Tools
add_library(SPIRV-Tools STATIC IMPORTED GLOBAL)
add_dependencies(SPIRV-Tools SPIRV-Tools-External)

set_target_properties(SPIRV-Tools PROPERTIES
  IMPORTED_LOCATION "${SPIRV_TOOLS_INSTALL_DIR}/lib/libSPIRV-Tools.a"
  INTERFACE_INCLUDE_DIRECTORIES "${SPIRV_TOOLS_INSTALL_DIR}/include"
)
