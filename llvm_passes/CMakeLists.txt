set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ standard to use.")
set(CMAKE_CXX_EXTENSIONS OFF)

execute_process(COMMAND "${LLVM_CONFIG}" "--cmakedir"
		OUTPUT_VARIABLE LLVM_DIR
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE RES)

if(NOT RES EQUAL 0)
  message(FATAL_ERROR "failed to run llvm-config (${LLVM_CONFIG})")
endif()

message(STATUS "LLVM CMake directory: ${LLVM_DIR}")

find_package(LLVM REQUIRED CONFIG HINTS "${LLVM_DIR}" PATHS "${LLVM_DIR}" NO_DEFAULT_PATH)

######################################

add_definitions(${LLVM_DEFINITIONS})

include_directories(${LLVM_INCLUDE_DIRS})

if(NOT LLVM_ENABLE_RTTI)
  add_compile_options("-fno-rtti")
endif()

if(NOT LLVM_ENABLE_EH)
  add_compile_options("-fno-exceptions")
endif()

add_library(LLVMHipDynMem MODULE HipDynMem.cpp)
add_library(LLVMHipStripCompilerUsed MODULE HipStripCompilerUsed.cpp)
add_library(LLVMHipTexture MODULE HipTexture.cpp)
install(TARGETS LLVMHipDynMem LLVMHipStripCompilerUsed LLVMHipTexture
        LIBRARY DESTINATION lib/llvm
        ARCHIVE DESTINATION lib/llvm
        )

if(DEFINED LLVM_VERSION_MAJOR AND "${LLVM_VERSION_MAJOR}" VERSION_GREATER 9)
  add_library(LLVMHipDefrost MODULE HipDefrost.cpp)
  install(TARGETS LLVMHipDefrost
          LIBRARY DESTINATION lib/llvm
          ARCHIVE DESTINATION lib/llvm
          )
endif()

if(DEFINED LLVM_VERSION_MAJOR AND "${LLVM_VERSION_MAJOR}" VERSION_GREATER 10)
  add_library(LLVMHipPasses MODULE HipPasses.cpp
    HipDynMem.cpp HipStripCompilerUsed.cpp HipTexture.cpp HipDefrost.cpp HipPrintf.cpp HipGlobalVariables.cpp)

  install(TARGETS LLVMHipPasses
          LIBRARY DESTINATION lib/llvm
          ARCHIVE DESTINATION lib/llvm
          )
endif()
