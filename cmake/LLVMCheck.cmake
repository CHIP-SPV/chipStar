if((CMAKE_CXX_COMPILER_ID MATCHES "[Cc]lang") OR
   (CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM"))

  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 20.0.0)
    message(FATAL_ERROR
      "Unsupported clang version '${CMAKE_CXX_COMPILER_VERSION}'. "
      "chipStar requires clang/LLVM 20, 21, or 22 (or the experimental "
      "'latest' toolchain); see scripts/configure_llvm.sh.")
  endif()
else()
  message(FATAL_ERROR "this project must be compiled with clang. CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")
endif()

string(REPLACE "." ";" VERSION_LIST ${CMAKE_CXX_COMPILER_VERSION})
list(GET VERSION_LIST 0 CLANG_VERSION_MAJOR)
get_filename_component(CLANG_BIN_NAME "${CMAKE_CXX_COMPILER}" NAME)
if(CLANG_BIN_NAME MATCHES "clang[+][+](-${CLANG_VERSION_MAJOR})")
  set(BINARY_VERSION_SUFFIX "-${CLANG_VERSION_MAJOR}")
else()
  set(BINARY_VERSION_SUFFIX)
endif()
