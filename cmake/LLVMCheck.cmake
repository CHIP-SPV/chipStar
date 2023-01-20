if((CMAKE_CXX_COMPILER_ID MATCHES "[Cc]lang") OR
   (CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM"))
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0.0)
    message(FATAL_ERROR "this project requires clang >= 8.0")
  endif()

  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 16.0.0)
    set(CLANG_VERSION_LESS_16 ON)
  endif()

  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 15.0.0)
    set(CLANG_VERSION_LESS_15 ON)
  endif()

  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14.0.0)
    message(WARNING "Deprecated clang version '${CMAKE_CXX_COMPILER_VERSION}'. \
            Support for Clang < 14.0 will be discontinued in the future.")
    set(CLANG_VERSION_LESS_14 ON)
  endif()

  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0.0)
    set(CLANG_VERSION_LESS_13 ON)
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