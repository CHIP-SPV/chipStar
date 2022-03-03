# String Options
macro(option_str OPTION_NAME DOC_STRING)
  set(${OPTION_NAME} "" CACHE PATH "${DOC_STRING}")
  if(${OPTION_NAME} STREQUAL "") 
    message(FATAL_ERROR "${OPTION_NAME} must be set")
  endif()
  message(STATUS "${OPTION_NAME}: ${${OPTION_NAME}}")
endmacro()

# Find library in LD_LIBRARY_PATH
macro(find_library_dynamic libname)
  message(STATUS "\nSearching for ${libname}")
  find_library(${libname}_LIBRARY
    NAMES ${libname}
    PATHS ENV LD_LIBRARY_PATH ./
  )
  if(${${libname}_LIBRARY} STREQUAL ${libname}_LIBRARY-NOTFOUND)
    message(FATAL_ERROR "${libname} library not found in LD_LIBRARY_PATH")
  else()
    message(STATUS "Found: ${${libname}_LIBRARY}\n")
    list(APPEND CHIP-SPV_LIBRARIES ${${libname}_LIBRARY})
  endif()
endmacro()

# Append '<key>=<value>' line into a variable.
macro(add_to_config _configvar _key _value)
  set(${_configvar} "${${_configvar}}${_key}=${_value}\n")
endmacro()
