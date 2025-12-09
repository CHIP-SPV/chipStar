# hip-lang-config.cmake
# Redirect to hip package for chipStar
if(NOT TARGET hip::device AND NOT TARGET hip::host)
  find_package(hip CONFIG REQUIRED NO_DEFAULT_PATH PATHS "${CMAKE_CURRENT_LIST_DIR}/../hip")
endif()
