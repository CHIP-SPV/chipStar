find_package(SPIRV-Tools)
if (NOT SPIRV-Tools_FOUND)
  message(WARNING "SPIRV-Tools not found. spirv-extractor will not be installed.")
  return()
endif()

add_executable(spirv-extractor spirv-extractor.cc)
target_include_directories(spirv-extractor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/SPIRV-Tools/include)
target_link_libraries(spirv-extractor SPIRV-Tools)
# Make sure it gets built at ${CMAKE_BINARY_DIR}/bin
set_target_properties(spirv-extractor PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
install(FILES ${CMAKE_BINARY_DIR}/bin/spirv-extractor DESTINATION ${BIN_INSTALL_DIR} PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ)
