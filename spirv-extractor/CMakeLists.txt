add_executable(spirvExtractor extractSPIRV.cc)
target_include_directories(spirvExtractor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/SPIRV-Tools/include)
target_link_libraries(spirvExtractor SPIRV-Tools)
# Make sure it gets built at ${CMAKE_BINARY_DIR}/bin
set_target_properties(spirvExtractor PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
install(FILES ${CMAKE_BINARY_DIR}/bin/spirvExtractor DESTINATION ${BIN_INSTALL_DIR} PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ)
