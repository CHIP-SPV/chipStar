add_executable(spirv-extractor spirv-extractor.cc)
target_include_directories(spirv-extractor PRIVATE /opt/homebrew/include ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_link_libraries(spirv-extractor /opt/homebrew/lib/libSPIRV-Tools-shared.dylib)
# Make sure it gets built at ${CMAKE_BINARY_DIR}/bin
set_target_properties(spirv-extractor PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
install(FILES ${CMAKE_BINARY_DIR}/bin/spirv-extractor DESTINATION ${BIN_INSTALL_DIR} PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ)

