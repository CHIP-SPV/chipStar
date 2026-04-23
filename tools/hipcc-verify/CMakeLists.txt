add_executable(chip-kernel-verify chip-kernel-verify.cc)
target_include_directories(chip-kernel-verify PRIVATE
  ${SPIRV-Tools_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../spirv-extractor
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_link_libraries(chip-kernel-verify SPIRV-Tools)
if(HIPCC_VERIFY_DEVICES)
  target_compile_definitions(chip-kernel-verify PRIVATE
    CHIP_KERNEL_VERIFY_DEFAULT_DEVICES="${HIPCC_VERIFY_DEVICES}")
endif()
set_target_properties(chip-kernel-verify PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
install(FILES ${CMAKE_BINARY_DIR}/bin/chip-kernel-verify
  DESTINATION ${BIN_INSTALL_DIR}
  PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ)
