# RecursiveGaussian

add_hiplz_test(RecursiveGaussian RecursiveGaussian Passed RecursiveGaussian.cpp
    -q -e -t -i 32)

set(DEST_IMAGE "${CMAKE_CURRENT_BINARY_DIR}/RecursiveGaussian_Input.bmp")

add_custom_command(OUTPUT "${DEST_IMAGE}"
  COMMAND "${CMAKE_COMMAND}" -E copy
    "${CMAKE_CURRENT_SOURCE_DIR}/RecursiveGaussian_Input.bmp"
    "${CMAKE_CURRENT_BINARY_DIR}"
  VERBATIM)

add_custom_target("RecursiveGaussian_image" ALL
    DEPENDS "${DEST_IMAGE}")

add_dependencies(RecursiveGaussian RecursiveGaussian_image)
