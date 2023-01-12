add_custom_command(OUTPUT saxpy.o   DEPENDS ${CMAKE_SOURCE_DIR}/samples/ccompat/saxpy.hip COMMAND ${CMAKE_BINARY_DIR}/bin/hipcc.bin ${CMAKE_SOURCE_DIR}/samples/ccompat/saxpy.hip -c -g)
add_custom_command(OUTPUT ccompat.o DEPENDS ${CMAKE_SOURCE_DIR}/samples/ccompat/ccompat.c COMMAND ${CMAKE_BINARY_DIR}/bin/hipcc.bin ${CMAKE_SOURCE_DIR}/samples/ccompat/ccompat.c -c -g)

add_custom_target(ccompat DEPENDS saxpy.o ccompat.o COMMAND ${CMAKE_BINARY_DIR}/bin/hipcc.bin ./ccompat.o ./saxpy.o -o ccompat -g)

add_dependencies(ccompat hipcc.bin CHIP)