# Test symbol access
#set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_chip_binary(
	hipTestDeviceLink
	hipDeviceLink.cpp
	hipDeviceLinkRead.cpp
	hipDeviceLinkWrite.cpp)

target_compile_options(hipTestDeviceLink PRIVATE -fPIE -fgpu-rdc)
target_link_options(hipTestDeviceLink PRIVATE -fPIE -fgpu-rdc --hip-link)


add_test(NAME hipTestDeviceLink
	 COMMAND "${CMAKE_CURRENT_BINARY_DIR}/hipTestDeviceLink"
	 )

set_tests_properties(hipTestDeviceLink PROPERTIES
	PASS_REGULAR_EXPRESSION PASSED)
	
