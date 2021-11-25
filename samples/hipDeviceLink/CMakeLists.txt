# Test symbol access

add_chip_binary(
	hipTestDeviceLink
	hipDeviceLink.cpp
	hipDeviceLinkRead.cpp
	hipDeviceLinkWrite.cpp)

add_test(NAME hipTestDeviceLink
	 COMMAND "${CMAKE_CURRENT_BINARY_DIR}/hipTestDeviceLink"
	 )

set_tests_properties(hipTestDeviceLink PROPERTIES
	PASS_REGULAR_EXPRESSION PASSED)
	
