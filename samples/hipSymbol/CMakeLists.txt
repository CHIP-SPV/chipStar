# Test symbol access

add_chip_test(hipTestDeviceSymbol hipTestDeviceSymbol PASSED
  hipTestDeviceSymbol.cpp)
add_chip_test(hipTestConstantDeviceSymbol hipConstantTestDeviceSymbol PASSED
  hipTestConstantDeviceSymbol.cpp)

