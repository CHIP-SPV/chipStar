# Test symbol access

add_chip_test(hipTestDeviceSymbol hipTestDeviceSymbol PASSED
  hipTestDeviceSymbol.cpp)
add_chip_test(hipTestConstantDeviceSymbol hipConstantTestDeviceSymbol PASSED
  hipTestConstantDeviceSymbol.cpp)
add_chip_test(hipTestSymbolReset hipTestSymbolReset PASSED
  hipTestSymbolReset.cc)
# Compares addresses of __device__ symbols, which native Vulkan cannot do.
if(NOT CHIP_USE_NATIVE_VULKAN_SPIRV)
  add_chip_test(hipTestSymbolInit hipTestSymbolInit PASSED hipTestSymbolInit.cc)
endif()
add_chip_test(hipTestResetStaticVar hipTestResetStaticVar PASSED
  hipTestResetStaticVar.cc)
add_chip_test(hipTestVariableTemplateSymbols hipTestVariableTemplateSymbols
  PASSED hipTestVariableTemplateSymbols.cc)
