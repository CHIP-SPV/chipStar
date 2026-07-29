# -fgpu-rdc objects cannot be built for MachO (macOS).
if(NOT APPLE)
  add_shell_test(TestStaticLibRDCOffloadArch.bash)
endif()
add_shell_test(TestStaticLibRDC.bash)