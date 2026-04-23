# Tests for chip-kernel-verify (HIPCC_VERIFY post-compile check).

# Reproducer for intel/intel-graphics-compiler#403. Fixture is real-world
# SPIR-V from rocThrust set_difference_by_key, dumped via
# CHIP_DUMP_PROCESSED_SPIRV. On dg2 IGC silently drops 3 lookback_set_op_kernel
# template instantiations (ff/ii/jj element types) under the SIMD32
# register-pressure retry path.
#
# This test is expected to FAIL on an IGC that still exhibits the bug —
# chip-kernel-verify exits 1 and names the dropped kernels. When IGC is fixed
# (or the fixture becomes stale), the test will start passing; that's the
# signal to remove the fixture.
add_test(NAME hipcc_verify_detects_dropped_kernels
  COMMAND ${CMAKE_BINARY_DIR}/bin/chip-kernel-verify
          ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/lookback_set_ops.spv)
set_tests_properties(hipcc_verify_detects_dropped_kernels PROPERTIES
  ENVIRONMENT "HIPCC_VERIFY_DEVICES=dg2"
  TIMEOUT 120)

# Sanity: off mode must produce no output and exit 0 regardless of input.
add_test(NAME hipcc_verify_off_mode_silent
  COMMAND ${CMAKE_BINARY_DIR}/bin/chip-kernel-verify
          ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/lookback_set_ops.spv)
set_tests_properties(hipcc_verify_off_mode_silent PROPERTIES
  ENVIRONMENT "HIPCC_VERIFY=0"
  FAIL_REGULAR_EXPRESSION "missing from native binary|ocloc"
  TIMEOUT 30)
