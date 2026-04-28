# Tests for chip-kernel-verify (HIPCC_VERIFY post-compile check).

# Reproducer for intel/intel-graphics-compiler#403. Fixture is real-world
# SPIR-V from rocThrust set_difference_by_key, dumped via
# CHIP_DUMP_PROCESSED_SPIRV. On dg2 IGC silently drops 3 lookback_set_op_kernel
# template instantiations (ff/ii/jj element types) under the SIMD32
# register-pressure retry path.
#
# WILL_FAIL flips the exit-code interpretation: while the bug is present
# chip-kernel-verify exits 1 (3 kernels missing), ctest reports the test as
# Passed, and the verifier output naming the dropped symbols is still visible
# in ctest -V / CI logs. If IGC is eventually fixed the verifier will exit 0
# and this test starts FAILING — that's the signal to retire the fixture.
# When ocloc is unavailable (e.g. macOS / aarch64 CI) the verifier prints
# "ocloc not found" and exits 0; SKIP_REGULAR_EXPRESSION marks the test as
# Skipped so it doesn't get flagged as an unexpected pass.
add_test(NAME hipcc_verify_detects_dropped_kernels
  COMMAND ${CMAKE_BINARY_DIR}/bin/chip-kernel-verify
          ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/lookback_set_ops.spv)
set_tests_properties(hipcc_verify_detects_dropped_kernels PROPERTIES
  ENVIRONMENT "HIPCC_VERIFY_DEVICES=dg2"
  WILL_FAIL TRUE
  SKIP_REGULAR_EXPRESSION "ocloc not found"
  TIMEOUT 120)

# Sanity: off mode must produce no output and exit 0 regardless of input.
add_test(NAME hipcc_verify_off_mode_silent
  COMMAND ${CMAKE_BINARY_DIR}/bin/chip-kernel-verify
          ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/lookback_set_ops.spv)
set_tests_properties(hipcc_verify_off_mode_silent PROPERTIES
  ENVIRONMENT "HIPCC_VERIFY=0"
  FAIL_REGULAR_EXPRESSION "missing from native binary|ocloc"
  TIMEOUT 30)
