add_hip_test(test_stream_wait_event_sync.cpp)
add_hip_test(test_hipMemcpyHtoD_link.cpp)
add_hip_test(test_atomicmax_unsigned.cpp)
add_hip_test(TestFixCoalescedThreadsAtomicAggInc.hip)
set_tests_properties(TestFixCoalescedThreadsAtomicAggInc PROPERTIES PASS_REGULAR_EXPRESSION "PASS" FAIL_REGULAR_EXPRESSION "FAIL")

add_hip_test(TestFixUmul64hi.hip)
# Reproduce merkle-hip: __umul64hi used (x*y)>>64 (64-bit shift = UB in SPIR-V),
# causing IGC to terminate during JIT with "Termination request".
# Fixed by using mul_hi(ulong,ulong) from OpenCL builtins.
set_tests_properties(TestFixUmul64hi PROPERTIES
  PASS_REGULAR_EXPRESSION "PASS"
  TIMEOUT 60)

add_hip_test(test_shutdown_crossqueue_dep.cpp)
