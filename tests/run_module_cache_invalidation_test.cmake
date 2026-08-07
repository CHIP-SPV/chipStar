# Module cache invalidation test.
#
# The module cache maps a compiled device binary to a key. If that key misses an
# input that changes what the driver emits, a warm cache serves a kernel built
# from different inputs -- silently, and with no way to tell from the results.
# This drives one test binary repeatedly under a fresh cache directory and
# checks the cache entry count after each run, which is an observable proxy for
# "did the key change":
#
#   same inputs      -> no new entry (hit)   ... the key must be stable
#   changed inputs   -> a new entry (miss)   ... the key must invalidate
#
# Both directions matter. A key that never invalidates serves stale kernels; a
# key that always invalidates (e.g. one keyed on a build id or timestamp) makes
# the cache useless without failing any correctness test.
#
# libCHIP reads CHIP_MODULE_CACHE_DIR at static-init time, so it has to be set
# before the process starts; hence a script wrapper rather than a test fixture.
#
# Invoked as:
#   cmake -DTEST_EXECUTABLE=<path> -P run_module_cache_invalidation_test.cmake

if(NOT TEST_EXECUTABLE)
  message(FATAL_ERROR "TEST_EXECUTABLE not set")
endif()

# Lowercase-only directory name, deliberately not mktemp -d. This test looks at
# the cache directory from the outside to count entries, and the runtime
# currently lowercases CHIP_MODULE_CACHE_DIR (issue #1396), so a path containing
# uppercase characters -- which mktemp names routinely do -- would have its
# entries written somewhere else and every run would look like a miss. Staying
# lowercase makes the test agree with the runtime either way, before or after
# that bug is fixed.
string(RANDOM LENGTH 12 ALPHABET "abcdefghijklmnopqrstuvwxyz0123456789" SUFFIX)
set(CACHE_DIR "$ENV{TMPDIR}")
if(NOT CACHE_DIR)
  set(CACHE_DIR "/tmp")
endif()
string(TOLOWER "${CACHE_DIR}/chipstar-cache-test-${SUFFIX}" CACHE_DIR)
file(REMOVE_RECURSE "${CACHE_DIR}")
file(MAKE_DIRECTORY "${CACHE_DIR}")

set(FAILURES "")

# Run TEST_EXECUTABLE with CHIP_MODULE_CACHE_DIR set plus any extra NAME=VALUE
# entries, and report how many entries the cache holds afterwards.
function(run_and_count LABEL OUT_COUNT)
  # Clear the variables this test toggles before applying the one under test, so
  # each run is a known delta from the baseline rather than from whatever the
  # caller's environment happened to hold. This is not hypothetical: the x86 CI
  # exports OverrideDefaultFP64Settings=1 for every job, which would make the
  # step that adds it a no-op and fail the test for the wrong reason.
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env
      --unset=CHIP_JIT_FLAGS_OVERRIDE
      --unset=OverrideDefaultFP64Settings
      --unset=NEOReadDebugKeys
      "CHIP_MODULE_CACHE_DIR=${CACHE_DIR}"
      ${ARGN}
      "${TEST_EXECUTABLE}"
    RESULT_VARIABLE RUN_RC
    OUTPUT_VARIABLE RUN_OUT
    ERROR_VARIABLE RUN_ERR)
  # The payload signals a platform it cannot run on (e.g. Mali, whose
  # clLinkProgram is unconditionally broken) the same way every other test in
  # this suite does: HIP_SKIP_THIS_TEST on stdout. Surface that as a skip of
  # the whole ctest test rather than letting a non-zero exit fall through to
  # the FATAL_ERROR below, whose message happens to also contain the marker --
  # relying on that would make the skip path fragile to unrelated wording
  # changes in the error message.
  #
  # return() here only exits this function, not the script, so the actual
  # abort happens at the call site below via RUN_SKIPPED: every remaining
  # run would hit the identical, deterministic platform gate, so it is
  # enough to check once, after the first (cold) call.
  set(RUN_SKIPPED FALSE PARENT_SCOPE)
  if(RUN_OUT MATCHES "HIP_SKIP_THIS_TEST")
    message(STATUS "${RUN_OUT}")
    message(STATUS "HIP_SKIP_THIS_TEST: ${LABEL} reported an unsupported platform")
    set(RUN_SKIPPED TRUE PARENT_SCOPE)
    return()
  endif()
  if(NOT RUN_RC EQUAL 0)
    message(FATAL_ERROR
      "${LABEL}: test executable failed (exit ${RUN_RC})\n${RUN_OUT}\n${RUN_ERR}")
  endif()
  # Count only cache entries; a crashed writer could leave a *.tmp.<pid> behind
  # and those are not cache hits.
  file(GLOB ENTRIES "${CACHE_DIR}/*")
  set(REAL_ENTRIES "")
  foreach(ENTRY IN LISTS ENTRIES)
    if(NOT ENTRY MATCHES "\\.tmp\\.[0-9]+$")
      list(APPEND REAL_ENTRIES "${ENTRY}")
    endif()
  endforeach()
  list(LENGTH REAL_ENTRIES COUNT)
  message(STATUS "${LABEL}: ${COUNT} cache entr(y/ies)")
  set(${OUT_COUNT} ${COUNT} PARENT_SCOPE)
endfunction()

function(expect LABEL ACTUAL EXPECTED)
  if(NOT ACTUAL EQUAL EXPECTED)
    set(FAILURES "${FAILURES}  ${LABEL}: expected ${EXPECTED} entries, got ${ACTUAL}\n"
        PARENT_SCOPE)
  endif()
endfunction()

# 1. Cold: the first compile must populate the cache.
run_and_count("cold run" N1)
if(RUN_SKIPPED)
  file(REMOVE_RECURSE "${CACHE_DIR}")
  return()
endif()
if(N1 LESS 1)
  file(REMOVE_RECURSE "${CACHE_DIR}")
  message(FATAL_ERROR
    "cold run produced no cache entries; caching is not active, so this test "
    "cannot detect anything")
endif()

# 2. Identical inputs must hit. If this grows, the key depends on something that
#    varies run to run (a timestamp, a randomized std::hash, an unsorted
#    environment) and the cache would never be reused.
run_and_count("repeat run (expect hit)" N2)
expect("repeat run" ${N2} ${N1})

# 3. CHIP_JIT_FLAGS_OVERRIDE changes the flags handed to the driver, so it must
#    invalidate. It was previously absent from the key even though the x86 CI
#    sets it.
math(EXPR N3_EXPECTED "${N1} + ${N1}")
run_and_count("with CHIP_JIT_FLAGS_OVERRIDE (expect miss)" N3
  "CHIP_JIT_FLAGS_OVERRIDE=-cl-opt-disable")
expect("CHIP_JIT_FLAGS_OVERRIDE" ${N3} ${N3_EXPECTED})

# 4. OverrideDefaultFP64Settings switches on fp64 emulation, which changes the
#    device's reported fp64 capability and therefore which rtdevlib modules get
#    linked. This covers the rtdevlib half of the key: the selected modules are
#    hashed, so a different selection must produce a different key.
math(EXPR N4_EXPECTED "${N3} + ${N1}")
run_and_count("with OverrideDefaultFP64Settings (expect miss)" N4
  "OverrideDefaultFP64Settings=1")
expect("OverrideDefaultFP64Settings" ${N4} ${N4_EXPECTED})

# 5. A compiler environment variable that does NOT change device capabilities
#    must still invalidate, because it still reaches the device compiler. This
#    is the one that pins the environment half of the key: it can only be caught
#    by collecting the variable itself, and the collector used to match IGC_*
#    only, so anything from the Compute Runtime or the Level Zero loader was
#    invisible to the key.
math(EXPR N5_EXPECTED "${N4} + ${N1}")
run_and_count("with NEOReadDebugKeys (expect miss)" N5 "NEOReadDebugKeys=1")
expect("NEOReadDebugKeys" ${N5} ${N5_EXPECTED})

# 6. Back to the original inputs: must hit the entry from step 1, proving the
#    key invalidates on real changes rather than simply always changing.
run_and_count("back to baseline (expect hit)" N6)
expect("baseline revisit" ${N6} ${N5})

file(REMOVE_RECURSE "${CACHE_DIR}")

if(NOT FAILURES STREQUAL "")
  message(FATAL_ERROR "module cache invalidation test failed:\n${FAILURES}")
endif()
message(STATUS "module cache invalidation test passed")
