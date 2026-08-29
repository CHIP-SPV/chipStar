# Module cache invalidation test.
#
# The module cache maps a compiled device binary to a key. If that key misses an
# input that changes what the driver emits, a warm cache serves a kernel built
# from different inputs -- silently, and with no way to tell from the results.
# This drives one test binary repeatedly under a fresh cache directory and
# asserts on the runtime's own outcome markers:
#
#   module-cache: HIT ...        the key matched and the cached binary BUILT
#   module-cache: MISS ...       no (valid) entry for this key
#   module-cache: REJECTED ...   an entry existed but its bytes were refused
#
#   same inputs      -> HIT   ... the key must be stable
#   changed inputs   -> MISS  ... the key must invalidate
#
# Both directions matter. A key that never invalidates serves stale kernels; a
# key that always invalidates (e.g. one keyed on a build id or timestamp) makes
# the cache useless without failing any correctness test.
#
# Markers, not file counting: a reader that rejects every file still grows the
# file count on every run, so counting files cannot detect the one failure mode
# a cache format change is most likely to introduce. The HIT marker is emitted
# only after the cached binary actually rebuilt into a program, so "HIT" means
# "we did not JIT", and REJECTED must never appear.
#
# libCHIP reads CHIP_MODULE_CACHE_DIR at static-init time, so it has to be set
# before the process starts; hence a script wrapper rather than a test fixture.
# The markers are info level (the release default is warn), so every run sets
# CHIP_LOGLEVEL=info, and spdlog writes to stderr, so RUN_ERR is what gets
# parsed.
#
# Invoked as:
#   cmake -DTEST_EXECUTABLE=<path> -P run_module_cache_invalidation_test.cmake

if(NOT TEST_EXECUTABLE)
  message(FATAL_ERROR "TEST_EXECUTABLE not set")
endif()

# Lowercase-only directory name, deliberately not mktemp -d: the runtime
# currently lowercases CHIP_MODULE_CACHE_DIR (issue #1396), so a path with
# uppercase characters would have its entries written somewhere else. Staying
# lowercase makes the test agree with the runtime before and after that fix.
string(RANDOM LENGTH 12 ALPHABET "abcdefghijklmnopqrstuvwxyz0123456789" SUFFIX)
set(SCRATCH_DIR "$ENV{TMPDIR}")
if(NOT SCRATCH_DIR)
  set(SCRATCH_DIR "/tmp")
endif()
string(TOLOWER "${SCRATCH_DIR}/chipstar-cache-test-${SUFFIX}" SCRATCH_DIR)
set(CACHE_DIR "${SCRATCH_DIR}/cache")
file(REMOVE_RECURSE "${SCRATCH_DIR}")
file(MAKE_DIRECTORY "${CACHE_DIR}")

set(FAILURES "")

# Run TEST_EXECUTABLE with the cache configured plus any extra NAME=VALUE
# entries, and report the HIT/MISS/REJECTED marker counts and the backend the
# markers name.
function(run_and_expect LABEL EXPECT_HITS EXPECT_MISSES)
  # Clear the variables this test toggles before applying the one under test,
  # so each run is a known delta from the baseline rather than from whatever
  # the caller's environment happened to hold. This is not hypothetical: the
  # x86 CI exports OverrideDefaultFP64Settings=1 for every job, which would
  # make the step that adds it a no-op and fail the test for the wrong reason.
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env
      --unset=CHIP_JIT_FLAGS_OVERRIDE
      --unset=OverrideDefaultFP64Settings
      --unset=NEOReadDebugKeys
      --unset=IGC_ShaderDumpRegexFilter
      # OCL_ICD_VENDORS and OCL_ICD_FILENAMES are deliberately NOT unset:
      # macOS and the hosted runners register pocl through OCL_ICD_VENDORS,
      # and the meatloaf lanes register the Intel driver through
      # OCL_ICD_FILENAMES (the oneAPI Khronos loader honors it; with it
      # stripped the loader has no platforms and init fails). The
      # loader-delta step overrides both per run instead.
      "CHIP_MODULE_CACHE_DIR=${CACHE_DIR}"
      "CHIP_LOGLEVEL=info"
      ${ARGN}
      "${TEST_EXECUTABLE}"
    RESULT_VARIABLE RUN_RC
    OUTPUT_VARIABLE RUN_OUT
    ERROR_VARIABLE RUN_ERR)
  # The payload signals a platform it cannot run on (e.g. Mali, whose
  # clLinkProgram is unconditionally broken) the same way every other test in
  # this suite does: HIP_SKIP_THIS_TEST on stdout. Surface that as a skip of
  # the whole ctest test. return() only exits this function; the actual abort
  # happens at the first (cold) call site via RUN_SKIPPED, since every later
  # run would hit the identical, deterministic platform gate.
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

  string(REGEX MATCHALL "module-cache: HIT" HIT_LIST "${RUN_ERR}")
  string(REGEX MATCHALL "module-cache: MISS" MISS_LIST "${RUN_ERR}")
  string(REGEX MATCHALL "module-cache: REJECTED" REJ_LIST "${RUN_ERR}")
  list(LENGTH HIT_LIST HITS)
  list(LENGTH MISS_LIST MISSES)
  list(LENGTH REJ_LIST REJECTED)
  message(STATUS "${LABEL}: ${HITS} hit(s), ${MISSES} miss(es), ${REJECTED} rejected")

  # Which backend produced the markers; used to gate backend-specific steps.
  if(RUN_ERR MATCHES "module-cache: (HIT|MISS) backend=([a-z0-9]+)")
    set(SEEN_BACKEND "${CMAKE_MATCH_2}" PARENT_SCOPE)
  endif()

  # Absolute counts, never deltas: a misspelled marker pattern would make
  # every count zero and a delta check would pass vacuously.
  if(NOT REJECTED EQUAL 0)
    set(FAILURES "${FAILURES}  ${LABEL}: ${REJECTED} REJECTED marker(s); the reader refused an entry it wrote\n"
        PARENT_SCOPE)
  endif()
  if(NOT EXPECT_HITS STREQUAL "any" AND NOT HITS EQUAL ${EXPECT_HITS})
    set(FAILURES "${FAILURES}  ${LABEL}: expected ${EXPECT_HITS} hit(s), got ${HITS}\n"
        PARENT_SCOPE)
  endif()
  if(NOT EXPECT_MISSES STREQUAL "any" AND NOT MISSES EQUAL ${EXPECT_MISSES})
    set(FAILURES "${FAILURES}  ${LABEL}: expected ${EXPECT_MISSES} miss(es), got ${MISSES}\n"
        PARENT_SCOPE)
  endif()
  set(LAST_HITS ${HITS} PARENT_SCOPE)
  set(LAST_MISSES ${MISSES} PARENT_SCOPE)
endfunction()

# 1. Cold: the first compile must populate the cache. The number of misses is
#    the number of modules this payload compiles; every later expectation is
#    stated in terms of it.
run_and_expect("cold run" 0 "any")
if(RUN_SKIPPED)
  file(REMOVE_RECURSE "${SCRATCH_DIR}")
  return()
endif()
if(LAST_MISSES LESS 1)
  file(REMOVE_RECURSE "${SCRATCH_DIR}")
  message(FATAL_ERROR
    "cold run produced no MISS markers; caching is not active (or the marker "
    "format changed), so this test cannot detect anything")
endif()
set(N ${LAST_MISSES})

# 2. Identical inputs must hit -- and HIT means the cached binary actually
#    rebuilt into a program, not merely that a file existed. If this misses,
#    the key depends on something that varies run to run (a timestamp, an
#    unsorted environment); if it rejects, the writer and reader disagree on
#    the format.
run_and_expect("repeat run (expect hit)" ${N} 0)

# 3. CHIP_JIT_FLAGS_OVERRIDE changes the options string handed to the driver,
#    which the key hashes verbatim, so it must miss. A bare -cl-opt-disable is
#    deliberate: the override replaces the OpenCL backend's defaults
#    (-cl-std=CL3.0 among them) for the user's program, and the rtdevlib link
#    must survive that (issue #1532: the Intel CPU runtime resolves the
#    library's generic-pointer float atomic only under CL2.0 or newer), so
#    this step also checks that the library keeps its own options.
run_and_expect("with CHIP_JIT_FLAGS_OVERRIDE (expect miss)" 0 ${N}
  "CHIP_JIT_FLAGS_OVERRIDE=-cl-opt-disable")

# 4. OverrideDefaultFP64Settings switches on fp64 emulation, which changes the
#    device's reported fp64 capability and therefore which rtdevlib modules
#    get linked, as well as the environment digest. Must miss.
run_and_expect("with OverrideDefaultFP64Settings (expect miss)" 0 ${N}
  "OverrideDefaultFP64Settings=1")

# 5. A Compute Runtime variable that does NOT change device capabilities must
#    still invalidate: NEOReadDebugKeys makes NEO read all of its debug
#    variables by bare name, so it is hashed like any other NEO variable and
#    widens what the environment digest covers.
run_and_expect("with NEOReadDebugKeys (expect miss)" 0 ${N}
  "NEOReadDebugKeys=1")

# 5b. The other half of that rule: with the gate on, a variable NEO cannot read
#     as a debug key must NOT move the key. Batch schedulers hand every launch
#     fresh PBS_*/PALS_*/HOSTNAME values, and hashing them made the key unique
#     per run, so a job could never hit the cache written by the previous one.
#     Both runs reuse the entries step 5 wrote, so both must hit.
run_and_expect("NEOReadDebugKeys plus launcher variables (expect hit)" ${N} 0
  "NEOReadDebugKeys=1" "PBS_JOBID=1.aurora" "PALS_APID=aaa" "HOSTNAME=x4001")
run_and_expect("NEOReadDebugKeys plus different launcher variables (expect hit)" ${N} 0
  "NEOReadDebugKeys=1" "PBS_JOBID=2.aurora" "PALS_APID=bbb" "HOSTNAME=x4002")

# 6. An IGC_-prefixed variable must invalidate purely through the environment
#    digest -- IGC reads every IGC_* name from the environment, and the value
#    chosen here is inert so the compile itself is unchanged.
run_and_expect("with IGC_ShaderDumpRegexFilter (expect miss)" 0 ${N}
  "IGC_ShaderDumpRegexFilter=nomatchxyz")

# 7. Back to the original inputs: must hit the entries from step 1, proving
#    the key invalidates on real changes rather than simply always changing.
run_and_expect("back to baseline (expect hit)" ${N} 0)

# 8. Runtime change: loading one more library during driver init must move the
#    loader-delta digest and therefore the key. This is the end-to-end check
#    that a swapped device compiler (IGC module load, package upgrade)
#    invalidates the cache -- the mechanism is the same: a different library
#    set at init. Implemented by redirecting the ICD loader to a vendors
#    directory whose single .icd names a COPY of the installed driver, so the
#    same driver code loads from a different path. OCL_ICD_VENDORS is the one
#    variable both loaders honor: Ubuntu's ocl-icd has no OCL_ICD_FILENAMES
#    support at all (verified against its binary), while the Khronos loader
#    supports both. Only applies when the payload actually ran on the OpenCL
#    backend, and needs an absolute library path in some
#    /etc/OpenCL/vendors/*.icd to copy.
if(SEEN_BACKEND STREQUAL "opencl")
  set(ICD_COPY "")
  # Candidate driver libraries: the first entry of every /etc/OpenCL/vendors
  # .icd, plus an OCL_ICD_FILENAMES value when the environment registers the
  # driver that way (meatloaf's module env does).
  set(ICD_CANDIDATES "")
  file(GLOB ICD_FILES "/etc/OpenCL/vendors/*.icd")
  foreach(ICD_FILE IN LISTS ICD_FILES)
    file(STRINGS "${ICD_FILE}" ICD_LINES LIMIT_COUNT 1)
    list(GET ICD_LINES 0 ICD_PATH)
    string(STRIP "${ICD_PATH}" ICD_PATH)
    list(APPEND ICD_CANDIDATES "${ICD_PATH}")
  endforeach()
  if(DEFINED ENV{OCL_ICD_FILENAMES})
    list(APPEND ICD_CANDIDATES "$ENV{OCL_ICD_FILENAMES}")
  endif()
  foreach(ICD_PATH IN LISTS ICD_CANDIDATES)
    if(ICD_PATH MATCHES "^/" AND EXISTS "${ICD_PATH}")
      set(ICD_COPY "${SCRATCH_DIR}/icd-copy.so")
      # copy_file in script mode needs CMake >= 3.21; -E copy works on 3.20.
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy "${ICD_PATH}" "${ICD_COPY}")
      file(MAKE_DIRECTORY "${SCRATCH_DIR}/vendors")
      file(WRITE "${SCRATCH_DIR}/vendors/icd-copy.icd" "${ICD_COPY}\n")
      message(STATUS "loader-delta step: shadowing ${ICD_PATH}")
      break()
    endif()
  endforeach()
  if(ICD_COPY)
    # The relocated driver library changes the init loader delta: every
    # module must miss under the new digest. The redirected loader exposes
    # only the copied ICD's platform, so the caller's CHIP_PLATFORM /
    # CHIP_DEVICE_TYPE (set per lane by check.py) may point at a platform
    # that no longer exists; select platform 0 device 0 explicitly for
    # these two runs.
    # OCL_ICD_FILENAMES is overridden too: left at its ambient value the
    # Khronos loader would keep the original driver registered alongside the
    # copy, and platform 0 could still resolve to the original, leaving the
    # loader delta unchanged.
    run_and_expect("with relocated ICD library (expect miss)" 0 ${N}
      "OCL_ICD_VENDORS=${SCRATCH_DIR}/vendors"
      "OCL_ICD_FILENAMES=${ICD_COPY}"
      "CHIP_PLATFORM=0" "CHIP_DEVICE=0" --unset=CHIP_DEVICE_TYPE)
    # ...and the digest must be deterministic, not merely different: the same
    # setup again must hit the entries just written.
    run_and_expect("relocated ICD library again (expect hit)" ${N} 0
      "OCL_ICD_VENDORS=${SCRATCH_DIR}/vendors"
      "OCL_ICD_FILENAMES=${ICD_COPY}"
      "CHIP_PLATFORM=0" "CHIP_DEVICE=0" --unset=CHIP_DEVICE_TYPE)
  else()
    message(STATUS
      "loader-delta step skipped: no absolute ICD path found under /etc/OpenCL/vendors")
  endif()
else()
  message(STATUS
    "loader-delta step skipped: payload ran on '${SEEN_BACKEND}', not opencl "
    "(the delta mechanism is backend-agnostic and unit tested; the end-to-end "
    "check needs the OpenCL ICD loader)")
endif()

file(REMOVE_RECURSE "${SCRATCH_DIR}")

if(NOT FAILURES STREQUAL "")
  message(FATAL_ERROR "module cache invalidation test failed:\n${FAILURES}")
endif()
message(STATUS "module cache invalidation test passed")
