#!/bin/bash
# spirv-extractor --check-for-doubles must propagate the wrapped test's exit
# status. It returned system()'s raw wait status from main(), so a test that
# exits 1 came back as 256 -> 0 and ctest reported it passed. Every build with
# CHIP_SKIP_TESTS_WITH_DOUBLES=ON wraps its tests this way, so on those builds
# any test without a PASS/FAIL regex could fail silently (chipStar issue #1592).
set -u
HIPCC="@CMAKE_BINARY_DIR@/bin/hipcc"
EXTRACTOR="@CMAKE_BINARY_DIR@/bin/spirv-extractor"
SRC="@CMAKE_CURRENT_SOURCE_DIR@/TestFix1592ExtractorExitStatus.hip"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

if [ ! -x "${EXTRACTOR}" ]; then
  echo "HIP_SKIP_THIS_TEST: spirv-extractor not built"
  exit 0
fi
rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}" || exit 1
"${HIPCC}" -O2 "${SRC}" -o fails > build.log 2>&1 || { echo "FAIL: could not build the reproducer"; tail -5 build.log; exit 1; }

./fails > direct.log 2>&1; DIRECT=$?
"${EXTRACTOR}" --check-for-doubles ./fails > wrapped.log 2>&1; WRAPPED=$?
echo "direct exit=${DIRECT} wrapped exit=${WRAPPED}"

if [ "${DIRECT}" -ne 1 ]; then
  echo "FAIL: the reproducer itself should exit 1, got ${DIRECT}"; cat direct.log; exit 1
fi
if [ "${WRAPPED}" -ne 1 ]; then
  echo "FAIL: spirv-extractor --check-for-doubles turned exit ${DIRECT} into exit ${WRAPPED}"
  echo "      a failing test wrapped this way is reported as passing (issue #1592)"
  cat wrapped.log
  exit 1
fi

# A signal death must arrive as 128+signal, not as the shell's own status and
# not as a success. Without this, mapping the signal arm to 0 goes undetected.
"${EXTRACTOR}" --check-for-doubles ./fails abort > signal.log 2>&1; SIGNALED=$?
echo "signal death wrapped exit=${SIGNALED}"
if [ "${SIGNALED}" -ne 134 ]; then
  echo "FAIL: a wrapped test killed by SIGABRT must be reported as 134 (128+6),"
  echo "      got ${SIGNALED}"
  cat signal.log
  exit 1
fi

# A test the wrapper cannot execute must not be reported as a pass. Without
# this, mapping the spawn-failure arm to 0 goes undetected.
cp ./fails ./noexec && chmod -x ./noexec
"${EXTRACTOR}" --check-for-doubles ./noexec > noexec.log 2>&1; NOEXEC=$?
echo "unrunnable wrapped exit=${NOEXEC}"
if [ "${NOEXEC}" -eq 0 ]; then
  echo "FAIL: a test the wrapper could not execute was reported as passing"
  cat noexec.log
  exit 1
fi
# and it must say why, or a CI log shows a bare status with no cause.
if ! grep -q "spirv-extractor: could not run" noexec.log; then
  echo "FAIL: the wrapper failed to run ./noexec and printed no diagnostic"
  cat noexec.log
  exit 1
fi
echo "PASSED"
