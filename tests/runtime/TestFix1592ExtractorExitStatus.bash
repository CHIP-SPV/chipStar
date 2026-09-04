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
  echo "FAIL: the reproducer itself should exit 1, got ${DIRECT}"; exit 1
fi
if [ "${WRAPPED}" -ne 1 ]; then
  echo "FAIL: spirv-extractor --check-for-doubles turned exit ${DIRECT} into exit ${WRAPPED}"
  echo "      a failing test wrapped this way is reported as passing (issue #1592)"
  exit 1
fi
echo "PASSED"
