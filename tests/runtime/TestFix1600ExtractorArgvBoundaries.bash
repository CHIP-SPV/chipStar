#!/bin/bash
# spirv-extractor --check-for-doubles must hand the wrapped test its arguments
# byte for byte. It joined them into one string for system(), so the shell
# re-split any argument containing a space and Catch2 tests whose case name has
# a space ran with a corrupted argv (chipStar issue #1600).
#
# Checking the exit status alone cannot catch this, so the child asserts its own
# argument count and contents.
set -u
HIPCC="@CMAKE_BINARY_DIR@/bin/hipcc"
EXTRACTOR="@CMAKE_BINARY_DIR@/bin/spirv-extractor"
SRC="@CMAKE_CURRENT_SOURCE_DIR@/TestFix1600ExtractorArgvBoundaries.hip"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"
ARG="Unit_Device_sincos_Accuracy_Positive - float"

if [ ! -x "${EXTRACTOR}" ]; then
  echo "HIP_SKIP_THIS_TEST: spirv-extractor not built"
  exit 0
fi
rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}" || exit 1
"${HIPCC}" -O2 "${SRC}" -o argvcheck > build.log 2>&1 || { echo "FAIL: could not build the reproducer"; tail -5 build.log; exit 1; }

./argvcheck "${ARG}" > direct.log 2>&1; DIRECT=$?
"${EXTRACTOR}" --check-for-doubles ./argvcheck "${ARG}" > wrapped.log 2>&1; WRAPPED=$?
echo "direct exit=${DIRECT} wrapped exit=${WRAPPED}"

if [ "${DIRECT}" -ne 0 ]; then
  echo "FAIL: the reproducer does not agree with itself when run directly"; cat direct.log; exit 1
fi
if [ "${WRAPPED}" -ne 0 ]; then
  echo "FAIL: spirv-extractor altered the wrapped test's arguments (issue #1600)"
  cat wrapped.log
  exit 1
fi
echo "PASSED"
