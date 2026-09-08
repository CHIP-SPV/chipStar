#!/bin/bash
# Gates chipStar issue #1600: the wrapper must hand the child its arguments byte
# for byte. The child asserts its own argv, because the wrapper's exit status
# cannot carry that.
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
# Redacted because ctest reads this script's stdout and add_shell_test treats a
# raw marker as a skip.
show() { sed "s/HIP_SKIP_THIS_TEST/<skip-marker>/g" "$1"; }

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}" || exit 1
"${HIPCC}" -O2 "${SRC}" -o argvcheck > build.log 2>&1 || { echo "FAIL: could not build the reproducer"; tail -5 build.log; exit 1; }

./argvcheck "${ARG}" > direct.log 2>&1; DIRECT=$?
"${EXTRACTOR}" --check-for-doubles ./argvcheck "${ARG}" > wrapped.log 2>&1; WRAPPED=$?
echo "direct exit=${DIRECT} wrapped exit=${WRAPPED}"

if [ "${DIRECT}" -ne 0 ]; then
  echo "FAIL: the reproducer does not agree with itself when run directly"; show direct.log; exit 1
fi
if grep -q "HIP_SKIP_THIS_TEST" wrapped.log; then
  echo "FAIL: the wrapper skipped ./argvcheck instead of running it, so nothing"
  echo "      exercised its argv; this test cannot gate issue #1600 that way"
  show wrapped.log
  exit 1
fi
if ! grep -qx "argc=2" wrapped.log ||
   ! grep -qxF "  argv[1]=<${ARG}>" wrapped.log; then
  echo "FAIL: spirv-extractor altered the wrapped test's arguments (issue #1600)"
  show wrapped.log
  exit 1
fi
if [ "${WRAPPED}" -ne 0 ]; then
  echo "FAIL: the child agreed on its argv but the wrapper still returned ${WRAPPED}"
  show wrapped.log
  exit 1
fi
echo "PASSED"
