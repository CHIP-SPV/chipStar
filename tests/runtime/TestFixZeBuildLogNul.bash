#!/bin/bash
# The runtime's diagnostic output must be NUL-free, and a build log must only
# be printed when there is one.
#
# The Level Zero backend's dumpBuildLog() copies zeModuleBuildLogGetString's
# output into a std::string using the size the driver reports, which counts
# the terminating NUL, and logs "ZE Build Log:\n<log>" at info level for every
# zeModuleCreate, empty log or not. At CHIP_LOGLEVEL=info this puts a literal
# NUL byte on stderr after each "ZE Build Log:" line. gtest death tests treat
# the child's captured stderr as a C string, so on PVC the NUL truncated it
# before the expected abort message and every Kokkos_CoreUnitTest_HIP death
# test failed with "died but not with expected error".
#
# Runs TestKernelArgs (any small test that builds a module will do) at info
# level, captures stderr and fails if it holds a NUL byte, or if a
# "ZE Build Log:" line is followed by an empty log. Both checks are backend
# agnostic. Only the Level Zero backend prints "ZE Build Log:", so on an
# OpenCL run (the CPU gate) a pass proves only that the OpenCL path is clean;
# run with CHIP_BE=level0 to exercise the path this guards.
set -u

BIN="@CMAKE_CURRENT_BINARY_DIR@/TestKernelArgs"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

rm -rf "${OUT}"
mkdir -p "${OUT}"
cd "${OUT}"

CHIP_LOGLEVEL=info "${BIN}" > stdout.log 2> stderr.log
RC=$?
if [ "${RC}" -ne 0 ]; then
  echo "FAIL: ${BIN} exited with ${RC}"
  exit 1
fi

STATUS=0
fail() {
  echo "FAIL: $1"
  STATUS=1
}

# Backend actually exercised, for the record.
if grep -q "ZE Build Log:" stderr.log; then
  echo "Level Zero backend: 'ZE Build Log:' lines seen"
else
  echo "NOTE: no 'ZE Build Log:' line; the Level Zero backend was not exercised"
fi

NULS=$(tr -cd '\000' < stderr.log | wc -c)
if [ "${NULS}" -ne 0 ]; then
  fail "stderr contains ${NULS} NUL byte(s):"
  grep -a -n -B1 -P '\x00' stderr.log | head -6 | cat -v
fi

# Each "ZE Build Log:" line must be followed by a non-empty log line. NULs are
# stripped first so this check is about printing for nothing, not about the
# NUL itself.
EMPTY=$(tr -d '\000' < stderr.log |
        awk '/ZE Build Log:/ { getline nxt; if (nxt == "") n++ } END { print n+0 }')
if [ "${EMPTY}" -ne 0 ]; then
  fail "${EMPTY} 'ZE Build Log:' line(s) printed with an empty log"
fi

if [ "${STATUS}" -ne 0 ]; then
  echo "See ${OUT}/stderr.log"
  exit 1
fi
echo "PASSED"
