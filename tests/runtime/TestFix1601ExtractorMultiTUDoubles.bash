#!/bin/bash
# spirv-extractor --check-for-doubles must find fp64 in ANY device module of a
# binary, not just the first. It walked to the first __CLANG_OFFLOAD_BUNDLE__ in
# .hip_fatbin and stopped, so a multi-TU binary whose doubles live in a later
# translation unit was reported as double-free and ran anyway. On a device
# without fp64 that surfaces as a build failure rather than a skip, which is how
# cuda-reduction failed on Mali-G52 (chipStar issue #1601).
#
# The link order is the whole point: the fp64 TU goes second.
set -u
HIPCC="@CMAKE_BINARY_DIR@/bin/hipcc"
EXTRACTOR="@CMAKE_BINARY_DIR@/bin/spirv-extractor"
SRC1="@CMAKE_CURRENT_SOURCE_DIR@/TestFix1601ExtractorMultiTUDoubles.hip"
SRC2="@CMAKE_CURRENT_SOURCE_DIR@/inputs/ExtractorMultiTUDoublesTU2.hip"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

if [ ! -x "${EXTRACTOR}" ]; then
  # tools/spirv-extractor is built unconditionally, so its absence means a
  # broken build rather than an unsupported configuration.
  echo "FAIL: spirv-extractor was not built at ${EXTRACTOR}"
  exit 1
fi
rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}" || exit 1
"${HIPCC}" -O1 -c "${SRC1}" -o tu1.o > build.log 2>&1 || { echo "FAIL: could not build TU1"; tail -5 build.log; exit 1; }
"${HIPCC}" -O1 -c "${SRC2}" -o tu2.o >> build.log 2>&1 || { echo "FAIL: could not build TU2"; tail -5 build.log; exit 1; }
"${HIPCC}" tu1.o tu2.o -o twotu >> build.log 2>&1 || { echo "FAIL: could not link"; tail -5 build.log; exit 1; }

# -o then count lines: grep -c counts matching LINES, and two bundle markers
# can share one line of a binary, which would undercount to 1 and let the test
# conclude the multi-module case was unbuildable when it was not.
BUNDLES=$(grep -a -o "__CLANG_OFFLOAD_BUNDLE__" twotu 2>/dev/null | wc -l)
echo "offload bundles in the binary: ${BUNDLES}"
if [ "${BUNDLES}" -lt 2 ]; then
  echo "HIP_SKIP_THIS_TEST: this toolchain emitted ${BUNDLES} bundle(s), so the multi-module case cannot be built here"
  exit 0
fi

OUTPUT=$("${EXTRACTOR}" --check-for-doubles ./twotu 2>&1)
# Never echo the extractor's output verbatim: it contains the skip marker this
# test is looking FOR, and ctest would read it off this script's own stdout and
# call the test skipped.
if echo "${OUTPUT}" | grep -q "HIP_SKIP_THIS_TEST: Kernel uses doubles"; then
  echo "extractor skipped the binary, as it must"
  echo "PASSED"
  exit 0
fi
echo "FAIL: --check-for-doubles missed the fp64 kernel in the second translation unit"
echo "      and ran the program instead of skipping it (issue #1601)"
echo "--- extractor output, marker redacted so ctest does not read it as a skip:"
echo "${OUTPUT}" | grep -v "driver name\|pci id" | sed "s/HIP_SKIP_THIS_TEST/<skip-marker>/"
exit 1
