#!/bin/bash
# Gates chipStar issue #1613: an input with no SPIR-V must produce a diagnostic,
# not a read past the end of the buffer.
set -u
EXTRACTOR="@CMAKE_BINARY_DIR@/bin/spirv-extractor"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}" || exit 1
cp "$(type -P true)" ./plain || exit 1
head -c 64 ./plain > ./truncated

for IN in plain truncated; do
  "${EXTRACTOR}" ./${IN} > ${IN}.log 2>&1; RC=$?
  echo "${IN}: exit=${RC}"
  if [ "${RC}" -ne 1 ] || ! grep -q "Failed to extract SPIR-V" ${IN}.log; then
    echo "FAIL: expected exit 1 with a diagnostic for a non-fatbinary input"
    cat ${IN}.log; exit 1
  fi
done

"${EXTRACTOR}" --check-for-doubles ./plain > wrapped.log 2>&1; RC=$?
echo "--check-for-doubles plain: exit=${RC}"
if [ "${RC}" -ne 0 ]; then
  echo "FAIL: --check-for-doubles should run the non-fatbinary and return its 0"
  cat wrapped.log; exit 1
fi
echo "PASSED"
