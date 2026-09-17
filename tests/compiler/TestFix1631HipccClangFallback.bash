#!/bin/bash
# Reproducer for CHIP-SPV/chipStar#1631: hipcc must fall back to <HIP_CLANG_PATH>/clang
# when clang++ is missing, and must keep using a bare-name HIP_COMPILER_BIN found on PATH.
set -eu
unset HIP_COMPILER_BIN HIP_PATH HIP_PLATFORM

HIPCC=@CMAKE_BINARY_DIR@/bin/hipcc
WORK=@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d

rm -rf "${WORK}"
mkdir -p "${WORK}/clangdir" "${WORK}/pathdir"
cd "${WORK}"

printf '#!/bin/bash\necho 22.0.0\n' > clangdir/llvm-config
printf '#!/bin/bash\nexit 0\n' > clangdir/clang
cp clangdir/clang pathdir/testfix1631-clang++
chmod +x clangdir/llvm-config clangdir/clang pathdir/testfix1631-clang++

echo 'int main() { return 0; }' > empty.hip

run_hipcc() {
  HIPCC_VERBOSE=1 HIP_CLANG_PATH="${WORK}/clangdir" "$@" \
    "${HIPCC}" -c empty.hip -o empty.o > out.txt 2>&1 || true
  grep -m1 '^hipcc-cmd: ' out.txt || echo "hipcc-cmd: <no command printed>"
}

CMD=$(run_hipcc env)
case "${CMD}" in
  "hipcc-cmd: ${WORK}/clangdir/clang "*) ;;
  *) echo "FAIL: no clang++ in HIP_CLANG_PATH, expected ${WORK}/clangdir/clang, got: ${CMD}"
     exit 1 ;;
esac

CMD=$(run_hipcc env "PATH=${WORK}/pathdir:${PATH}" HIP_COMPILER_BIN=testfix1631-clang++)
case "${CMD}" in
  "hipcc-cmd: testfix1631-clang++ "*) ;;
  *) echo "FAIL: bare-name HIP_COMPILER_BIN, expected testfix1631-clang++, got: ${CMD}"
     exit 1 ;;
esac

# The AMD backend carries the same code; a relocated hipcc finds no .hipInfo and selects it.
mkdir -p "${WORK}/amdbin"
cp @CMAKE_BINARY_DIR@/bin/hipcc.bin "${WORK}/amdbin/"
HIPCC="${WORK}/amdbin/hipcc.bin"

CMD=$(run_hipcc env HIP_PLATFORM=amd)
case "${CMD}" in
  "hipcc-cmd: ${WORK}/clangdir/clang "*) ;;
  *) echo "FAIL: amd backend, expected ${WORK}/clangdir/clang, got: ${CMD}"
     exit 1 ;;
esac

CMD=$(run_hipcc env HIP_PLATFORM=amd "PATH=${WORK}/pathdir:${PATH}" \
  HIP_COMPILER_BIN=testfix1631-clang++)
case "${CMD}" in
  "hipcc-cmd: testfix1631-clang++ "*) ;;
  *) echo "FAIL: amd backend bare name, expected testfix1631-clang++, got: ${CMD}"
     exit 1 ;;
esac

echo PASSED
