#!/bin/bash
# Canary for https://github.com/CHIP-SPV/chipStar/issues/1987. MEANT TO FAIL EVENTUALLY.
# Asserts IGC still crashes on a recursive function with a generic pointer
# argument on bmg and pvc. No upstream IGC report exists yet. When it fires,
# delete HipNoInlineRecursionPass from llvm_passes/HipPasses.cpp and this
# canary, then close #1987.
# Needs no GPU: ocloc is an offline compiler and -device names a target.
set -u
if ! command -v ocloc >/dev/null 2>&1; then
  echo "HIP_SKIP_THIS_TEST: ocloc not found"
  exit 0
fi
cd "$(mktemp -d)"
F='void fib(int *a, int n) { if (n < 2) { *a += n; return; } fib(a, n - 1); fib(a, n - 2); }'
K='kernel void k(global int *o, int n) { int a = 0; fib(&a, n); *o = a; }'
printf '%s\n%s\n' "__attribute__((noinline)) $F" "$K" > ok.cl
printf '%s\n%s\n' "$F" "$K" > bad.cl
for DEV in bmg pvc; do
  ocloc compile -file ok.cl -device $DEV -options -cl-std=CL2.0 > ok.log 2>&1 ||
    { echo "FAIL: the noinline control does not build for $DEV"; cat ok.log; exit 1; }
  if ocloc compile -file bad.cl -device $DEV -options -cl-std=CL2.0 > bad.log 2>&1; then
    echo "CANARY FIRED: IGC now builds recursion through a generic pointer on $DEV, see the header"
    exit 1
  elif [ $? != 139 ] && ! grep -q "Segmentation violation" bad.log; then
    echo "FAIL: $DEV rejects the recursion without the known crash"; cat bad.log; exit 1
  fi
done
echo PASSED
