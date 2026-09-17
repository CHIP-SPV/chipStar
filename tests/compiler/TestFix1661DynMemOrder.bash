#!/bin/bash
# Regression test for CHIP-SPV/chipStar#1661: lowering a module must not depend
# on allocation addresses, so repeated post-link pipeline runs agree.
set -eu

OPT="@LLVM_TOOLS_BINARY_DIR@/opt"
PLUGIN="@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"

# More than 16 users: a small pointer set iterates in insertion order anyway.
{
  echo '@__smem = external addrspace(3) global [0 x i32]'
  for i in $(seq 32); do
    printf 'define spir_kernel void @k%d(i32 %%v) {\n' "${i}"
    printf '  store i32 %%v, ptr addrspace(3) @__smem\n  ret void\n}\n'
  done
} > in.ll

# Heap addresses only change between processes, so compare several runs.
for i in $(seq 8); do
  "${OPT}" -load-pass-plugin "${PLUGIN}" -passes=hip-post-link-passes \
      in.ll -S -o "run${i}.ll"
  if ! cmp -s run1.ll "run${i}.ll"; then
    echo "FAIL: run ${i} lowered the same module differently from run 1"
    diff run1.ll "run${i}.ll" | head -10
    exit 1
  fi
done
echo "PASSED"
