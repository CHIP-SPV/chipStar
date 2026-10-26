#!/bin/bash
# Canary for https://github.com/CHIP-SPV/chipStar/issues/1633. MEANT TO FAIL EVENTUALLY.
# The in-tree backend half of CanaryHintIntrinsicsRejected: asserts llc
# -mtriple=spirv64 still fails to legalize the intrinsics
# HipLowerHintIntrinsicsPass (llvm_passes/workarounds/HipLowerHintIntrinsics.*)
# erases or folds that every supported LLVM (21, 22, 23) rejects. prefetch is implemented by
# https://github.com/llvm/llvm-project/pull/215505 (main, LLVM 24), which is
# not enough by itself, see the pass. When it fires for one, delete that case
# from the pass only once CanaryHintIntrinsicsRejected has fired for it too.
# Delete the pass file when both canaries have fired for every case, then close
# the issue.
set -u
LLC="@LLVM_TOOLS_BINARY_DIR@/llc"
cd "$(mktemp -d)"
"$LLC" --version | grep -q spirv64 || { echo "HIP_SKIP_THIS_TEST: llc has no SPIR-V target"; exit 0; }

emit() { # $1 declaration, $2 call
  printf 'target triple = "spirv64-unknown-unknown"\ndeclare %s\ndefine spir_kernel void @k(ptr addrspace(1) %%p) {\n  %s\n  ret void\n}\n' "$1" "$2" > m.ll
  "$LLC" -mtriple=spirv64v1.3-unknown-chipstar -filetype=obj m.ll -o m.spv > m.log 2>&1
}

emit 'void @llvm.memcpy.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)' \
  'call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %p, ptr addrspace(1) %p, i64 1, i1 false)' ||
  { echo "FAIL: the control module does not compile"; cat m.log; exit 1; }

while IFS='|' read -r DECL CALL; do
  if emit "$DECL" "$CALL"; then
    echo "CANARY FIRED: llc now accepts $DECL, see the header"; exit 1
  fi
  grep -q 'unable to legalize instruction' m.log ||
    { echo "FAIL: $DECL was rejected for another reason"; cat m.log; exit 1; }
done <<'LIST'
void @llvm.prefetch.p1(ptr addrspace(1), i32, i32, i32)|call void @llvm.prefetch.p1(ptr addrspace(1) %p, i32 0, i32 3, i32 1)
i64 @llvm.readcyclecounter()|%v = call i64 @llvm.readcyclecounter()
i64 @llvm.readsteadycounter()|%v = call i64 @llvm.readsteadycounter()
i32 @llvm.get.rounding()|%v = call i32 @llvm.get.rounding()
LIST
echo PASSED
