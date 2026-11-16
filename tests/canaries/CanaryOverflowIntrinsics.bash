#!/bin/bash
# Canary for https://github.com/CHIP-SPV/chipStar/issues/1475. MEANT TO FAIL EVENTUALLY.
# Asserts the two gaps HipLowerOverflowIntrinsicsPass works around still exist:
# llvm-spirv rejects llvm.smul.with.overflow (no upstream fix merged;
# https://github.com/KhronosGroup/SPIRV-LLVM-Translator/pull/3990 was closed
# unmerged), and the in-tree backend lowers llvm.umul.with.overflow to
# OpUMulExtended, which IGC fails to link
# (https://github.com/intel/intel-graphics-compiler/issues/432).
# When it fires, drop the matching half of
# llvm_passes/workarounds/HipLowerOverflowIntrinsics.cpp once every supported LLVM and IGC
# carries the fix. Delete the pass and this file once both halves have fired,
# then close #1475.
# Needs no GPU: ocloc is an offline compiler.
set -u
LLVM_AS="@LLVM_TOOLS_BINARY_DIR@/llvm-as"
LLC="@LLVM_TOOLS_BINARY_DIR@/llc"
LLVM_SPIRV="@LLVM_SPIRV@"
cd "$(mktemp -d)"

emit() { # $1 s|u
  printf 'target triple = "spirv64-unknown-unknown"\ndeclare {i64,i1} @llvm.%smul.with.overflow.i64(i64,i64)\ndefine spir_kernel void @k(i64 %%a, i64 %%b, ptr addrspace(1) %%p) {\n  %%r = call {i64,i1} @llvm.%smul.with.overflow.i64(i64 %%a, i64 %%b)\n  %%v = extractvalue {i64,i1} %%r, 0\n  %%o = extractvalue {i64,i1} %%r, 1\n  %%s = select i1 %%o, i64 -1, i64 %%v\n  store i64 %%s, ptr addrspace(1) %%p\n  ret void\n}\n' "$1" "$1" > m.ll
}

PROBED=no
if [ "$LLVM_SPIRV" != NOT_NEEDED ] && [ -x "$LLVM_SPIRV" ]; then
  PROBED=yes
  emit u; { "$LLVM_AS" m.ll -o m.bc && "$LLVM_SPIRV" m.bc -o m.spv; } > m.log 2>&1 ||
    { echo "FAIL: the umul control does not translate"; cat m.log; exit 1; }
  emit s
  if { "$LLVM_AS" m.ll -o m.bc && "$LLVM_SPIRV" m.bc -o m.spv; } > m.log 2>&1; then
    echo "CANARY FIRED: llvm-spirv now translates llvm.smul.with.overflow, see the header"; exit 1
  fi
  grep -q 'Unexpected llvm intrinsic' m.log ||
    { echo "FAIL: smul rejected for another reason"; cat m.log; exit 1; }
fi

if "$LLC" --version | grep -q spirv64; then
  emit u
  "$LLC" -mtriple=spirv64v1.3-unknown-chipstar -filetype=obj m.ll -o m.spv > m.log 2>&1 ||
    { echo "FAIL: the backend does not compile umul"; cat m.log; exit 1; }
  if command -v ocloc >/dev/null 2>&1; then
    PROBED=yes
    if ocloc compile -spirv_input -file m.spv -device bmg -output_no_suffix -output o > o.log 2>&1; then
      echo "CANARY FIRED: IGC now builds the backend's OpUMulExtended, see the header"; exit 1
    fi
    grep -q '_Z20__spirv_UMulExtendedll' o.log ||
      { echo "FAIL: IGC rejected the module for another reason"; cat o.log; exit 1; }
  else
    echo "note: ocloc not found, IGC half not probed"
  fi
fi
[ "$PROBED" = yes ] || { echo "HIP_SKIP_THIS_TEST: no SPIR-V producer"; exit 0; }
echo PASSED
