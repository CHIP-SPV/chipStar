#!/bin/bash
# Canary for HipCanonicalizeGEPPass, https://github.com/CHIP-SPV/chipStar/issues/1360. MEANT TO FAIL EVENTUALLY.
# Asserts IGC still miscompiles a private-array load through a single-index
# getelementptr over [4 x i8], the shape LLVM 23's InstCombine emits for every
# typed GEP: IGC's SoA promotion over-strides it
# (https://github.com/intel/intel-graphics-compiler/issues/429, fix proposed in
# https://github.com/intel/intel-graphics-compiler/pull/430) and here deletes
# the kernel's only global store, while the byte GEP the pass emits keeps it.
# IGC 2.40.0 sidesteps it (d73553dfd19d). When it fires on every IGC chipStar
# supports, delete llvm_passes/workarounds/HipCanonicalizeGEP.*, its registration in
# llvm_passes/HipPasses.cpp and llvm_passes/CMakeLists.txt and this file, then
# close #1360.
# Needs no GPU: ocloc is an offline compiler.
set -u
LLVM_AS="@LLVM_TOOLS_BINARY_DIR@/llvm-as"
LLVM_SPIRV="@LLVM_SPIRV@"
# Translator lane only: this reduced module compiles correctly from llc output.
[ "$LLVM_SPIRV" != NOT_NEEDED ] && [ -x "$LLVM_SPIRV" ] || { echo "HIP_SKIP_THIS_TEST: no llvm-spirv"; exit 0; }
command -v ocloc >/dev/null 2>&1 || { echo "HIP_SKIP_THIS_TEST: ocloc not found"; exit 0; }
cd "$(mktemp -d)"

stores() { # $1 the two lines computing %r from %j; prints the stores to %o IGC kept
  printf '%s\n' 'target triple = "spirv64-unknown-unknown"' \
    'define spir_kernel void @k(ptr addrspace(1) %o, i64 %j) {' \
    '  %a = alloca [8 x float], align 4' '  br label %fill' 'fill:' \
    '  %n = phi i64 [0, %0], [%n1, %fill]' \
    '  %e = getelementptr inbounds [4 x i8], ptr %a, i64 %n' \
    '  %f = sitofp i64 %n to float' '  store float %f, ptr %e, align 4' \
    '  %n1 = add i64 %n, 1' '  %c = icmp ult i64 %n1, 8' \
    '  br i1 %c, label %fill, label %done' 'done:' "$1" \
    '  %v = load float, ptr %r, align 4' \
    '  store float %v, ptr addrspace(1) %o, align 4' '  ret void' '}' > m.ll
  { "$LLVM_AS" m.ll -o m.bc && "$LLVM_SPIRV" m.bc -o m.spv; } > m.log 2>&1 ||
    { echo "FAIL: no SPIR-V for the module"; cat m.log >&2; return 1; }
  rm -rf d && mkdir d
  ( cd d && IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir="$PWD" \
      ocloc compile -spirv_input -file ../m.spv -device bmg ) > ocloc.log 2>&1 ||
    { echo "FAIL: ocloc did not build the module"; cat ocloc.log >&2; return 1; }
  compgen -G 'd/*_codegen.ll' > /dev/null || { echo "FAIL: no codegen dump for bmg" >&2; return 1; }
  cat d/*_codegen.ll | grep -c 'store float %[^,]*, ptr addrspace(1) %o' || true
}

N=$(stores '  %s = shl i64 %j, 2
  %r = getelementptr inbounds i8, ptr %a, i64 %s') || { echo "$N"; exit 1; }
[ "$N" = 1 ] || { echo "FAIL: the byte-GEP control lost its store ($N)"; exit 1; }
N=$(stores '  %r = getelementptr inbounds [4 x i8], ptr %a, i64 %j') || { echo "$N"; exit 1; }
if [ "$N" != 0 ]; then
  echo "CANARY FIRED: IGC keeps the store through a [4 x i8] GEP, see the header"; exit 1
fi
echo "canary: IGC still drops the store, the workaround is still needed"
echo PASSED
