#!/bin/bash
# Canary for https://github.com/CHIP-SPV/chipStar/issues/1958. MEANT TO FAIL EVENTUALLY.
# Asserts the in-tree SPIR-V backend still emits an overlapping llvm.memmove as
# OpCopyMemorySized, which SPIR-V consumers read back as memcpy (no upstream
# LLVM issue filed yet). When it fails, delete HipLowerMemmovePass in
# llvm_passes/HipPasses.cpp, its registration and this canary together, then
# close #1958.
set -u
cd "$(mktemp -d)"
printf '%s\n' 'define spir_func void @f(ptr %p) {' \
  '  %d = getelementptr inbounds i8, ptr %p, i64 4' \
  '  call void @llvm.memmove.p0.p0.i64(ptr %d, ptr %p, i64 12, i1 false)' \
  '  ret void' '}' > m.ll
"@LLVM_TOOLS_BINARY_DIR@/llc" -mtriple=spirv64v1.3-unknown-chipstar m.ll -o m.spt > m.log 2>&1 ||
  { echo "FAIL: the module does not compile"; cat m.log; exit 1; }
cat m.spt
# Bad shape: all 12 bytes copied from %p straight to p+4
# (%7 = OpConstant %6 12; %11 = OpInBoundsPtrAccessChain %3 %p %8; OpCopyMemorySized %11 %p %7).
if awk '$3 == "OpConstant" && $NF == "12" { twelve[$1] = 1 }
        $3 ~ /PtrAccessChain$/ { base[$1] = $5 }
        $1 == "OpCopyMemorySized" && base[$2] == $3 && twelve[$4] { bad = 1 } END { exit !bad }' m.spt; then
  echo PASSED
else
  echo "CANARY FIRED: the SPIR-V backend no longer emits an overlapping memmove as one OpCopyMemorySized, see the header"; exit 1
fi
