#!/bin/bash
# Canary for https://github.com/CHIP-SPV/chipStar/issues/1633. MEANT TO FAIL EVENTUALLY.
# Asserts llvm-spirv still rejects each intrinsic HipLowerHintIntrinsicsPass lowers.
# When it fires for one, delete that case from the pass only once llc
# -mtriple=spirv64 on every supported LLVM accepts it too. memcpy.inline and
# objectsize should fire when
# https://github.com/KhronosGroup/SPIRV-LLVM-Translator/pull/3990 lands.
# prefetch needs more: https://github.com/llvm/llvm-project/pull/215505 emits an
# invalid prefetch on non-global pointers, and HIP kernel pointers reach it generic.
set -u
LLVM_AS="@LLVM_TOOLS_BINARY_DIR@/llvm-as"
LLVM_SPIRV="@LLVM_SPIRV@"
cd "$(mktemp -d)"

emit() { # $1 declaration, $2 call
  printf 'target triple = "spirv64-unknown-unknown"\ndeclare %s\ndefine spir_func void @f(ptr addrspace(4) %%p) {\n  %s\n  ret void\n}\n' "$1" "$2" > m.ll
  { "$LLVM_AS" m.ll -o m.bc && "$LLVM_SPIRV" m.bc -o m.spv; } > m.log 2>&1
}

emit 'void @llvm.memcpy.p4.p4.i64(ptr addrspace(4), ptr addrspace(4), i64, i1)' \
  'call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) %p, ptr addrspace(4) %p, i64 1, i1 false)' ||
  { echo "FAIL: the control module does not translate"; cat m.log; exit 1; }

while IFS='|' read -r DECL CALL; do
  if emit "$DECL" "$CALL"; then
    echo "CANARY FIRED: llvm-spirv now accepts $DECL, see the header"; exit 1
  fi
  grep -q 'Unexpected llvm intrinsic' m.log ||
    { echo "FAIL: $DECL was rejected for another reason"; cat m.log; exit 1; }
done <<'EOF'
void @llvm.prefetch.p4(ptr addrspace(4), i32, i32, i32)|call void @llvm.prefetch.p4(ptr addrspace(4) %p, i32 0, i32 3, i32 1)
i64 @llvm.readcyclecounter()|%v = call i64 @llvm.readcyclecounter()
i64 @llvm.readsteadycounter()|%v = call i64 @llvm.readsteadycounter()
i32 @llvm.get.rounding()|%v = call i32 @llvm.get.rounding()
i1 @llvm.allow.runtime.check(metadata)|%v = call i1 @llvm.allow.runtime.check(metadata !"x")
ptr @llvm.returnaddress(i32)|%v = call ptr @llvm.returnaddress(i32 0)
ptr @llvm.frameaddress.p0(i32)|%v = call ptr @llvm.frameaddress.p0(i32 0)
i64 @llvm.objectsize.i64.p4(ptr addrspace(4), i1, i1, i1)|%v = call i64 @llvm.objectsize.i64.p4(ptr addrspace(4) %p, i1 false, i1 true, i1 false)
void @llvm.memcpy.inline.p4.p4.i64(ptr addrspace(4), ptr addrspace(4), i64, i1)|call void @llvm.memcpy.inline.p4.p4.i64(ptr addrspace(4) %p, ptr addrspace(4) %p, i64 1, i1 false)
EOF
echo PASSED
