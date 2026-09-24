#!/bin/bash
# Canary for https://github.com/CHIP-SPV/chipStar/issues/1680. MEANT TO FAIL EVENTUALLY.
# Asserts the configured llvm-spirv still lacks
# https://github.com/KhronosGroup/SPIRV-LLVM-Translator/pull/3866. When it fires,
# delete llvm_passes/HipCoalesceDuplicatePhiPreds.*, its registration in
# llvm_passes/HipPasses.cpp and llvm_passes/CMakeLists.txt, and this test
# together, then close #1680 naming the translator release that shipped #3866.
# #3866 also stops emitting OpBranchConditional with equal targets; that half is
# probed as well because an llvm-spirv built with chipStar's old translator
# patch coalesces the phi without having #3866.
set -u
LLVM_AS="@LLVM_TOOLS_BINARY_DIR@/llvm-as"
LLVM_SPIRV="@LLVM_SPIRV@"
cd "$(mktemp -d)"

emit() { # $1 name, $2 function body
  printf 'target triple = "spirv64-unknown-unknown"\ndefine spir_func i32 @f(i32 %%s, i1 %%c) {\n%s\n}\n' "$2" > "$1.ll"
  { "$LLVM_AS" "$1.ll" -o "$1.bc" && "$LLVM_SPIRV" --spirv-text "$1.bc" -o "$1.spt"; } > "$1.log" 2>&1
}

emit phi 'entry:
  switch i32 %s, label %other [ i32 0, label %join
                                i32 4, label %join ]
other:
  br label %join
join:
  %v = phi i32 [ 7, %entry ], [ 7, %entry ], [ 9, %other ]
  ret i32 %v' || { echo "FAIL: the phi module does not translate"; cat phi.log; exit 1; }
PHI_FIXED=yes
awk '$2 == "Phi" { for (i = 6; i <= NF; i += 2) if (seen[$i]++) d = 1 } END { exit !d }' phi.spt && PHI_FIXED=no

BR_FIXED=no
if emit br 'entry:
  br i1 %c, label %l, label %l
l:
  ret i32 0'; then
  grep -q BranchConditional br.spt || BR_FIXED=yes
else
  grep -q 'TrueLabelId != FalseLabelId' br.log ||
    { echo "FAIL: the branch module was rejected for another reason"; cat br.log; exit 1; }
fi

echo "OpPhi coalesced: $PHI_FIXED, equal-target branch folded: $BR_FIXED"
if [ "$PHI_FIXED" = yes ] && [ "$BR_FIXED" = yes ]; then
  echo "CANARY FIRED: llvm-spirv now carries #3866, see the header"; exit 1
fi
echo PASSED
