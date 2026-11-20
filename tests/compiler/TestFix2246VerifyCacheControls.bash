#!/bin/bash
# HipVerify's llvm-spirv list must accept SPV_INTEL_cache_controls decorations:
# https://github.com/CHIP-SPV/chipStar/issues/2246
set -eu

OPT="@LLVM_TOOLS_BINARY_DIR@/opt"
PLUGIN="@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"

cat > cc.ll <<'IR'
target triple = "spirv64"

define spir_kernel void @k(ptr addrspace(1) %p, ptr addrspace(1) %o) {
  %g = getelementptr inbounds i32, ptr addrspace(1) %p, i64 0, !spirv.Decorations !0
  %v = load volatile i32, ptr addrspace(1) %g
  store i32 %v, ptr addrspace(1) %o
  ret void
}

!0 = !{!1}
!1 = !{i32 6442, i32 0, i32 0}
IR

CHIP_VERIFY_MODE=all "${OPT}" -load-pass-plugin "${PLUGIN}" \
  -passes=hip-post-link-passes cc.ll -o /dev/null 2> cc.txt
cat cc.txt
! grep -q RequiresExtension cc.txt ||
  { echo "FAIL: llvm-spirv rejected the cache control decoration"; exit 1; }
grep -Eq '\| *Post-HIP passes *\| *PASS *\| *PASS *\| *PASS *\|' cc.txt ||
  { echo "FAIL: expected PASS PASS PASS in the Post-HIP passes row"; exit 1; }
echo PASSED
