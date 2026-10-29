#!/bin/bash
# HipVerify must validate with the producer the backend lane uses (clang -cc1
# with the driver's -spirv-ext list): https://github.com/CHIP-SPV/chipStar/issues/2100
set -eu

OPT="@LLVM_TOOLS_BINARY_DIR@/opt"
PLUGIN="@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"

verify() {
  CHIP_VERIFY_MODE=all "${OPT}" -load-pass-plugin "${PLUGIN}" \
    -passes=hip-post-link-passes "$1" -o /dev/null 2> "$2"
}

cat > ok.ll <<'IR'
target triple = "spirv64"

define spir_kernel void @k(ptr addrspace(1) %p) {
  store i32 1, ptr addrspace(1) %p
  ret void
}
IR

verify ok.ll ok.txt
grep "Post-HIP passes" ok.txt || true
grep -Eq '\| *Post-HIP passes *\| *PASS *\| *PASS *\| *PASS *\|' ok.txt ||
  { echo "FAIL: expected PASS PASS PASS in the Post-HIP passes row"; exit 1; }
grep -q '^clang: ' ok.txt && ! grep -q '^llvm-spirv: ' ok.txt ||
  { echo "FAIL: tool paths should list clang, not llvm-spirv"; exit 1; }

# The expectation mirrors what clang -cc1 does with this extension list.
BACKEND_EXT="@CHIP_HIPVERIFY_BACKEND_EXT@"
cat > cc.ll <<'IR'
target triple = "spirv64v1.3-unknown-chipstar"

define spir_kernel void @k(ptr addrspace(1) %p, ptr addrspace(1) %o) {
  %g = getelementptr inbounds i32, ptr addrspace(1) %p, i64 0, !spirv.Decorations !0
  %v = load volatile i32, ptr addrspace(1) %g
  store i32 %v, ptr addrspace(1) %o
  ret void
}

!0 = !{!1}
!1 = !{i32 6442, i32 0, i32 0}
IR

verify cc.ll cc.txt
grep "Post-HIP passes" cc.txt || true
case "${BACKEND_EXT}" in
  *SPV_INTEL_cache_controls*) WANT='PASS *\| *PASS *\| *PASS' ;;
  *) WANT='PASS *\| *clang failed' ;;
esac
grep -Eq "\\| *Post-HIP passes *\\| *${WANT}" cc.txt ||
  { echo "FAIL: unexpected backend verdict for the cache-control kernel"; exit 1; }

# Before the inliner runs, clang's -O0 pipeline still inlines this always-inline
# helper and drops the pointer-vector load; a bare llc rejects the module.
cat > inl.ll <<'IR'
target triple = "spirv64v1.3-unknown-chipstar"

define internal void @h(i1 %c, ptr addrspace(1) %p, ptr addrspace(1) %o) alwaysinline {
  br i1 %c, label %t, label %e
t:
  %v = load <2 x ptr addrspace(1)>, ptr addrspace(1) %p
  %x = extractelement <2 x ptr addrspace(1)> %v, i32 0
  store ptr addrspace(1) %x, ptr addrspace(1) %o
  br label %e
e:
  ret void
}

define spir_kernel void @k(ptr addrspace(1) %p, ptr addrspace(1) %o) {
  call void @h(i1 false, ptr addrspace(1) %p, ptr addrspace(1) %o)
  ret void
}
IR

verify inl.ll inl.txt
grep "HipSanityChecksPass" inl.txt || true
grep -Eq '\| *HipSanityChecksPass *\| *PASS *\| *PASS *\| *PASS' inl.txt ||
  { echo "FAIL: pre-inliner checkpoint rejected a module clang accepts"; exit 1; }
echo PASSED
