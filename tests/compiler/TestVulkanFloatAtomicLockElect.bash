#!/bin/bash
# A shared double atomic's lock is tried by one elected lane per subgroup.
set -eu
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.out.ll"
"@LLVM_TOOLS_BINARY_DIR@/opt" -load-pass-plugin "@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so" \
  -passes=hip-post-link-passes -S "@CMAKE_CURRENT_SOURCE_DIR@/TestVulkanFloatAtomicLockElect.ll" -o "${OUT}"
grep -q "cmpxchg ptr addrspace(3) @__chip_fatomic_lock" "${OUT}" || { echo "FAIL: no lock"; exit 1; }
grep -q "call i1 @llvm.spv.wave.is.first.lane()" "${OUT}" || { echo "FAIL: lock not elected"; exit 1; }
echo PASSED
