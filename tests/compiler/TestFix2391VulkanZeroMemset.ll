; Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2391: on Vulkan a zero
; memset covering a whole typed variable must become one typed zero store.
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64-unknown-chipstar-shadervulkan"

@lsums = internal addrspace(3) global { [3 x double] } undef, align 8

define spir_kernel void @k() {
  call void @llvm.memset.p4.i64(ptr addrspace(4) addrspacecast (ptr addrspace(3) @lsums to ptr addrspace(4)), i8 0, i64 24, i1 false)
  ret void
}

declare void @llvm.memset.p4.i64(ptr addrspace(4), i8, i64, i1)
