; Reproducer for https://github.com/CHIP-SPV/chipStar/issues/1650: globals that
; are not lowered but hold a lowered device global's address.
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

@Scalar = hidden addrspace(1) externally_initialized global i32 42, align 4
@_ZZ4getPvE1P = internal addrspace(1) global ptr addrspace(4) addrspacecast (ptr addrspace(1) @Scalar to ptr addrspace(4)), align 8
@__const._Z1kPi.p = private unnamed_addr addrspace(1) constant ptr addrspace(4) addrspacecast (ptr addrspace(1) @Scalar to ptr addrspace(4)), align 8

define hidden spir_kernel void @_Z1kPi(ptr addrspace(1) %Out) {
entry:
  %0 = load ptr addrspace(4), ptr addrspace(1) @_ZZ4getPvE1P, align 8
  %1 = load ptr addrspace(4), ptr addrspace(1) @__const._Z1kPi.p, align 8
  store ptr addrspace(4) %0, ptr addrspace(1) %Out, align 8
  %2 = getelementptr inbounds ptr addrspace(4), ptr addrspace(1) %Out, i64 1
  store ptr addrspace(4) %1, ptr addrspace(1) %2, align 8
  ret void
}
