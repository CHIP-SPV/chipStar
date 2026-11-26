; A shared double atomic add becomes a work-group lock that only an elected
; lane of a subgroup tries: lanes of one subgroup contending for it hang the
; B570 (aop-hip).
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64-unknown-chipstar-shadervulkan"

@s = internal addrspace(3) global double undef, align 8

define spir_kernel void @k(double %v) {
  %old = atomicrmw fadd ptr addrspace(3) @s, double %v monotonic, align 8
  ret void
}
