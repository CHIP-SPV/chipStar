; A load through a select of two private variables, each only partly written
; (warpsort-hip), must become a select of loads: logical SPIR-V cannot select
; between variables.
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64-unknown-chipstar-shadervulkan"

define spir_kernel void @k(i1 %c, ptr addrspace(1) %o) {
  %a = alloca i64, align 8
  %b = alloca i64, align 8
  store i32 0, ptr %a, align 8
  %b4 = getelementptr i8, ptr %b, i64 4
  store i32 0, ptr %b4, align 4
  %p = select i1 %c, ptr %a, ptr %b
  %v = load i64, ptr %p, align 4
  store i64 %v, ptr addrspace(1) %o, align 8
  ret void
}
