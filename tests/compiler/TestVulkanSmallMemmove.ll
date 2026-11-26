; A small constant memmove inside one private variable (deredundancy-hip)
; must not compare pointers to pick a direction: Vulkan cannot.
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64-unknown-chipstar-shadervulkan"

define spir_kernel void @k(ptr addrspace(1) %o, i32 %v) {
  %a = alloca i32, align 4
  store i32 %v, ptr %a, align 4
  %src = getelementptr i8, ptr %a, i64 1
  call void @llvm.memmove.p0.p0.i64(ptr align 4 %a, ptr align 1 %src, i64 3, i1 false)
  %r = load i32, ptr %a, align 4
  store i32 %r, ptr addrspace(1) %o, align 4
  ret void
}

declare void @llvm.memmove.p0.p0.i64(ptr, ptr, i64, i1)
