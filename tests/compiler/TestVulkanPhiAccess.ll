; A load and a store sunk behind phis of two shared arrays and of two offsets
; into one buffer (lud-hip) must move back into the branches: logical SPIR-V
; cannot pick a variable through a phi.
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64-unknown-chipstar-shadervulkan"

@a = internal addrspace(3) global [16 x float] undef, align 4
@b = internal addrspace(3) global [16 x float] undef, align 4

define spir_kernel void @k(i1 %c, i64 %i, ptr addrspace(1) %o) {
entry:
  br i1 %c, label %l, label %r
l:
  %pa = getelementptr [4 x i8], ptr addrspace(3) @a, i64 %i
  br label %m
r:
  %pb = getelementptr [4 x i8], ptr addrspace(3) @b, i64 %i
  %ob = getelementptr i8, ptr addrspace(1) %o, i64 4
  br label %m
m:
  %p = phi ptr addrspace(3) [ %pa, %l ], [ %pb, %r ]
  %q = phi ptr addrspace(1) [ %o, %l ], [ %ob, %r ]
  %g = getelementptr i8, ptr addrspace(3) %p, i64 8
  %v = load float, ptr addrspace(3) %g, align 4
  store float %v, ptr addrspace(1) %q, align 4
  ret void
}
