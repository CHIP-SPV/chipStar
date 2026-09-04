; Companion to TestFix1577PointerVectors.ll for CHIP-SPV/chipStar#1577.
;
; Here the pointer vector is not merely copied, it is taken apart with
; extractelement, so the load/store rewrite cannot remove it. SPIR-V still
; cannot represent the value, and hip-lower-pointer-vectors is expected to say
; so by name rather than let SPIR-V emission abort on an assertion.

target triple = "spirv64"

define spir_kernel void @use_ptr_vec(ptr %src, ptr %out) {
entry:
  %v = load <3 x ptr addrspace(4)>, ptr %src, align 8
  %p = extractelement <3 x ptr addrspace(4)> %v, i32 0
  %x = load i32, ptr addrspace(4) %p, align 4
  store i32 %x, ptr %out, align 4
  ret void
}
