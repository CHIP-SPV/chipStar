; ModuleID = 'phi-constant.bc'
source_filename = "phi-constant.cu"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv64-unknown-unknown"

; A non-standard-width PHI with only constant incoming values must promote: https://github.com/CHIP-SPV/chipStar/issues/2118
define spir_kernel void @k(ptr addrspace(1) %out, i1 %c) {
entry:
  br i1 %c, label %l, label %r
l:
  br label %m
r:
  br label %m
m:
  %p = phi i20 [ 0, %l ], [ 1, %r ]
  %z = zext i20 %p to i32
  store i32 %z, ptr addrspace(1) %out
  ret void
}
