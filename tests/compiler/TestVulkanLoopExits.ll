; A loop with two exits (mr-hip's Miller-Rabin rounds) must get one exit
; before StructurizeCFG, which otherwise rewires it wrongly.
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64-unknown-chipstar-shadervulkan"

define spir_kernel void @k(ptr addrspace(1) %o, i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i1, %next ]
  %x = mul i32 %i, %i
  %hit = icmp eq i32 %x, %n
  br i1 %hit, label %found, label %next
next:
  %i1 = add i32 %i, 1
  %done = icmp eq i32 %i1, 100
  br i1 %done, label %notfound, label %loop
found:
  store i32 1, ptr addrspace(1) %o, align 4
  ret void
notfound:
  store i32 0, ptr addrspace(1) %o, align 4
  ret void
}
