target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv64-unknown-unknown"

; EXPECT-FOLDED: store i32 0, ptr addrspace(1) %zout
; EXPECT-FOLDED: store i32 -524288, ptr addrspace(1) %sout
; EXPECT-FOLDED: store i32 0, ptr addrspace(1) %czout
; EXPECT-FOLDED: store i32 -524288, ptr addrspace(1) %csout
; EXPECT-FOLDED: store i32 1048575, ptr addrspace(1) %ezout
; EXPECT-FOLDED: store i32 -1, ptr addrspace(1) %esout

@in_z = addrspace(1) constant i32 524288
@in_s = addrspace(1) constant i32 262144

; Expected stored value: 0 (0x80000 + 0x80000 wraps to 0 in i20).
define spir_kernel void @zext_wrap(ptr addrspace(1) %zout) {
  %a = load i32, ptr addrspace(1) @in_z
  %t = trunc i32 %a to i20
  %s = add i20 %t, %t
  %z = zext i20 %s to i32
  store i32 %z, ptr addrspace(1) %zout
  ret void
}

; Expected stored value: -524288 (0x80000 is negative in i20).
define spir_kernel void @sext_wrap(ptr addrspace(1) %sout) {
  %a = load i32, ptr addrspace(1) @in_s
  %t = trunc i32 %a to i20
  %s = add i20 %t, %t
  %z = sext i20 %s to i32
  store i32 %z, ptr addrspace(1) %sout
  ret void
}

; Expected stored value: 0 (constant operands fold before zext).
define spir_kernel void @const_zext_wrap(ptr addrspace(1) %czout) {
  %s = add i20 524288, 524288
  %z = zext i20 %s to i32
  store i32 %z, ptr addrspace(1) %czout
  ret void
}

; Expected stored value: -524288 (constant operands fold before sext).
define spir_kernel void @const_sext_wrap(ptr addrspace(1) %csout) {
  %s = add i20 262144, 262144
  %z = sext i20 %s to i32
  store i32 %z, ptr addrspace(1) %csout
  ret void
}

; Expected stored value: 1048575 (extractelement of a constant folds to i20 -1).
define spir_kernel void @ee_zext_wrap(ptr addrspace(1) %ezout) {
  %e = extractelement <2 x i20> <i20 -1, i20 0>, i32 0
  %z = zext i20 %e to i32
  store i32 %z, ptr addrspace(1) %ezout
  ret void
}

; Expected stored value: -1.
define spir_kernel void @ee_sext_wrap(ptr addrspace(1) %esout) {
  %e = extractelement <2 x i20> <i20 -1, i20 0>, i32 0
  %z = sext i20 %e to i32
  store i32 %z, ptr addrspace(1) %esout
  ret void
}
