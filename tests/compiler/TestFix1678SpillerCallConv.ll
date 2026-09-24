; Reproducer for https://github.com/CHIP-SPV/chipStar/issues/1678: a kernel
; whose byval argument is too large for the kernel argument buffer.
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

%struct.Big = type { [1024 x i64] }

define spir_kernel void @k(ptr addrspace(1) %o, ptr byval(%struct.Big) align 8 %b) {
entry:
  %v = load i64, ptr %b, align 8
  store i64 %v, ptr addrspace(1) %o, align 8
  ret void
}
