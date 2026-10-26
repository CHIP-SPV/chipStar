; Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2119: HipPromoteInts segfaulted on a cycle of i20 PHIs.
target triple = "spirv64-unknown-unknown"

define spir_kernel void @k(ptr addrspace(1) %out, i32 %a, i1 %c) {
entry:
  %t = trunc i32 %a to i20
  br label %loop
loop:
  %R = phi i20 [ %t, %entry ], [ %P, %loop ]
  %Q = phi i20 [ %t, %entry ], [ %R, %loop ]
  %P = phi i20 [ %t, %entry ], [ %Q, %loop ]
  %z = zext i20 %P to i32
  store i32 %z, ptr addrspace(1) %out
  br i1 %c, label %loop, label %exit
exit:
  ret void
}
