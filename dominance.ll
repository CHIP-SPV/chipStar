; File: i33_reproducer.ll
; This function has a PHI and arithmetic in i33, which is not a standard bitwidth.
target triple = "spirv64-unknown-unknown"

define i64 @testphi(i64 %a, i64 %b, i1 %cond) {
entry:
  ; Branch on %cond
  br i1 %cond, label %then, label %else

then:
  ; Fall through
  %a1 = trunc i64 %a to i33
  br label %merge

else:
  ; Fall through
  %b1 = trunc i64 %b to i33
  br label %merge

merge:
  ; PHI node that merges i33 from two preds
  %phi = phi i33 [ %a1, %then ], [ %b1, %else ]


  ; A simple add in i33
  %phi1 = zext i33 %phi to i64
  %res = add i64 %phi1, 13

  ret i64 %res
}