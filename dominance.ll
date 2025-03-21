; File: i33_reproducer.ll
; This function has a PHI and arithmetic in i33, which is not a standard bitwidth.

define i33 @testphi(i33 %a, i33 %b, i1 %cond) {
entry:
  ; Branch on %cond
  br i1 %cond, label %then, label %else

then:
  ; Fall through
  br label %merge

else:
  ; Fall through
  br label %merge

merge:
  ; PHI node that merges i33 from two preds
  %phi = phi i33 [ %a, %then ], [ %b, %else ]

  ; A simple add in i33
  %res = add i33 %phi, 13

  ; Return i33
  ret i33 %res
}