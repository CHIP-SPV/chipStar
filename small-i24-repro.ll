; ModuleID = 'i24_minimal'
target triple = "spir64-unknown-unknown"

define i32 @foo() {
entry:
  ; Create an i24 value via trunc from i32 257
  %val = trunc i32 257 to i24

  ; Conditionally branch to exit or to 'somewhere'
  br i1 true, label %exit, label %somewhere

somewhere:
  ; Just fall through to exit
  br label %exit

exit:
  ; PHI node that merges i24 values from two blocks
  ; - from 'entry' we use %val
  ; - from 'somewhere' we use a constant 0 of type i24
  %phi = phi i24 [ %val, %entry ], [ 0, %somewhere ]

  ; Use the PHI result, e.g. zext it to i32
  %ext = zext i24 %phi to i32

  ; Return the i32
  ret i32 %ext
}