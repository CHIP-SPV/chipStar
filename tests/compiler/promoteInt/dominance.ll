target triple = "spirv64-unknown-unknown"

; Test for proper dominance-based handling of non-standard integer value types
define i32 @dominance_test(i32 %arg) {
entry:
  %cmp = icmp sgt i32 %arg, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  ; Create a non-standard integer type (i20)
  %nonstandard = trunc i32 %arg to i20
  br label %loop.header

if.else:
  ; Create a different non-standard type value
  %other = trunc i32 %arg to i20
  br label %loop.header

loop.header:
  ; PHI node that combines two non-standard integer values
  %phi = phi i20 [ %nonstandard, %if.then ], [ %other, %if.else ], [ %next, %loop.body ]
  %i = phi i32 [ 0, %if.then ], [ 0, %if.else ], [ %i.next, %loop.body ]
  %cond = icmp slt i32 %i, 10
  br i1 %cond, label %loop.body, label %loop.exit

loop.body:
  ; Modify the non-standard integer
  %inc = add i20 %phi, 1
  %next = mul i20 %inc, 2  
  %i.next = add i32 %i, 1
  br label %loop.header

loop.exit:
  ; Use the final result
  %result = zext i20 %phi to i32
  ret i32 %result
} 