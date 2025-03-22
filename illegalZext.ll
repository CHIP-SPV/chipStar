; Simplified reproducer for HipPromoteInts pass issue
define i32 @test_function(i32 %arg) {
entry:
  %cmp = icmp sgt i32 %arg, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  ; Create a non-standard integer type (i24)
  %non_std = trunc i32 %arg to i24
  ; Convert to i8 to match the PHI type
  %trunc1 = trunc i24 %non_std to i8
  br label %merge

if.else:
  ; Create a boolean value
  %bool_val = icmp eq i32 %arg, 42
  ; Convert boolean to i8
  %frombool = zext i1 %bool_val to i8
  br label %merge

merge:
  ; PHI node combining values of the same type but one comes from non-standard type
  %result = phi i8 [ %trunc1, %if.then ], [ %frombool, %if.else ]
  
  ; Use the phi result
  %final = zext i8 %result to i32
  ret i32 %final
}