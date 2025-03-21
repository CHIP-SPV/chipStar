; Minimal reproducer for HipPromoteIntsPass bug
; This demonstrates the issue with promoting non-standard integer types (i33)
; when boolean values (i1) are involved in the instruction chain

define i33 @test_reproducer(i33 %a) {
entry:
  ; Create a boolean value via comparison
  %cmp = icmp ne i33 %a, 0
  
  ; Convert the i1 to i33 - this is a non-standard integer type
  %frombool = zext i1 %cmp to i33
  
  ; Create a phi node that will involve both the i1 and i33 types
  br i1 %cmp, label %if.then, label %if.else
  
if.then:
  ; Use the non-standard integer type (i33)
  br label %if.end
  
if.else:
  ; Create another path with different value
  br label %if.end
  
if.end:
  ; PHI node will receive values of different types after promotion
  %result = phi i33 [ %frombool, %if.then ], [ 0, %if.else ]
  ret i33 %result
}

; To trigger this bug:
; 1. The HipPromoteIntsPass will detect the non-standard i33 type and attempt to promote it to i64
; 2. When processing the PHI node, it will encounter mixed types due to the promotion chain
; 3. It will attempt to create a zero extension from i64 to i8 in some code paths
; 4. This will trigger the assertion failure because you can't zext from larger to smaller type 