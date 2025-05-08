; ModuleID = 'func-return-nonstd'
target triple = "spirv64-unknown-unknown"

; Function that returns a non-standard integer type (i24)
define i24 @return_nonstd() {
entry:
  ; Create an i24 value via trunc from i32 257
  %val = trunc i32 257 to i24
  ret i24 %val
}

; Function that calls the non-standard return value function
define i32 @caller() {
entry:
  ; Call the function returning a non-standard type
  %nonstd = call i24 @return_nonstd()
  
  ; Extend to i32 and return
  %ext = zext i24 %nonstd to i32
  ret i32 %ext
}

; Another function demonstrating multiple non-standard integers
define i32 @multi_nonstd(i32 %arg) {
entry:
  ; Create various non-standard types
  %a1 = trunc i32 %arg to i17
  %a2 = trunc i32 %arg to i19
  %a3 = trunc i32 %arg to i23
  
  ; Perform some operations on them
  %b1 = add i17 %a1, %a1
  %b2 = sub i19 %a2, 1
  %b3 = mul i23 %a3, 2
  
  ; Convert back to standard types
  %c1 = zext i17 %b1 to i32
  %c2 = zext i19 %b2 to i32
  %c3 = zext i23 %b3 to i32
  
  ; Combine results
  %r1 = add i32 %c1, %c2
  %r2 = add i32 %r1, %c3
  
  ret i32 %r2
} 