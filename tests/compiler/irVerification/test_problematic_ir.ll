; Test LLVM IR file with constructs that will cause verification errors

target triple = "spirv64-unknown-unknown"

; Function with mismatched return type (declares i32 but returns void)
define i32 @mismatched_return_type(i32 %input) {
entry:
  %result = add i32 %input, 42
  ret void  ; ERROR: returning void when function should return i32
}

; Function with undefined value usage
define void @undefined_value_usage(ptr addrspace(1) %output) {
entry:
  ; Using %undefined_val without defining it first
  %result = add i32 %undefined_val, 42  ; ERROR: undefined value
  store i32 %result, ptr addrspace(1) %output, align 4
  ret void
}

; Function with type mismatch in instruction
define void @type_mismatch(ptr addrspace(1) %output) {
entry:
  %int_val = add i32 1, 2
  ; Trying to use integer as pointer
  store i32 42, ptr %int_val, align 4  ; ERROR: type mismatch
  ret void
}

; Function with invalid phi node (phi with no predecessors)
define void @invalid_phi() {
entry:
  br label %loop

loop:
  ; This phi node references a block that doesn't exist
  %bad_phi = phi i32 [ 0, %nonexistent_block ], [ 1, %entry ]  ; ERROR: invalid phi
  br label %loop
}

; Function with branch to non-existent label
define void @bad_branch() {
entry:
  br label %nonexistent_label  ; ERROR: undefined label
}

; Function with call to non-existent function
define void @call_nonexistent() {
entry:
  call void @this_function_does_not_exist()  ; ERROR: undefined function
  ret void
} 