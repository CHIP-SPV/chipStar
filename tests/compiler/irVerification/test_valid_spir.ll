; Simplest valid LLVM IR for SPIR-V

target triple = "spirv64-unknown-unknown"

; Minimal kernel function
define spir_kernel void @simple_kernel(ptr addrspace(1) %output) {
entry:
  ; Simple operation: store a constant value
  store i32 42, ptr addrspace(1) %output, align 4
  ret void
}

; A device function with proper SPIR calling convention
define spir_func i32 @valid_device_function(i32 %input, ptr addrspace(2) %constant_data) {
entry:
  ; Load from constant memory (address space 2)
  %const_val = load i32, ptr addrspace(2) %constant_data, align 4
  %result = add i32 %input, %const_val
  ret i32 %result
}

; Built-in function declarations (these are OK as declarations)
declare i32 @llvm.spirv.GlobalInvocationId.x() #0
declare i32 @llvm.spirv.LocalInvocationId.x() #0

attributes #0 = { nounwind readnone } 