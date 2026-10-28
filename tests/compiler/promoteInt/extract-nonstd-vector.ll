; Reproduces CHIP-SPV/chipStar#2126: extractelement from a non-standard int vector was left as an i20 extract.
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv64-unknown-unknown"

define spir_kernel void @extract_const_vec(ptr addrspace(1) %o) {
entry:
  %v = load i32, ptr addrspace(1) %o, align 4
  %e = extractelement <2 x i20> <i20 -1, i20 -1>, i32 %v
  %z = zext i20 %e to i32
  store i32 %z, ptr addrspace(1) %o, align 4
  ret void
}

define spir_kernel void @extract_const_vec_sext(ptr addrspace(1) %i, ptr addrspace(1) %o) {
entry:
  %idx = load i32, ptr addrspace(1) %i, align 4
  %e = extractelement <2 x i20> <i20 -1, i20 -1>, i32 %idx
  %s = sext i20 %e to i32
  store i32 %s, ptr addrspace(1) %o, align 4
  ret void
}
