; Every 8 and 16 bit atomic form clang can emit for a HIP device module, in the
; global, local and generic address spaces. OpenCL SPIR-V consumers only
; implement 32 and 64 bit atomics, so HipLowerSubwordAtomicsPass must rewrite
; all of these onto the aligned containing 32 bit word before SPIR-V emission.
;
; See CHIP-SPV/chipStar#1497.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv64"

@lmem = internal addrspace(3) global [64 x i8] undef, align 4

define spir_kernel void @subword_atomics(ptr addrspace(1) %P8,
                                         ptr addrspace(1) %P16,
                                         ptr addrspace(1) %PH,
                                         ptr addrspace(1) %P32,
                                         ptr addrspace(1) %Out8,
                                         ptr addrspace(1) %Out16,
                                         i8 %V8, i16 %V16, half %VH, i32 %V32) {
entry:
  %G8 = addrspacecast ptr addrspace(1) %P8 to ptr addrspace(4)
  %L8 = getelementptr inbounds [64 x i8], ptr addrspace(3) @lmem, i64 0, i64 5

  ; load / store, every ordering clang emits for them
  %ld.mono = load atomic i8, ptr addrspace(1) %P8 syncscope("device") monotonic, align 1
  %ld.acq = load atomic i8, ptr addrspace(4) %G8 syncscope("device") acquire, align 1
  %ld.seq = load atomic i16, ptr addrspace(1) %P16 seq_cst, align 2
  %ld.local = load atomic i8, ptr addrspace(3) %L8 syncscope("workgroup") monotonic, align 1
  store atomic i8 %V8, ptr addrspace(1) %P8 syncscope("device") monotonic, align 1
  store atomic i8 %V8, ptr addrspace(4) %G8 syncscope("device") release, align 1
  store atomic i16 %V16, ptr addrspace(1) %P16 seq_cst, align 2
  store atomic i8 %V8, ptr addrspace(3) %L8 syncscope("workgroup") monotonic, align 1

  ; atomicrmw, every integer operation, on i8
  %xchg = atomicrmw xchg ptr addrspace(1) %P8, i8 %V8 syncscope("device") monotonic, align 1
  %add = atomicrmw add ptr addrspace(4) %G8, i8 %V8 syncscope("device") monotonic, align 1
  %sub = atomicrmw sub ptr addrspace(1) %P8, i8 %V8 syncscope("device") acq_rel, align 1
  %and = atomicrmw and ptr addrspace(1) %P8, i8 %V8 syncscope("device") monotonic, align 1
  %or = atomicrmw or ptr addrspace(1) %P8, i8 %V8 syncscope("device") monotonic, align 1
  %xor = atomicrmw xor ptr addrspace(3) %L8, i8 %V8 syncscope("workgroup") monotonic, align 1
  %min = atomicrmw min ptr addrspace(1) %P8, i8 %V8 syncscope("device") monotonic, align 1
  %max = atomicrmw max ptr addrspace(1) %P8, i8 %V8 syncscope("device") monotonic, align 1
  %umin = atomicrmw umin ptr addrspace(1) %P8, i8 %V8 syncscope("device") monotonic, align 1
  %umax = atomicrmw umax ptr addrspace(1) %P8, i8 %V8 syncscope("device") monotonic, align 1
  %nand = atomicrmw nand ptr addrspace(1) %P8, i8 %V8 syncscope("device") monotonic, align 1

  ; the same on i16, plus a 16 bit float add
  %xchg16 = atomicrmw xchg ptr addrspace(1) %P16, i16 %V16 syncscope("device") monotonic, align 2
  %add16 = atomicrmw add ptr addrspace(1) %P16, i16 %V16 syncscope("device") seq_cst, align 2
  %sub16 = atomicrmw sub ptr addrspace(1) %P16, i16 %V16 syncscope("device") monotonic, align 2
  %and16 = atomicrmw and ptr addrspace(1) %P16, i16 %V16 syncscope("device") monotonic, align 2
  %or16 = atomicrmw or ptr addrspace(1) %P16, i16 %V16 syncscope("device") monotonic, align 2
  %xor16 = atomicrmw xor ptr addrspace(1) %P16, i16 %V16 syncscope("device") monotonic, align 2
  %min16 = atomicrmw min ptr addrspace(1) %P16, i16 %V16 syncscope("device") monotonic, align 2
  %max16 = atomicrmw max ptr addrspace(1) %P16, i16 %V16 syncscope("device") monotonic, align 2
  %umin16 = atomicrmw umin ptr addrspace(1) %P16, i16 %V16 syncscope("device") monotonic, align 2
  %umax16 = atomicrmw umax ptr addrspace(1) %P16, i16 %V16 syncscope("device") monotonic, align 2
  %nand16 = atomicrmw nand ptr addrspace(1) %P16, i16 %V16 syncscope("device") monotonic, align 2
  %faddh = atomicrmw fadd ptr addrspace(1) %PH, half %VH syncscope("device") monotonic, align 2

  ; cmpxchg: strong and weak, i8 and i16, global and generic
  %cas = cmpxchg ptr addrspace(1) %P8, i8 %V8, i8 %add syncscope("device") monotonic monotonic, align 1
  %cas.val = extractvalue { i8, i1 } %cas, 0
  %cas.ok = extractvalue { i8, i1 } %cas, 1
  %casw = cmpxchg weak ptr addrspace(4) %G8, i8 %V8, i8 %sub syncscope("device") acq_rel monotonic, align 1
  %casw.val = extractvalue { i8, i1 } %casw, 0
  %cas16 = cmpxchg ptr addrspace(1) %P16, i16 %V16, i16 %add16 seq_cst seq_cst, align 2
  %cas16.val = extractvalue { i16, i1 } %cas16, 0
  %casl = cmpxchg ptr addrspace(3) %L8, i8 %V8, i8 %xor syncscope("workgroup") monotonic monotonic, align 1
  %casl.val = extractvalue { i8, i1 } %casl, 0

  ; a 32 bit atomic that must be left alone
  %add32 = atomicrmw add ptr addrspace(1) %P32, i32 %V32 syncscope("device") monotonic, align 4

  ; keep every result alive
  %s0 = add i8 %ld.mono, %ld.acq
  %s1 = add i8 %s0, %ld.local
  %s2 = add i8 %s1, %xchg
  %s3 = add i8 %s2, %add
  %s4 = add i8 %s3, %sub
  %s5 = add i8 %s4, %and
  %s6 = add i8 %s5, %or
  %s7 = add i8 %s6, %xor
  %s8 = add i8 %s7, %min
  %s9 = add i8 %s8, %max
  %s10 = add i8 %s9, %umin
  %s11 = add i8 %s10, %umax
  %s12 = add i8 %s11, %nand
  %s13 = add i8 %s12, %cas.val
  %s14 = add i8 %s13, %casw.val
  %s15 = add i8 %s14, %casl.val
  %ok8 = zext i1 %cas.ok to i8
  %s16 = add i8 %s15, %ok8
  store i8 %s16, ptr addrspace(1) %Out8, align 1
  %t0 = add i16 %ld.seq, %xchg16
  %t1 = add i16 %t0, %add16
  %t2 = add i16 %t1, %sub16
  %t3 = add i16 %t2, %and16
  %t4 = add i16 %t3, %or16
  %t5 = add i16 %t4, %xor16
  %t6 = add i16 %t5, %min16
  %t7 = add i16 %t6, %max16
  %t8 = add i16 %t7, %umin16
  %t9 = add i16 %t8, %umax16
  %t10 = add i16 %t9, %nand16
  %t11 = add i16 %t10, %cas16.val
  %hbits = bitcast half %faddh to i16
  %t12 = add i16 %t11, %hbits
  %a32 = trunc i32 %add32 to i16
  %t13 = add i16 %t12, %a32
  store i16 %t13, ptr addrspace(1) %Out16, align 2
  ret void
}

!opencl.ocl.version = !{!0}
!0 = !{i32 2, i32 0}
