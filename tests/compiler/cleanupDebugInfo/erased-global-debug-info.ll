; HipCleanupPass erases __hip_cuid* globals. It must also detach their debug
; info: leaving a DIGlobalVariableExpression in the compile unit's globals list
; makes the SPIR-V producers emit a DebugGlobalVariable describing storage that
; no longer exists, which segfaults Intel's CPU OpenCL runtime.
;
; @kept_global is not removed by the pass, so its debug info must survive --
; this guards against over-stripping.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv64-unknown-unknown"

@__hip_cuid_deadbeef = addrspace(1) global i8 0, align 1, !dbg !6
@kept_global = addrspace(1) global i32 7, align 4, !dbg !10

define spir_kernel void @k(ptr addrspace(1) %out) !dbg !14 {
entry:
  %v = load i32, ptr addrspace(1) @kept_global, align 4, !dbg !17
  store i32 %v, ptr addrspace(1) %out, align 4, !dbg !17
  ret void, !dbg !17
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "chipStar test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !2)
!1 = !DIFile(filename: "erased-global-debug-info.ll", directory: "/")
!2 = !{!6, !10}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "__hip_cuid_deadbeef", scope: !0, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true)
!8 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "kept_global", scope: !0, file: !1, line: 2, type: !12, isLocal: false, isDefinition: true)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DISubroutineType(types: !16)
!14 = distinct !DISubprogram(name: "k", scope: !1, file: !1, line: 4, type: !13, scopeLine: 4, spFlags: DISPFlagDefinition, unit: !0)
!16 = !{null}
!17 = !DILocation(line: 5, column: 3, scope: !14)
