source_filename = "minimal_rocprim.ll"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv64"

%"struct.rocprim::detail::lookback_scan_state" = type { ptr addrspace(4) }

define spir_kernel void @minimal_kernel(ptr noundef byval(%"struct.rocprim::detail::lookback_scan_state") align 8 %lookback_scan_state, i32 noundef %number_of_blocks, i32 noundef %add) !dbg !6 {
entry:
  br label %if.end.i

if.end.i:                                         ; preds = %entry
  %cmp.i.i38 = icmp ult i32 %add, %number_of_blocks, !dbg !7
  br i1 %cmp.i.i38, label %if.then.i.i40, label %if.end.i.i39, !dbg !8

if.then.i.i40:                                    ; preds = %if.end.i
  %ptr_field_addr = getelementptr inbounds %"struct.rocprim::detail::lookback_scan_state", ptr %lookback_scan_state, i32 0, i32 0, !dbg !9
  %agg.tmp5.sroa.0.0.copyload = load ptr addrspace(4), ptr %ptr_field_addr, align 8, !dbg !9
  %prefix.i.i25.sroa.4.0.insert.ext = zext i56 0 to i64, !dbg !10 ; Replaced undef with 0
  %prefix.i.i25.sroa.4.0.insert.shift = shl i64 %prefix.i.i25.sroa.4.0.insert.ext, 8, !dbg !10
  %prefix.i.i25.sroa.4.0.insert.mask = and i64 0, 255, !dbg !10    ; Replaced undef with 0
  %prefix.i.i25.sroa.4.0.insert.insert = or i64 %prefix.i.i25.sroa.4.0.insert.mask, %prefix.i.i25.sroa.4.0.insert.shift, !dbg !10
  %prefix.i.i25.sroa.0.0.insert.ext = zext i8 0 to i64, !dbg !10
  %prefix.i.i25.sroa.0.0.insert.mask = and i64 %prefix.i.i25.sroa.4.0.insert.insert, -256, !dbg !10 ; -256 is 0xFFFFFFFFFFFFFF00
  %prefix.i.i25.sroa.0.0.insert.insert = or i64 %prefix.i.i25.sroa.0.0.insert.mask, %prefix.i.i25.sroa.0.0.insert.ext, !dbg !10
  %add.i.i41 = add i32 32, %add, !dbg !11
  %idxprom.i.i42 = zext i32 %add.i.i41 to i64, !dbg !12
  %arrayidx.i.i43 = getelementptr inbounds i64, ptr addrspace(4) %agg.tmp5.sroa.0.0.copyload, i64 %idxprom.i.i42, !dbg !12
  store i64 %prefix.i.i25.sroa.0.0.insert.insert, ptr addrspace(4) %arrayidx.i.i43, align 8, !dbg !13
  br label %if.end.i.i39, !dbg !14

if.end.i.i39:                                       ; preds = %if.then.i.i40, %if.end.i
  ret void, !dbg !15
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 10.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "minimal.cpp", directory: "/")
!2 = !{}
!3 = !{i32 7, !"PIC Level", i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 10.0.0"}
!6 = distinct !DISubprogram(name: "minimal_kernel", scope: !1, file: !1, line: 1, type: !16, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!7 = !DILocation(line: 10, column: 1, scope: !6)
!8 = !DILocation(line: 11, column: 1, scope: !6)
!9 = !DILocation(line: 13, column: 1, scope: !6)
!10 = !DILocation(line: 14, column: 1, scope: !6)
!11 = !DILocation(line: 15, column: 1, scope: !6)
!12 = !DILocation(line: 16, column: 1, scope: !6)
!13 = !DILocation(line: 17, column: 1, scope: !6)
!14 = !DILocation(line: 18, column: 1, scope: !6)
!15 = !DILocation(line: 20, column: 1, scope: !6)
!16 = !DISubroutineType(types: !17)
!17 = !{null}

