; ModuleID = 'TestHipccInvalidBitWidth-hip-spirv64-generic.bc'
source_filename = "/space/pvelesko/chipStar/fix-promote-int-pass/tests/compiler/TestHipccInvalidBitWidth.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv64-unknown-unknown"

@__chipspv_abort_called = weak hidden addrspace(1) externally_initialized global i32 0, align 4
@.str = private unnamed_addr addrspace(1) constant [47 x i8] c"%s:%u: %s: Device-side assertion `%s' failed.\0A\00", align 1
@.str.5 = private unnamed_addr addrspace(2) constant [3 x i8] c"%c\00", align 1
@.str.5.12 = private unnamed_addr addrspace(2) constant [3 x i8] c"%c\00", align 1
@llvm.used = appending global [20 x ptr] [ptr @_chip_tex1df_impl, ptr @_chip_tex1df_impl.3, ptr @_chip_tex1dfetchf_impl, ptr @_chip_tex1dfetchf_impl.4, ptr @_chip_tex1dfetchi_impl, ptr @_chip_tex1dfetchi_impl.5, ptr @_chip_tex1dfetchu_impl, ptr @_chip_tex1dfetchu_impl.6, ptr @_chip_tex1di_impl, ptr @_chip_tex1di_impl.7, ptr @_chip_tex1du_impl, ptr @_chip_tex1du_impl.8, ptr @_chip_tex2df_impl, ptr @_chip_tex2df_impl.9, ptr @_chip_tex2di_impl, ptr @_chip_tex2di_impl.10, ptr @_chip_tex2du_impl, ptr @_chip_tex2du_impl.11, ptr @_cl_print_str, ptr @_cl_print_str.2], section "llvm.metadata"
@llvm.compiler.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__chipspv_abort_called to ptr)], section "llvm.metadata"

; Function Attrs: convergent mustprogress noinline nounwind
define weak hidden spir_func void @__assert_fail(ptr addrspace(4) noundef %assertion, ptr addrspace(4) noundef %file, i32 noundef %line, ptr addrspace(4) noundef %function) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func i32 (ptr addrspace(4), ...) @printf(ptr addrspace(4) noundef addrspacecast (ptr addrspace(1) @.str to ptr addrspace(4)), ptr addrspace(4) noundef %file, i32 noundef %line, ptr addrspace(4) noundef %function, ptr addrspace(4) noundef %assertion) #7
  tail call spir_func void @__chipspv_abort(ptr addrspace(4) noundef addrspacecast (ptr addrspace(1) @__chipspv_abort_called to ptr addrspace(4))) #7
  ret void
}

; Function Attrs: convergent nofree nounwind
declare hidden spir_func noundef i32 @printf(ptr addrspace(4) nocapture noundef readonly, ...) local_unnamed_addr #1

; Function Attrs: convergent norecurse nounwind
define hidden spir_kernel void @_Z12testWarpCalcPi(ptr addrspace(1) noundef %debug.coerce) local_unnamed_addr #2 {
entry:
  %call.i8 = tail call spir_func i64 @_Z12get_local_idj(i32 noundef 0) #7
  %conv.i = trunc i64 %call.i8 to i32
  %call.i = tail call spir_func i64 @_Z12get_group_idj(i32 noundef 0) #7
  %conv.i9 = trunc i64 %call.i to i32
  %call.i10 = tail call spir_func i64 @_Z14get_local_sizej(i32 noundef 0) #7
  %conv.i11 = trunc i64 %call.i10 to i32
  %mul = mul i32 %conv.i11, %conv.i9
  %add = add i32 %mul, %conv.i
  %cmp.not12 = icmp slt i32 %conv.i, 0
  br i1 %cmp.not12, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %0 = add nsw i32 %conv.i, -1
  %1 = zext i32 %0 to i33
  %2 = trunc i64 %call.i8 to i33
  %3 = and i33 %2, 4294967295
  %4 = mul i33 %3, %1
  %5 = lshr i33 %4, 1
  %6 = trunc i33 %5 to i32
  %7 = add i32 %conv.i, %6
  %8 = mul i32 %add, %7
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body.preheader, %entry
  %result.0.lcssa = phi i32 [ 0, %entry ], [ %8, %for.body.preheader ]
  %9 = addrspacecast ptr addrspace(1) %debug.coerce to ptr addrspace(4)
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i32, ptr addrspace(4) %9, i64 %idxprom
  %call.i.i = tail call spir_func noundef i32 @_Z24atomic_exchange_explicitPU3AS4Vii12memory_order12memory_scope(ptr addrspace(4) noundef %arrayidx, i32 noundef %result.0.lcssa, i32 noundef 0, i32 noundef 2) #7
  ret void
}

; Function Attrs: convergent nounwind
declare hidden spir_func void @__chipspv_abort(ptr addrspace(4) noundef) local_unnamed_addr #3

; Function Attrs: convergent nounwind
declare hidden spir_func i64 @_Z12get_local_idj(i32 noundef) local_unnamed_addr #3

; Function Attrs: convergent nounwind
declare hidden spir_func i64 @_Z12get_group_idj(i32 noundef) local_unnamed_addr #3

; Function Attrs: convergent nounwind
declare hidden spir_func i64 @_Z14get_local_sizej(i32 noundef) local_unnamed_addr #3

; Function Attrs: convergent nofree norecurse nounwind
define internal spir_func void @_cl_print_str(ptr addrspace(4) nocapture noundef readonly %S) #4 {
entry:
  %0 = load i8, ptr addrspace(4) %S, align 1, !tbaa !5
  %cmp.not4 = icmp eq i8 %0, 0
  br i1 %cmp.not4, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %1 = phi i8 [ %2, %while.body ], [ %0, %entry ]
  %Pos.05 = phi i32 [ %inc, %while.body ], [ 0, %entry ]
  %conv = sext i8 %1 to i32
  %call = tail call spir_func i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) noundef @.str.5, i32 noundef %conv) #7
  %inc = add i32 %Pos.05, 1
  %idxprom = zext i32 %inc to i64
  %arrayidx = getelementptr inbounds i8, ptr addrspace(4) %S, i64 %idxprom
  %2 = load i8, ptr addrspace(4) %arrayidx, align 1, !tbaa !5
  %cmp.not = icmp eq i8 %2, 0
  br i1 %cmp.not, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  ret void
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(read)
define internal spir_func <4 x float> @_chip_tex1df_impl(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, float noundef %Pos) #5 {
entry:
  %call = tail call spir_func <4 x float> @_Z11read_imagef14ocl_image1d_ro11ocl_samplerf(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, float noundef %Pos) #8
  ret <4 x float> %call
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(read)
define internal spir_func <4 x float> @_chip_tex1dfetchf_impl(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, i32 noundef %Pos) #5 {
entry:
  %call = tail call spir_func <4 x float> @_Z11read_imagef14ocl_image1d_ro11ocl_sampleri(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, i32 noundef %Pos) #8
  ret <4 x float> %call
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(read)
define internal spir_func <4 x i32> @_chip_tex1dfetchi_impl(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, i32 noundef %Pos) #5 {
entry:
  %call = tail call spir_func <4 x i32> @_Z11read_imagei14ocl_image1d_ro11ocl_sampleri(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, i32 noundef %Pos) #8
  ret <4 x i32> %call
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(read)
define internal spir_func <4 x i32> @_chip_tex1dfetchu_impl(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, i32 noundef %Pos) #5 {
entry:
  %call = tail call spir_func <4 x i32> @_Z12read_imageui14ocl_image1d_ro11ocl_sampleri(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, i32 noundef %Pos) #8
  ret <4 x i32> %call
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(read)
define internal spir_func <4 x i32> @_chip_tex1di_impl(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, float noundef %Pos) #5 {
entry:
  %call = tail call spir_func <4 x i32> @_Z11read_imagei14ocl_image1d_ro11ocl_samplerf(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, float noundef %Pos) #8
  ret <4 x i32> %call
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(read)
define internal spir_func <4 x i32> @_chip_tex1du_impl(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, float noundef %Pos) #5 {
entry:
  %call = tail call spir_func <4 x i32> @_Z12read_imageui14ocl_image1d_ro11ocl_samplerf(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, float noundef %Pos) #8
  ret <4 x i32> %call
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(read)
define internal spir_func <4 x float> @_chip_tex2df_impl(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, <2 x float> noundef %Pos) #5 {
entry:
  %call = tail call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, <2 x float> noundef %Pos) #8
  ret <4 x float> %call
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(read)
define internal spir_func <4 x i32> @_chip_tex2di_impl(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, <2 x float> noundef %Pos) #5 {
entry:
  %call = tail call spir_func <4 x i32> @_Z11read_imagei14ocl_image2d_ro11ocl_samplerDv2_f(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, <2 x float> noundef %Pos) #8
  ret <4 x i32> %call
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(read)
define internal spir_func <4 x i32> @_chip_tex2du_impl(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, <2 x float> noundef %Pos) #5 {
entry:
  %call = tail call spir_func <4 x i32> @_Z12read_imageui14ocl_image2d_ro11ocl_samplerDv2_f(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, <2 x float> noundef %Pos) #8
  ret <4 x i32> %call
}

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(read)
declare spir_func <4 x i32> @_Z12read_imageui14ocl_image2d_ro11ocl_samplerDv2_f(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0), target("spirv.Sampler"), <2 x float> noundef) local_unnamed_addr #6

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(read)
declare spir_func <4 x i32> @_Z11read_imagei14ocl_image2d_ro11ocl_samplerDv2_f(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0), target("spirv.Sampler"), <2 x float> noundef) local_unnamed_addr #6

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(read)
declare spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0), target("spirv.Sampler"), <2 x float> noundef) local_unnamed_addr #6

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(read)
declare spir_func <4 x i32> @_Z12read_imageui14ocl_image1d_ro11ocl_samplerf(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0), target("spirv.Sampler"), float noundef) local_unnamed_addr #6

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(read)
declare spir_func <4 x i32> @_Z11read_imagei14ocl_image1d_ro11ocl_samplerf(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0), target("spirv.Sampler"), float noundef) local_unnamed_addr #6

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(read)
declare spir_func <4 x i32> @_Z12read_imageui14ocl_image1d_ro11ocl_sampleri(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0), target("spirv.Sampler"), i32 noundef) local_unnamed_addr #6

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(read)
declare spir_func <4 x i32> @_Z11read_imagei14ocl_image1d_ro11ocl_sampleri(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0), target("spirv.Sampler"), i32 noundef) local_unnamed_addr #6

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(read)
declare spir_func <4 x float> @_Z11read_imagef14ocl_image1d_ro11ocl_sampleri(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0), target("spirv.Sampler"), i32 noundef) local_unnamed_addr #6

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(read)
declare spir_func <4 x float> @_Z11read_imagef14ocl_image1d_ro11ocl_samplerf(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0), target("spirv.Sampler"), float noundef) local_unnamed_addr #6

; Function Attrs: convergent nounwind
declare spir_func i32 @_Z24atomic_exchange_explicitPU3AS4Vii12memory_order12memory_scope(ptr addrspace(4) noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #3

; Function Attrs: convergent nofree norecurse nounwind
define internal spir_func void @_cl_print_str.2(ptr addrspace(4) nocapture noundef readonly %S) #4 {
entry:
  %0 = load i8, ptr addrspace(4) %S, align 1, !tbaa !5
  %cmp.not4 = icmp eq i8 %0, 0
  br i1 %cmp.not4, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %1 = phi i8 [ %2, %while.body ], [ %0, %entry ]
  %Pos.05 = phi i32 [ %inc, %while.body ], [ 0, %entry ]
  %conv = sext i8 %1 to i32
  %call = tail call spir_func i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) noundef @.str.5.12, i32 noundef %conv) #7
  %inc = add i32 %Pos.05, 1
  %idxprom = zext i32 %inc to i64
  %arrayidx = getelementptr inbounds i8, ptr addrspace(4) %S, i64 %idxprom
  %2 = load i8, ptr addrspace(4) %arrayidx, align 1, !tbaa !5
  %cmp.not = icmp eq i8 %2, 0
  br i1 %cmp.not, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  ret void
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(read)
define internal spir_func <4 x float> @_chip_tex1df_impl.3(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, float noundef %Pos) #5 {
entry:
  %call = tail call spir_func <4 x float> @_Z11read_imagef14ocl_image1d_ro11ocl_samplerf(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, float noundef %Pos) #8
  ret <4 x float> %call
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(read)
define internal spir_func <4 x float> @_chip_tex1dfetchf_impl.4(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, i32 noundef %Pos) #5 {
entry:
  %call = tail call spir_func <4 x float> @_Z11read_imagef14ocl_image1d_ro11ocl_sampleri(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, i32 noundef %Pos) #8
  ret <4 x float> %call
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(read)
define internal spir_func <4 x i32> @_chip_tex1dfetchi_impl.5(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, i32 noundef %Pos) #5 {
entry:
  %call = tail call spir_func <4 x i32> @_Z11read_imagei14ocl_image1d_ro11ocl_sampleri(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, i32 noundef %Pos) #8
  ret <4 x i32> %call
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(read)
define internal spir_func <4 x i32> @_chip_tex1dfetchu_impl.6(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, i32 noundef %Pos) #5 {
entry:
  %call = tail call spir_func <4 x i32> @_Z12read_imageui14ocl_image1d_ro11ocl_sampleri(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, i32 noundef %Pos) #8
  ret <4 x i32> %call
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(read)
define internal spir_func <4 x i32> @_chip_tex1di_impl.7(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, float noundef %Pos) #5 {
entry:
  %call = tail call spir_func <4 x i32> @_Z11read_imagei14ocl_image1d_ro11ocl_samplerf(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, float noundef %Pos) #8
  ret <4 x i32> %call
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(read)
define internal spir_func <4 x i32> @_chip_tex1du_impl.8(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, float noundef %Pos) #5 {
entry:
  %call = tail call spir_func <4 x i32> @_Z12read_imageui14ocl_image1d_ro11ocl_samplerf(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, float noundef %Pos) #8
  ret <4 x i32> %call
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(read)
define internal spir_func <4 x float> @_chip_tex2df_impl.9(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, <2 x float> noundef %Pos) #5 {
entry:
  %call = tail call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, <2 x float> noundef %Pos) #8
  ret <4 x float> %call
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(read)
define internal spir_func <4 x i32> @_chip_tex2di_impl.10(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, <2 x float> noundef %Pos) #5 {
entry:
  %call = tail call spir_func <4 x i32> @_Z11read_imagei14ocl_image2d_ro11ocl_samplerDv2_f(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, <2 x float> noundef %Pos) #8
  ret <4 x i32> %call
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(read)
define internal spir_func <4 x i32> @_chip_tex2du_impl.11(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, <2 x float> noundef %Pos) #5 {
entry:
  %call = tail call spir_func <4 x i32> @_Z12read_imageui14ocl_image2d_ro11ocl_samplerDv2_f(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %I, target("spirv.Sampler") %S, <2 x float> noundef %Pos) #8
  ret <4 x i32> %call
}

attributes #0 = { convergent mustprogress noinline nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent nofree nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" }
attributes #3 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { convergent nofree norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #5 = { convergent mustprogress nofree norecurse nounwind willreturn memory(read) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #6 = { convergent mustprogress nofree nounwind willreturn memory(read) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #7 = { convergent nounwind }
attributes #8 = { convergent nounwind willreturn memory(read) }

!llvm.module.flags = !{!0, !1}
!opencl.ocl.version = !{!2, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3}
!llvm.ident = !{!4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 0, i32 0}
!3 = !{i32 2, i32 0}
!4 = !{!"clang version 18.1.5 (https://github.com/CHIP-SPV/llvm-project.git 5c39d7d1aa6e54a9c8df41002d419c398ec8830c)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
