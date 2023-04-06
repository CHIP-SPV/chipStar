/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries (orig. repo location: irif/inc)
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#ifndef IRIF_H
#define IRIF_H

#pragma OPENCL EXTENSION cl_khr_fp16 : enable



#define REQUIRES_16BIT_INSTS __attribute__((target("16-bit-insts")))
#define REQUIRES_GFX9_INSTS __attribute__((target("gfx9-insts")))

// Generic intrinsics
extern __attribute__((const)) half __llvm_sqrt_f16(half) __asm("llvm.sqrt.f16");
extern __attribute__((const)) half __llvm_exp2_f16(half) __asm("llvm.exp2.f16");
extern __attribute__((const)) half __llvm_exp_f16(half) __asm("llvm.exp.f16");
extern __attribute__((const)) half __llvm_log_f16(half) __asm("llvm.log.f16");
extern __attribute__((const)) half __llvm_log2_f16(half) __asm("llvm.log2.f16");
extern __attribute__((const)) half __llvm_log10_f16(half) __asm("llvm.log10.f16");

extern __attribute__((const)) half __llvm_sin_f16(half) __asm("llvm.sin.f16");
extern __attribute__((const)) half __llvm_cos_f16(half) __asm("llvm.cos.f16");

extern __attribute__((const)) half __llvm_fma_f16(half, half, half) __asm("llvm.fma.f16");
extern __attribute__((const)) half2 __llvm_fma_2f16(half2, half2, half2) __asm("llvm.fma.v2f16");

extern __attribute__((const)) half __llvm_fabs_f16(half) __asm("llvm.fabs.f16");
extern __attribute__((const)) half2 __llvm_fabs_2f16(half2) __asm("llvm.fabs.v2f16");

extern __attribute__((const)) half __llvm_minnum_f16(half, half) __asm("llvm.minnum.f16");
extern __attribute__((const)) half2 __llvm_minnum_2f16(half2, half2) __asm("llvm.minnum.v2f16");

extern __attribute__((const)) half __llvm_maxnum_f16(half, half) __asm("llvm.maxnum.f16");
extern __attribute__((const)) half2 __llvm_maxnum_2f16(half2, half2) __asm("llvm.maxnum.v2f16");

extern __attribute__((const)) half __llvm_copysign_f16(half, half) __asm("llvm.copysign.f16");
extern __attribute__((const)) half2 __llvm_copysign_2f16(half2, half2) __asm("llvm.copysign.v2f16");

extern __attribute__((const)) half __llvm_floor_f16(half) __asm("llvm.floor.f16");
extern __attribute__((const)) half2 __llvm_floor_2f16(half2) __asm("llvm.floor.v2f16");

extern __attribute__((const)) half __llvm_ceil_f16(half) __asm("llvm.ceil.f16");
extern __attribute__((const)) half2 __llvm_ceil_2f16(half2) __asm("llvm.ceil.v2f16");

extern __attribute__((const)) half __llvm_trunc_f16(half) __asm("llvm.trunc.f16");
extern __attribute__((const)) half2 __llvm_trunc_2f16(half2) __asm("llvm.trunc.v2f16");

extern __attribute__((const)) half __llvm_rint_f16(half) __asm("llvm.rint.f16");
extern __attribute__((const)) half2 __llvm_rint_2f16(half2) __asm("llvm.rint.v2f16");


extern __attribute__((const)) half __llvm_canonicalize_f16(half) __asm("llvm.canonicalize.f16");
extern __attribute__((const)) half2 __llvm_canonicalize_2f16(half2) __asm("llvm.canonicalize.v2f16");

// Intrinsics requiring wrapping
extern __attribute__((const)) uchar __llvm_ctlz_i8(uchar);
extern __attribute__((const)) ushort __llvm_ctlz_i16(ushort);
extern __attribute__((const)) uint __llvm_ctlz_i32(uint);
extern __attribute__((const)) ulong __llvm_ctlz_i64(ulong);

extern __attribute__((const)) uchar __llvm_cttz_i8(uchar);
extern __attribute__((const)) ushort __llvm_cttz_i16(ushort);
extern __attribute__((const)) uint __llvm_cttz_i32(uint);
extern __attribute__((const)) ulong __llvm_cttz_i64(ulong);

// Fence intrinsics
extern void __llvm_fence_acq_wi(void);
extern void __llvm_fence_acq_sg(void);
extern void __llvm_fence_acq_wg(void);
extern void __llvm_fence_acq_dev(void);
extern void __llvm_fence_acq_sys(void);
extern void __llvm_fence_rel_wi(void);
extern void __llvm_fence_rel_sg(void);
extern void __llvm_fence_rel_wg(void);
extern void __llvm_fence_rel_dev(void);
extern void __llvm_fence_rel_sys(void);
extern void __llvm_fence_ar_wi(void);
extern void __llvm_fence_ar_sg(void);
extern void __llvm_fence_ar_wg(void);
extern void __llvm_fence_ar_dev(void);
extern void __llvm_fence_ar_sys(void);
extern void __llvm_fence_sc_wi(void);
extern void __llvm_fence_sc_sg(void);
extern void __llvm_fence_sc_wg(void);
extern void __llvm_fence_sc_dev(void);
extern void __llvm_fence_sc_sys(void);

// Atomics
extern uint __llvm_ld_atomic_a1_x_dev_i32(__global uint *);
extern ulong __llvm_ld_atomic_a1_x_dev_i64(__global ulong *);
extern uint __llvm_ld_atomic_a3_x_wg_i32(__local uint *);
extern ulong __llvm_ld_atomic_a3_x_wg_i64(__local ulong *);

extern void __llvm_st_atomic_a1_x_dev_i32(__global uint *, uint);
extern void __llvm_st_atomic_a1_x_dev_i64(__global ulong *, ulong);
extern void __llvm_st_atomic_a3_x_wg_i32(__local uint *, uint);
extern void __llvm_st_atomic_a3_x_wg_i64(__local ulong *, ulong);

extern uint __llvm_atomic_add_a1_x_dev_i32(__global uint *, uint);
extern ulong __llvm_atomic_add_a1_x_dev_i64(__global ulong *, ulong);
extern uint __llvm_atomic_add_a3_x_wg_i32(__local uint *, uint);
extern ulong __llvm_atomic_add_a3_x_wg_i64(__local ulong *, ulong);

extern uint __llvm_atomic_and_a1_x_dev_i32(__global uint *, uint);
extern ulong __llvm_atomic_and_a1_x_dev_i64(__global ulong *, ulong);
extern uint __llvm_atomic_and_a3_x_wg_i32(__local uint *, uint);
extern ulong __llvm_atomic_and_a3_x_wg_i64(__local ulong *, ulong);

extern uint __llvm_atomic_or_a1_x_dev_i32(__global uint *, uint);
extern ulong __llvm_atomic_or_a1_x_dev_i64(__global ulong *, ulong);
extern uint __llvm_atomic_or_a3_x_wg_i32(__local uint *, uint);
extern ulong __llvm_atomic_or_a3_x_wg_i64(__local ulong *, ulong);

extern uint __llvm_atomic_max_a1_x_dev_i32(__global int *, int);
extern uint __llvm_atomic_umax_a1_x_dev_i32(__global uint *, uint);
extern ulong __llvm_atomic_max_a1_x_dev_i64(__global long *, long);
extern ulong __llvm_atomic_umax_a1_x_dev_i64(__global ulong *, ulong);
extern uint __llvm_atomic_max_a3_x_wg_i32(__local int *, int);
extern uint __llvm_atomic_umax_a3_x_wg_i32(__local uint *, uint);
extern ulong __llvm_atomic_max_a3_x_wg_i64(__local long *, long);
extern ulong __llvm_atomic_umax_a3_x_wg_i64(__local ulong *, ulong);

extern uint __llvm_atomic_min_a1_x_dev_i32(__global int *, int);
extern uint __llvm_atomic_umin_a1_x_dev_i32(__global uint *, uint);
extern ulong __llvm_atomic_min_a1_x_dev_i64(__global long *, long);
extern ulong __llvm_atomic_umin_a1_x_dev_i64(__global ulong *, ulong);
extern uint __llvm_atomic_min_a3_x_wg_i32(__local int *, int);
extern uint __llvm_atomic_umin_a3_x_wg_i32(__local uint *, uint);
extern ulong __llvm_atomic_min_a3_x_wg_i64(__local long *, long);
extern ulong __llvm_atomic_umin_a3_x_wg_i64(__local ulong *, ulong);

extern uint __llvm_cmpxchg_a1_x_x_dev_i32(__global uint *, uint, uint);
extern ulong __llvm_cmpxchg_a1_x_x_dev_i64(__global ulong *, ulong, ulong);
extern uint __llvm_cmpxchg_a3_x_x_wg_i32(__local uint *, uint, uint);
extern ulong __llvm_cmpxchg_a3_x_x_wg_i64(__local ulong *, ulong, ulong);

// AMDGPU intrinsics
extern __attribute__((const)) bool __llvm_amdgcn_class_f16(half, int) __asm("llvm.amdgcn.class.f16");

extern __attribute__((const)) half __llvm_amdgcn_fract_f16(half) __asm("llvm.amdgcn.fract.f16");
extern __attribute__((const)) half __llvm_amdgcn_rcp_f16(half) __asm("llvm.amdgcn.rcp.f16");
extern __attribute__((const)) half __llvm_amdgcn_rsq_f16(half) __asm("llvm.amdgcn.rsq.f16");
extern __attribute__((const)) half __llvm_amdgcn_ldexp_f16(half, int) __asm("llvm.amdgcn.ldexp.f16");


extern __attribute__((const)) half __llvm_amdgcn_frexp_mant_f16(half) __asm("llvm.amdgcn.frexp.mant.f16");
extern __attribute__((const)) short __llvm_amdgcn_frexp_exp_i16_f16(half) __asm("llvm.amdgcn.frexp.exp.i16.f16");

// llvm.amdgcn.update.dpp.i32 <old> <src> <dpp_ctrl> <row_mask> <bank_mask> <bound_ctrl>
extern uint __llvm_amdgcn_update_dpp_i32(uint, uint, uint, uint, uint, bool) __asm("llvm.amdgcn.update.dpp.i32");

// Operand bits: [0..3]=VM_CNT, [4..6]=EXP_CNT (Export), [8..11]=LGKM_CNT (LDS, GDS, Konstant, Message)
extern void __llvm_amdgcn_s_waitcnt(int) __asm("llvm.amdgcn.s.waitcnt");

extern __attribute__((const)) uint __llvm_amdgcn_mbcnt_lo(uint, uint) __asm("llvm.amdgcn.mbcnt.lo");
extern __attribute__((const)) uint __llvm_amdgcn_mbcnt_hi(uint, uint) __asm("llvm.amdgcn.mbcnt.hi");

extern __attribute__((const)) uint __llvm_amdgcn_ubfe_i32(uint, uint, uint) __asm("llvm.amdgcn.ubfe.i32");
extern __attribute__((const)) int __llvm_amdgcn_sbfe_i32(int, uint, uint) __asm("llvm.amdgcn.sbfe.i32");

extern __attribute__((const)) uint __llvm_amdgcn_alignbit(uint, uint, uint) __asm("llvm.amdgcn.alignbit");
extern __attribute__((const)) uint __llvm_amdgcn_alignbyte(uint, uint, uint) __asm("llvm.amdgcn.alignbyte");

extern __attribute__((const)) ulong __llvm_amdgcn_mqsad_pk_u16_u8(ulong, uint, ulong) __asm("llvm.amdgcn.mqsad.pk.u16.u8");
extern __attribute__((const)) uint __llvm_amdgcn_cvt_pk_u8_f32(float, uint, uint) __asm("llvm.amdgcn.cvt.pk.u8.f32");
extern __attribute__((const)) ulong __llvm_amdgcn_qsad_pk_u16_u8(ulong, uint, ulong) __asm("llvm.amdgcn.qsad.pk.u16.u8");
extern __attribute__((const)) uint __llvm_amdgcn_sad_u8(uint, uint, uint) __asm("llvm.amdgcn.sad.u8");
extern __attribute__((const)) uint __llvm_amdgcn_sad_hi_u8(uint, uint, uint) __asm("llvm.amdgcn.sad.hi.u8");
extern __attribute__((const)) uint __llvm_amdgcn_sad_u16(uint, uint, uint) __asm("llvm.amdgcn.sad.u16");
extern __attribute__((const)) uint __llvm_amdgcn_msad_u8(uint, uint, uint) __asm("llvm.amdgcn.msad.u8");

// Buffer Load/Store

extern __attribute__((pure)) float4 __llvm_amdgcn_buffer_load_format_v4f32(uint4 v, uint i, uint o, bool glc, bool slc) __asm("llvm.amdgcn.buffer.load.format.v4f32");
extern __attribute__((pure)) half4 __llvm_amdgcn_buffer_load_format_v4f16(uint4 v, uint i, uint o, bool glc, bool slc) __asm("llvm.amdgcn.buffer.load.format.v4f16");
extern void __llvm_amdgcn_buffer_store_format_v4f32(float4 p, uint4 v, uint i, uint o, bool glc, bool slc) __asm("llvm.amdgcn.buffer.store.format.v4f32");
extern void __llvm_amdgcn_buffer_store_format_v4f16(half4 p, uint4 v, uint i, uint o, bool glc, bool slc) __asm("llvm.amdgcn.buffer.store.format.v4f16");

// Image load, store, sample, gather
extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_1d_v4f32_i32(uint ix, uint8 t);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_2d_v4f32_i32(uint ix, uint iy, uint8 t);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_3d_v4f32_i32(uint ix, uint iy, uint iz, uint8 t);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_cube_v4f32_i32(uint ix, uint iy, uint iface, uint8 t);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_1darray_v4f32_i32(uint ix, uint islice, uint8 t);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_2darray_v4f32_i32(uint ix, uint iy, uint islice, uint8 t);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_mip_1d_v4f32_i32(uint ix, uint imip, uint8 t);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_mip_2d_v4f32_i32(uint ix, uint iy, uint imip, uint8 t);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_mip_3d_v4f32_i32(uint ix, uint iy, uint iz, uint imip, uint8 t);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_mip_cube_v4f32_i32(uint ix, uint iy, uint iface, uint imip, uint8 t);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_mip_1darray_v4f32_i32(uint ix, uint islice, uint imip, uint8 t);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_mip_2darray_v4f32_i32(uint ix, uint iy, uint islice, uint imip, uint8 t);

extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_1d_v4f16_i32(uint ix, uint8 t);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_2d_v4f16_i32(uint ix, uint iy, uint8 t);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_3d_v4f16_i32(uint ix, uint iy, uint iz, uint8 t);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_cube_v4f16_i32(uint ix, uint iy, uint iface, uint8 t);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_1darray_v4f16_i32(uint ix, uint islice, uint8 t);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_2darray_v4f16_i32(uint ix, uint iy, uint islice, uint8 t);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_mip_1d_v4f16_i32(uint ix, uint imip, uint8 t);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_mip_2d_v4f16_i32(uint ix, uint iy, uint imip, uint8 t);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_mip_3d_v4f16_i32(uint ix, uint iy, uint iz, uint imip, uint8 t);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_mip_cube_v4f16_i32(uint ix, uint iy, uint iface, uint imip, uint8 t);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_mip_1darray_v4f16_i32(uint ix, uint islice, uint imip, uint8 t);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_mip_2darray_v4f16_i32(uint ix, uint iy, uint islice, uint imip, uint8 t);

extern __attribute__((pure)) float __llvm_amdgcn_image_load_2d_f32_i32(uint ix, uint iy, uint8 t);
extern __attribute__((pure)) float __llvm_amdgcn_image_load_2darray_f32_i32(uint ix, uint iy, uint islice, uint8 t);
extern __attribute__((pure)) float __llvm_amdgcn_image_load_mip_2d_f32_i32(uint ix, uint iy, uint imip, uint8 t);
extern __attribute__((pure)) float __llvm_amdgcn_image_load_mip_2darray_f32_i32(uint ix, uint iy, uint islice, uint imip, uint8 t);

extern void __llvm_amdgcn_image_store_1d_v4f32_i32(float4 pix, uint ix, uint8 t);
extern void __llvm_amdgcn_image_store_2d_v4f32_i32(float4 pix, uint ix, uint iy, uint8 t);
extern void __llvm_amdgcn_image_store_3d_v4f32_i32(float4 pix, uint ix, uint iy, uint iz, uint8 t);
extern void __llvm_amdgcn_image_store_cube_v4f32_i32(float4 pix, uint ix, uint iy, uint iface, uint8 t);
extern void __llvm_amdgcn_image_store_1darray_v4f32_i32(float4 pix, uint ix, uint islice, uint8 t);
extern void __llvm_amdgcn_image_store_2darray_v4f32_i32(float4 pix, uint ix, uint iy, uint islice, uint8 t);
extern void __llvm_amdgcn_image_store_mip_1d_v4f32_i32(float4 pix, uint ix, uint imip, uint8 t);
extern void __llvm_amdgcn_image_store_mip_2d_v4f32_i32(float4 pix, uint ix, uint iy, uint imip, uint8 t);
extern void __llvm_amdgcn_image_store_mip_3d_v4f32_i32(float4 pix, uint ix, uint iy, uint iz, uint imip, uint8 t);
extern void __llvm_amdgcn_image_store_mip_cube_v4f32_i32(float4 pix, uint ix, uint iy, uint iface, uint imip, uint8 t);
extern void __llvm_amdgcn_image_store_mip_1darray_v4f32_i32(float4 pix, uint ix, uint islice, uint imip, uint8 t);
extern void __llvm_amdgcn_image_store_mip_2darray_v4f32_i32(float4 pix, uint ix, uint iy, uint islice, uint imip, uint8 t);

extern void __llvm_amdgcn_image_store_1d_v4f16_i32(half4 pix, uint ix, uint8 t);
extern void __llvm_amdgcn_image_store_2d_v4f16_i32(half4 pix, uint ix, uint iy, uint8 t);
extern void __llvm_amdgcn_image_store_3d_v4f16_i32(half4 pix, uint ix, uint iy, uint iz, uint8 t);
extern void __llvm_amdgcn_image_store_cube_v4f16_i32(half4 pix, uint ix, uint iy, uint iface, uint8 t);
extern void __llvm_amdgcn_image_store_1darray_v4f16_i32(half4 pix, uint ix, uint islice, uint8 t);
extern void __llvm_amdgcn_image_store_2darray_v4f16_i32(half4 pix, uint ix, uint iy, uint islice, uint8 t);
extern void __llvm_amdgcn_image_store_mip_1d_v4f16_i32(half4 pix, uint ix, uint imip, uint8 t);
extern void __llvm_amdgcn_image_store_mip_2d_v4f16_i32(half4 pix, uint ix, uint iy, uint imip, uint8 t);
extern void __llvm_amdgcn_image_store_mip_3d_v4f16_i32(half4 pix, uint ix, uint iy, uint iz, uint imip, uint8 t);
extern void __llvm_amdgcn_image_store_mip_cube_v4f16_i32(half4 pix, uint ix, uint iy, uint iface, uint imip, uint8 t);
extern void __llvm_amdgcn_image_store_mip_1darray_v4f16_i32(half4 pix, uint ix, uint islice, uint imip, uint8 t);
extern void __llvm_amdgcn_image_store_mip_2darray_v4f16_i32(half4 pix, uint ix, uint iy, uint islice, uint imip, uint8 t);

extern void __llvm_amdgcn_image_store_2d_f32_i32(float pix, uint ix, uint iy, uint8 t);
extern void __llvm_amdgcn_image_store_2darray_f32_i32(float pix, uint ix, uint iy, uint islice, uint8 t);
extern void __llvm_amdgcn_image_store_mip_2d_f32_i32(float pix, uint ix, uint iy, uint imip, uint8 t);
extern void __llvm_amdgcn_image_store_mip_2darray_f32_i32(float pix, uint ix, uint iy, uint islice, uint imip, uint8 t);

extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_lz_1d_v4f32_f32(float x, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_l_1d_v4f32_f32(float x, float lod, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_d_1d_v4f32_f32_f32(float dxdh, float dxdv, float x, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_lz_2d_v4f32_f32(float x, float y, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_l_2d_v4f32_f32(float x, float y, float lod, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_d_2d_v4f32_f32_f32(float dxdh, float dydh, float dxdv, float dydv, float x, float y, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_lz_3d_v4f32_f32(float x, float y, float z, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_l_3d_v4f32_f32(float x, float y, float z, float lod, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_d_3d_v4f32_f32_f32(float dxdh, float dydh, float dzdh, float dxdv, float dydv, float dzdv, float x, float y, float z, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_lz_cube_v4f32_f32(float x, float y, float face, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_l_cube_v4f32_f32(float x, float y, float face, float lod, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_lz_1darray_v4f32_f32(float x, float slice, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_l_1darray_v4f32_f32(float x, float slice, float lod, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_d_1darray_v4f32_f32_f32(float dxdh, float dxdv, float x, float slice, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_lz_2darray_v4f32_f32(float x, float y, float slice, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_l_2darray_v4f32_f32(float x, float y, float slice, float lod, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_d_2darray_v4f32_f32_f32(float dxdh, float dydh, float dxdv, float dydv, float x, float y, float slice, uint8 t, uint4 s);

extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_lz_1d_v4f16_f32(float x, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_l_1d_v4f16_f32(float x, float lod, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_d_1d_v4f16_f32_f32(float dxdh, float dxdv, float x, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_lz_2d_v4f16_f32(float x, float y, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_l_2d_v4f16_f32(float x, float y, float lod, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_d_2d_v4f16_f32_f32(float dxdh, float dydh, float dxdv, float dydv, float x, float y, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_lz_3d_v4f16_f32(float x, float y, float z, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_l_3d_v4f16_f32(float x, float y, float z, float lod, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_d_3d_v4f16_f32_f32(float dxdh, float dydh, float dzdh, float dxdv, float dydv, float dzdv, float x, float y, float z, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_lz_cube_v4f16_f32(float x, float y, float face, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_l_cube_v4f16_f32(float x, float y, float face, float lod, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_lz_1darray_v4f16_f32(float x, float slice, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_l_1darray_v4f16_f32(float x, float slice, float lod, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_d_1darray_v4f16_f32_f32(float dxdh, float dxdv, float x, float slice, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_lz_2darray_v4f16_f32(float x, float y, float slice, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_l_2darray_v4f16_f32(float x, float y, float slice, float lod, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_d_2darray_v4f16_f32_f32(float dxdh, float dydh, float dxdv, float dydv, float x, float y, float slice, uint8 t, uint4 s);

extern __attribute__((pure)) float __llvm_amdgcn_image_sample_lz_2d_f32_f32(float x, float y, uint8 t, uint4 s);
extern __attribute__((pure)) float __llvm_amdgcn_image_sample_l_2d_f32_f32(float x, float y, float lod, uint8 t, uint4 s);
extern __attribute__((pure)) float __llvm_amdgcn_image_sample_d_2d_f32_f32_f32(float dxdh, float dydh, float dxdv, float dydv, float x, float y, uint8 t, uint4 s);
extern __attribute__((pure)) float __llvm_amdgcn_image_sample_lz_2darray_f32_f32(float x, float y, float slice, uint8 t, uint4 s);
extern __attribute__((pure)) float __llvm_amdgcn_image_sample_l_2darray_f32_f32(float x, float y, float slice, float lod, uint8 t, uint4 s);
extern __attribute__((pure)) float __llvm_amdgcn_image_sample_d_2darray_f32_f32_f32(float dxdh, float dydh, float dxdv, float dydv, float x, float y, float slice, uint8 t, uint4 s);

extern __attribute__((pure)) float4 __llvm_amdgcn_image_gather4_lz_2d_v4f32_f32_r(float x, float y, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_gather4_lz_2d_v4f32_f32_g(float x, float y, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_gather4_lz_2d_v4f32_f32_b(float x, float y, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_gather4_lz_2d_v4f32_f32_a(float x, float y, uint8 t, uint4 s);


#pragma OPENCL EXTENSION cl_khr_fp16 : disable
#endif // IRIF_H
