## chipStar Device Library Coverage Roadmap

This is a roadmap/checklist of the device-side library (math + intrinsic)
coverage in chipStar's OpenCL/SPIR-V based devicelib. It complements the
higher-level count in `Device_API_support_matrix.md` and the per-function
table in `Device_API_functions.md`, and points at the open issues that
track the remaining gaps.

Every entry below is grounded in a source file that was inspected; the
source is cited in the notes column. Implementations come from three
places (see `bitcode/README-devicelib.md`):

* **OpenCL** built-ins (mapped in `bitcode/c_to_opencl.def` and the
  `include/hip/devicelib/**` headers),
* **OCML** (ROCm-Device-Libs, `__ocml_*`), and
* **chipStar custom** code in `bitcode/devicelib.cl`.

Legend: [x] implemented · [~] partial / conditional · [ ] missing.

### Integer intrinsics

| Status | Functions | Source / notes |
|--------|-----------|----------------|
| [x] | `__clz` / `__clzll`, `__ffs` / `__ffsll`, `__popc` / `__popcll` | `integer/int_intrinsics.hh`; `__ffs*` -> `__chip_ffs*`, `__clz*` -> OpenCL `clz` (`devicelib.cl`) |
| [x] | `__mul24` / `__umul24`, `__mulhi` / `__umulhi`, `__mul64hi` / `__umul64hi` | `integer/int_intrinsics.hh`; 64-bit high-multiply is custom (`__chip_mul64hi`/`__chip_umul64hi`, `devicelib.cl`). Related: [#631](https://github.com/CHIP-SPV/chipStar/issues/631) |
| [x] | `__hadd` / `__uhadd`, `__rhadd` / `__urhadd` | `integer/int_intrinsics.hh` -> OpenCL `hadd`/`rhadd` |
| [x] | `__sad` / `__usad`, `__byte_perm`, `__brev` / `__brevll` | custom in `devicelib.cl` |
| [x] | `__funnelshift_l/lc/r/rc` | custom in `devicelib.cl` |

### Type-cast intrinsics

| Status | Functions | Source / notes |
|--------|-----------|----------------|
| [x] | `__int_as_float` / `__float_as_int`, `__uint_as_float` / `__float_as_uint` | bit-reinterpret casts in `type_casting_intrinsics.hh` |
| [x] | `__longlong_as_double` / `__double_as_longlong` | bit-reinterpret casts in `type_casting_intrinsics.hh` |
| [x] | `__double2hiint` / `__double2loint`, `__hiloint2double` | double <-> hi/lo int halves in `type_casting_intrinsics.hh` |

### Single- and double-precision math

| Status | Functions | Source / notes |
|--------|-----------|----------------|
| [x] | Standard float/double math (`sin`, `cos`, `exp`, `log`, `pow`, `lgamma`, `cbrt`, ...) | 62 C-math -> OpenCL maps in `bitcode/c_to_opencl.def` |
| [x] | Bessel `j0/j1/y0/y1/jn/yn` (f32/f64) | via OCML `__ocml_j0_f32` etc. (`devicelib.cl`) |
| [~] | Fast float intrinsics (`__expf`, `__cosf`, `__fdividef`, ...) | mapped to OpenCL `native_*` (`single_precision/sp_intrinsics.hh`). Reduced precision by design; only a subset of CUDA's fast intrinsics have `native_*` equivalents |
| [~] | Rounding-mode float intrinsics (`__fsqrt_rd/rn/ru/rz`, `__fadd_rn`, ...) | rounding suffix currently ignored: all `__fsqrt_r*` collapse to `sqrt(x)` (`sp_intrinsics.hh`). Precision caveat — see [#428](https://github.com/CHIP-SPV/chipStar/issues/428) |

### Half (fp16) and bfloat16

| Status | Functions | Source / notes |
|--------|-----------|----------------|
| [~] | `__half` / `__half2` arithmetic, comparison, math, conversion | `__half` wraps native `_Float16` (`spirv_hip_fp16.h`); headers in `include/hip/devicelib/half/` |
| [~] | fp16 correctness / performance on non-fp16 HW | fp16 quality/software-fallback path tracked by [#927](https://github.com/CHIP-SPV/chipStar/issues/927) |
| [x] | fp16 conversion round modes | `__ocml_cvtrtn/rtp/rtz_f16_f32` in `devicelib.cl` |
| [~] | `__nv_bfloat16` / `__nv_bfloat162` | headers present in `include/hip/devicelib/bfloat16/`; coverage not on par with CUDA |

Per `Device_API_support_matrix.md`: ~81/96 half and ~99/115 half2
funcs+intrinsics; still missing e.g. some `__hadd_rn`, `__hfma_relu`,
`__hmul_rn`, `__hmax_nan`, `ldcv/ldlu/stwb/stwt`.

### Atomics

| Status | Functions | Source / notes |
|--------|-----------|----------------|
| [x] | Integer/unsigned/long `atomicAdd/Sub/Min/Max/And/Or/Xor/Exch/CAS/Inc/Dec` incl. `_system` CAS/Min/Max | `__chip_atomic_*` in `devicelib.cl` |
| [~] | Float/double `atomicAdd`, float `atomicMin/Max` | native vs. emulated builds selected at link time: `atomicAdd{Float,Double}_{native,emulation}.cl`, `atomicMinMaxFloat_emulation.cl`. Efficient path needs `cl_ext_float_atomics` |

### Warp / cross-lane

| Status | Functions | Source / notes |
|--------|-----------|----------------|
| [~] | Ballot / vote (`__ballot`, `__all`, `__any`) | `bitcode/ballot_native.cl`; needs subgroup support |
| [~] | Shuffle (`__shfl*`) | works when subgroup size == warp (32) and lanes map 1:1; `width` ignored (see `Device_API_support_matrix.md`) |
| [ ] | Cooperative groups, warp-matrix (WMMA) | headers only / unsupported |

### Textures / surfaces

| Status | Functions | Source / notes |
|--------|-----------|----------------|
| [~] | `tex1D`, `tex1Dfetch`, `tex2D`, `tex3D` | declared in `spirv_texture_functions.h`; runtime in `bitcode/texture.cl` (1D/2D best supported) |
| [ ] | `tex2Dgather` and other gather/layered/cubemap | no `gather` entry point in `spirv_texture_functions.h`. Tracked by [#176](https://github.com/CHIP-SPV/chipStar/issues/176) |
| [ ] | Surface functions | unsupported |

### Device-side allocation

| Status | Functions | Source / notes |
|--------|-----------|----------------|
| [~] | `malloc`/`free`, `operator new`/`delete` | `bitcode/malloc.cl` (device_malloc/device_free wrappers, `devicelib.cl`) |

### Known precision differences

* Fast (`__*f`) intrinsics use OpenCL `native_*` — lower accuracy than the
  correctly-rounded default variants (`single_precision/sp_intrinsics.hh`).
* Rounding-mode suffixes on float intrinsics are currently ignored
  (`sp_intrinsics.hh`).
* OCML-sourced functions are compiled with a fixed OCLC control-library
  configuration (finite-only off, unsafe-math off, DAZ off; see
  `bitcode/README-devicelib.md`); correctness of the amdgcn-derived paths
  is "poorly tested" per that README.
* Detailed precision documentation is tracked separately in
  [#428](https://github.com/CHIP-SPV/chipStar/issues/428).
