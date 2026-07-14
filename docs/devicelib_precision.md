# chipStar Device Library Math Function Precision

This page documents the numerical precision (accuracy) of the HIP/CUDA
device-side math functions provided by the chipStar Device Library.

## How chipStar implements device math functions

chipStar does not reimplement the bulk of the HIP/CUDA math API. Instead, each
HIP math builtin (for example `sinf`, `expf`, `sqrt`, `pow`) is mapped to an
underlying device-side implementation provided by one of:

1. **OpenCL** built-in math functions (the common case),
2. the **OCML** library (AMD's Open Compute Math Library, shipped as part of
   [ROCm-Device-Libs](https://github.com/RadeonOpenCompute/ROCm-Device-Libs)),
   used when OpenCL does not provide an equivalent, or
3. a **custom implementation** in `bitcode/devicelib.cl` when neither OpenCL nor
   OCML provides the function.

See [`bitcode/README-devicelib.md`](../bitcode/README-devicelib.md) for the
mapping mechanism and the naming conventions, and
[`Device_API_functions.md`](Device_API_functions.md) /
[`Device_API_support_matrix.md`](Device_API_support_matrix.md) for the full list
of supported functions.

## Precision guarantee

Because most functions resolve to their OpenCL counterparts, **the precision of
chipStar device math functions follows the OpenCL numerical compliance
specification**. The OpenCL specification defines the maximum error, in ULP
(units in the last place), that a conforming device is allowed to produce for
each built-in math function.

The authoritative reference is the "Relative Error as ULPs" table in the OpenCL
specification:

- OpenCL C specification, *Relative Error as ULPs*:
  <https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_C.html#relative-error-as-ulps>

Because the actual accuracy is provided by the OpenCL driver / device, the exact
ULP error observed for a given function can vary between vendors and devices, but
a conforming device must stay within the bounds listed in that table. For
functions provided by OCML or by chipStar's own `devicelib.cl`, the accuracy is
whatever those implementations provide and is generally intended to match the
same OpenCL ULP bounds.

## ULP bounds and measured accuracy for commonly used functions

The table below lists, for the most frequently used single-precision functions:
the OpenCL specification's maximum allowed error (the *guarantee* a conforming
full-profile device must meet), and the error actually *measured* on an Intel
Arc B570 dGPU. Measured values come from the
[ulp-test](https://github.com/pvelesko/ulp-test) tool, which compares 1000
inputs per function against a CPU reference. Measured numbers are specific to
this device/driver and are shown only to illustrate real-world behaviour --- the
spec bound is the portable guarantee.

| Function        | OpenCL spec max (single, full profile) | Measured max ULP, standard (Arc B570) |
|-----------------|----------------------------------------|----------------------------------------|
| `sqrtf`         | correctly rounded (<= 0.5 ULP)         | 1                                      |
| `sinf`          | <= 4 ULP                               | 1                                      |
| `cosf`          | <= 4 ULP                               | 1                                      |
| `tanf`          | <= 5 ULP                               | 2                                      |
| `expf`          | <= 3 ULP                               | 2                                      |
| `logf`          | <= 3 ULP                               | 1                                      |
| `powf`          | <= 16 ULP                              | 2                                      |
| `rsqrtf` (1/sqrt)| <= 2 ULP                              | 1                                      |
| `1/x`           | <= 2.5 ULP                             | 1                                      |
| `x/y` (division)| <= 2.5 ULP                             | 1                                      |

On this device the standard OpenCL functions come in well inside the spec bounds.
Actual ULP error varies by vendor and device; the OpenCL spec table linked above
is the authoritative source, including double- and half-precision entries.

## Fast math (`CHIP_FAST_MATH`)

chipStar supports a fast-math mode. When the device library is compiled with the
`CHIP_FAST_MATH` macro defined (which happens under fast-math compilation), a
number of single-precision functions such as `sinf`, `cosf`, `expf`, `logf`,
`sqrtf`, and division are switched from their accurate OpenCL builtins to the
OpenCL `native_*` builtins (for example `native_sin`, `native_sqrt`,
`native_divide`).

The `native_*` builtins trade accuracy for speed: their precision is
**implementation-defined** by the OpenCL specification, which places no fixed ULP
requirement on them. In practice the accuracy loss ranges from negligible to
severe. Measured on the Arc B570 with [ulp-test](https://github.com/pvelesko/ulp-test)
(max ULP of the `native_*` variant vs. a CPU reference over 1000 inputs):

| Function | `native_*` max ULP (Arc B570) |
|----------|-------------------------------|
| `sqrtf`  | 1                             |
| division | 1                             |
| `rsqrtf` | 1                             |
| `expf`   | 7                             |
| `logf`   | 15                            |
| `cosf`   | 511                           |
| `sinf`   | ~8.7e8                        |
| `tanf`   | ~8.7e8                        |

Note how `native_sin`/`native_tan` lose essentially all accuracy for large
arguments (they have no argument-reduction guarantee). Use fast math only when
the reduced precision is acceptable for your workload.

See `include/hip/devicelib/single_precision/sp_math.hh` for the exact set of
functions affected by `CHIP_FAST_MATH`.
