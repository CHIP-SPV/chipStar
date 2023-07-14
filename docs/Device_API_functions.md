
# List of CUDA/HIP device-side functions supported by chipStar

## Double precision intrinsics

|   **CUDA**                                                |   **HIP**                         |  **chipStar**|
|-----------------------------------------------------------|-----------------------------------|:------------:|
|  double \_\_dadd_rd ( double  x, double  y ) | N | N |
|  double \_\_dadd_rn ( double  x, double  y ) | N | N |
|  double \_\_dadd_ru ( double  x, double  y ) | N | N |
|  double \_\_dadd_rz ( double  x, double  y ) | N | N |
|  double \_\_ddiv_rd ( double  x, double  y ) | N | N |
|  double \_\_ddiv_rn ( double  x, double  y ) | N | N |
|  double \_\_ddiv_ru ( double  x, double  y ) | N | N |
|  double \_\_ddiv_rz ( double  x, double  y ) | N | N |
|  double \_\_dmul_rd ( double  x, double  y ) | N | N |
|  double \_\_dmul_rn ( double  x, double  y ) | N | N |
|  double \_\_dmul_ru ( double  x, double  y ) | N | N |
|  double \_\_dmul_rz ( double  x, double  y ) | N | N |
|  double \_\_drcp_rd ( double  x ) | N | N |
|  double \_\_drcp_rn ( double  x ) | N | N |
|  double \_\_drcp_ru ( double  x ) | N | N |
|  double \_\_drcp_rz ( double  x ) | N | N |
|  double \_\_dsqrt_rd ( double  x ) | N | N |
|  double \_\_dsqrt_rn ( double  x ) | Y | N |
|  double \_\_dsqrt_ru ( double  x ) | N | N |
|  double \_\_dsqrt_rz ( double  x ) | N | N |
|  double \_\_dsub_rd ( double  x, double  y ) | N | N |
|  double \_\_dsub_rn ( double  x, double  y ) | N | N |
|  double \_\_dsub_ru ( double  x, double  y ) | N | N |
|  double \_\_dsub_rz ( double  x, double  y ) | N | N |
|  double \_\_fma_rd ( double  x, double  y, double  z ) | N | N |
|  double \_\_fma_rn ( double  x, double  y, double  z ) | N | N |
|  double \_\_fma_ru ( double  x, double  y, double  z ) | N | N |
|  double \_\_fma_rz ( double  x, double  y, double  z ) | N | N |

## Double precision math library

|   **CUDA**                                                |   **HIP**                         |  **chipStar**|
|-----------------------------------------------------------|-----------------------------------|:----------------:|
|  double acos(double x) | Y | Y |
|  double acosh ( double  x ) | Y | Y |
|  double asin ( double  x ) | Y | Y |
|  double asinh ( double  x ) | Y | Y |
|  double atan2 ( double  y, double  x ) | Y | Y |
|  double atan ( double  x ) | Y | Y |
|  double atanh ( double  x ) | Y | Y |
|  double cbrt ( double  x ) | Y | Y |
|  double ceil ( double  x ) | Y | Y |
|  long convert_long(double x); | Y | Y |
|  double copysign ( double  x, double  y ) | Y | Y |
|  double cos ( double  x ) | Y | Y |
|  double cosh ( double  x ) | Y | Y |
|  double cospi ( double  x ) | Y | Y |
|  double cyl_bessel_i0 ( double  x ) | Y | Y |
|  double cyl_bessel_i1 ( double  x ) | Y | Y |
|  double erfc ( double  x ) | Y | Y |
|  double erfcinv ( double  x ) | Y | Y |
|  double erfcx ( double  x ) | Y | Y |
|  double erf ( double  x ) | Y | Y |
|  double erfinv ( double  x ) | Y | Y |
|  double exp10 ( double  x ) | Y | Y |
|  double exp2 ( double  x ) | Y | Y |
|  double exp ( double  x ) | Y | Y |
|  double expm1 ( double  x ) | Y | Y |
|  double fabs ( double  x ) | Y | Y |
|  double fdim ( double  x, double  y ) | Y | Y |
|  double floor ( double  x ) | Y | Y |
|  double fma ( double  x, double  y, double  z ) | Y | Y |
|  double fmax ( double , double ) | Y | Y |
|  double fmin ( double  x, double  y ) | Y | Y |
|  double fmod ( double  x, double  y ) | Y | Y |
|  double frexp ( double  x, int* nptr ) | Y | Y |
|  double hypot ( double  x, double  y ) | Y | Y |
|  int ilogb ( double  x ) | Y | Y |
|  bool isfinite ( double  a ) | Y | Y |
|  bool isinf ( double  a ) | Y | Y |
|  bool isnan ( double  a ) | Y | Y |
|  double j0 ( double  x ) | Y | Y |
|  double j1 ( double  x ) | Y | Y |
|  double jn ( int  n, double  x ) | Y | Y |
|  double ldexp ( double  x, int  exp ) | Y | Y |
|  double lgamma(double x); | Y | Y |
|  double log10 ( double  x ) | Y | Y |
|  double log1p ( double  x ) | Y | Y |
|  double log2 ( double  x ) | Y | Y |
|  double logb ( double  x ) | Y | Y |
|  double log ( double  x ) | Y | Y |
|  long lrint(double x) { | Y | Y |
|  long lround(double x) { | Y | Y |
|  long llrint(double x) { return lrint(x); } | Y | Y |
|  long llround(double x) { return lround(x); } | Y | Y |
|  double max ( const double  a, const double  b ) | Y | Y |
|  double max ( const double  a, const float  b ) | Y | Y |
|  double max ( const float  a, const double  b ) | Y | Y |
|  double min ( const double  a, const double  b ) | Y | Y |
|  double min ( const double  a, const float  b ) | Y | Y |
|  double min ( const float  a, const double  b ) | Y | Y |
|  double modf ( double  x, double* iptr ) | Y | Y |
|  double nan ( const char* tagp ) | Y | Y |
|  double nearbyint ( double  x ) | Y | Y |
|  double nextafter ( double  x, double  y ) | Y | Y |
|  double norm3d ( double  a, double  b, double  c ) | Y | Y |
|  double norm4d ( double  a, double  b, double  c, double  d) | Y | Y |
|  double normcdf ( double  x ) | Y | Y |
|  double normcdfinv ( double  x ) | Y | Y |
|  double norm ( int  dim, const double* p ) | Y | Y |
|  double pow ( double  x, double  y ) | Y | Y |
|  double rcbrt ( double  x ) | Y | Y |
|  double remainder ( double  x, double  y ) | Y | Y |
|  double remquo ( double  x, double  y, int* quo ) | Y | Y |
|  double rhypot ( double  x, double  y ) | Y | Y |
|  double rint(double x); | Y | Y |
|  double rnorm3d(double a, double b, double c); | Y | Y |
|  double rnorm4d(double a, double b, double c, double d); | Y | Y |
|  double rnorm ( int  dim, const double* p ) | Y | Y |
|  double round(double x); | Y | Y |
|  double rsqrt ( double  x ) | Y | Y |
|  double scalbln ( double  x, long int  n ) | Y | Y |
|  double scalbn ( double  x, int  n ) | Y | Y |
|  bool signbit ( double  a ) | Y | Y |
|  void sincos ( double  x, double* sptr, double* cptr ) | Y | Y |
|  void sincospi ( double  x, double* sptr, double* cptr ) | Y | Y |
|  double sin ( double  x ) | Y | Y |
|  double sinh ( double  x ) | Y | Y |
|  double sinpi ( double  x ) | Y | Y |
|  double sqrt ( double  x ) | Y | Y |
|  double tan ( double  x ) | Y | Y |
|  double tanh ( double  x ) | Y | Y |
|  double tgamma ( double  x ) | Y | Y |
|  double trunc ( double  x ) | Y | Y |
|  double y0 ( double  x ) | Y | Y |
|  double y1 ( double  x ) | Y | Y |
|  double yn ( int  n, double  x ) | Y | Y |

## Single precision intrinsics

|   **CUDA**                                                |   **HIP**                         |  **chipStar**|
|-----------------------------------------------------------|-----------------------------------|:----------------:|
|  \_\_cosf ( float  x ) | Y | N |
|  \_\_sincosf ( float  x, float* sptr, float* cptr ) | N | Y |
|  \_\_exp10f ( float  x ) | N | N |
|  \_\_expf ( float  x ) | Y | N |
|  \_\_fadd_rd ( float  x, float  y ) | N | N |
|  \_\_fadd_rn ( float  x, float  y ) | N | N |
|  \_\_fadd_ru ( float  x, float  y ) | N | N |
|  \_\_fadd_rz ( float  x, float  y ) | N | N |
|  \_\_fdividef ( float  x, float  y ) | N | N |
|  \_\_fdiv_rd ( float  x, float  y ) | N | N |
|  \_\_fdiv_rn ( float  x, float  y ) | N | N |
|  \_\_fdiv_ru ( float  x, float  y ) | N | N |
|  \_\_fdiv_rz ( float  x, float  y ) | N | N |
|  \_\_fmaf_ieee_rd ( float  x, float  y, float  z ) | N | N |
|  \_\_fmaf_ieee_rn ( float  x, float  y, float  z ) | N | N |
|  \_\_fmaf_ieee_ru ( float  x, float  y, float  z ) | N | N |
|  \_\_fmaf_ieee_rz ( float  x, float  y, float  z ) | N | N |
|  \_\_fmaf_rd ( float  x, float  y, float  z ) | N | N |
|  \_\_fmaf_rn ( float  x, float  y, float  z ) | N | N |
|  \_\_fmaf_ru ( float  x, float  y, float  z ) | N | N |
|  \_\_fmaf_rz ( float  x, float  y, float  z ) | N | N |
|  \_\_fmul_rd ( float  x, float  y ) | N | N |
|  \_\_fmul_rn ( float  x, float  y ) | N | N |
|  \_\_fmul_ru ( float  x, float  y ) | N | N |
|  \_\_fmul_rz ( float  x, float  y ) | N | N |
|  \_\_frcp_rd ( float  x ) | N | N |
|  \_\_frcp_rn ( float  x ) | N | N |
|  \_\_frcp_ru ( float  x ) | N | N |
|  \_\_frcp_rz ( float  x ) | N | N |
|  \_\_frsqrt_rn ( float  x ) | Y | N |
|  \_\_fsqrt_rd ( float  x ) | N | N |
|  \_\_fsqrt_rn ( float  x ) | N | N |
|  \_\_fsqrt_ru ( float  x ) | N | N |
|  \_\_fsqrt_rz ( float  x ) | N | N |
|  \_\_fsub_rd ( float  x, float  y ) | N | N |
|  \_\_fsub_rn ( float  x, float  y ) | N | N |
|  \_\_fsub_ru ( float  x, float  y ) | N | N |
|  \_\_fsub_rz ( float  x, float  y ) | N | N |
|  \_\_log10f ( float  x ) | Y | N |
|  \_\_log2f ( float  x ) | Y | N |
|  \_\_logf ( float  x ) | Y | N |
|  \_\_powf ( float  x, float  y ) | Y | Y |
|  \_\_saturatef ( float  x ) | N | N |
|  \_\_sinf ( float  x ) | Y | N |
|  \_\_tanf ( float  x ) | Y | N |

## Single precision math library

|   **CUDA**                                                |   **HIP**                         |  **chipStar**|
|-----------------------------------------------------------|-----------------------------------|:----------------:|
| float acosf(float x) | Y | Y |
| float acoshf ( float  x ) | Y | Y |
| float asinf ( float  x ) | Y | Y |
| float asinhf ( float  x ) | Y | Y |
| float atan2f ( float  y, float  x ) | Y | Y |
| float atanf ( float  x ) | Y | Y |
| float atanhf ( float  x ) | Y | Y |
| float cbrtf ( float  x ) | Y | Y |
| float ceilf ( float  x ) | Y | Y |
| float copysignf ( float  x, float  y ) | Y | Y |
| long  convert_long(float x); | Y | Y |
| float cosf ( float  x ) | Y | Y |
| float coshf ( float  x ) | Y | Y |
| float cospif ( float  x ) | Y | Y |
| float cyl_bessel_i0f ( float  x ) | Y | Y |
| float cyl_bessel_i1f ( float  x ) | Y | Y |
| float erfcf ( float  x ) | Y | Y |
| float erfcinvf ( float  x ) | Y | Y |
| float erfcxf ( float  x ) | Y | Y |
| float erff ( float  x ) | Y | Y |
| float erfinvf ( float  x ) | Y | Y |
| float exp10f ( float  x ) | Y | Y |
| float exp2f ( float  x ) | Y | Y |
| float expf ( float  x ) | Y | Y |
| float expm1f ( float  x ) | Y | Y |
| float fabsf ( float  x ) | Y | Y |
| float fdimf ( float  x, float  y ) | Y | Y |
| float fdividef ( float  x, float  y ) | Y | Y |
| float floorf ( float  x ) | Y | Y |
| float fmaf ( float  x, float  y, float  z ) | Y | Y |
| float fmaxf ( float  x, float  y ) | Y | Y |
| float fminf ( float  x, float  y ) | Y | Y |
| float fmodf ( float  x, float  y ) | Y | Y |
| float frexpf ( float  x, int* nptr ) | Y | Y |
| float hypotf ( float  x, float  y ) | Y | Y |
| int   ilogbf ( float  x ) | Y | Y |
| bool  isfinite ( float  a ) | Y | Y |
| bool  isinf ( float  a ) | Y | Y |
| bool  isnan ( float  a ) | Y | Y |
| float j0f ( float  x ) | Y | Y |
| float j1f ( float  x ) | Y | Y |
| float jnf ( int  n, float  x ) | Y | Y |
| float ldexpf ( float  x, int  exp ) | Y | Y |
| float lgammaf(float x) { return (lgamma(x)); }; | Y | Y |
| float lgamma(float x); | Y | Y |
| float log10f ( float  x ) | Y | Y |
| float log1pf ( float  x ) | Y | Y |
| float log2f ( float  x ) | Y | Y |
| float logbf ( float  x ) | Y | Y |
| float logf ( float  x ) | Y | Y |
| long  lrintf(float x)                         OK | Y | Y |
| long  lroundf(float x)                        OK | Y | Y |
| long  llrintf(float x)                        OK | Y | Y |
| long  llroundf(float x)                       OK | Y | Y |
| float max ( const float  a, const float  b ) | Y | Y |
| float min ( const float  a, const float  b ) | Y | Y |
| float modff ( float  x, float* iptr ) | Y | Y |
| float nanf ( const char* tagp ) | Y | Y |
| float nearbyintf ( float  x ) | Y | Y |
| float nextafterf ( float  x, float  y ) | Y | Y |
| float norm3df ( float  a, float  b, float  c ) | Y | Y |
| float norm4df ( float  a, float  b, float  c, float  d ) | Y | Y |
| float normcdff ( float  x ) | Y | Y |
| float normcdfinvf ( float  x ) | Y | Y |
| float normf ( int  dim, const float* p ) | Y | Y |
| float powf ( float  x, float  y ) | Y | Y |
| float rcbrtf ( float  x ) | Y | Y |
| float remainderf ( float  x, float  y ) | Y | Y |
| float remquof ( float  x, float  y, int* quo ) | Y | Y |
| float rhypotf ( float  x, float  y ) | Y | Y |
| float rintf(float x) { return rint(x); } | Y | Y |
| float rint(float x); | Y | Y |
| float rnorm3df(float a, float b, float c); | Y | Y |
| float rnorm4df(float a, float b, float c, float d); | Y | Y |
| float rnormf ( int  dim, const float* p ) | Y | Y |
| float roundf(float x) { return round(x); } | Y | Y |
| float round(float x); | Y | Y |
| float rsqrtf ( float  x ) | Y | Y |
| float scalblnf ( float  x, long int  n ) | Y | Y |
| float scalbnf ( float  x, int  n ) | Y | Y |
| bool  signbit ( float  a ) | Y | Y |
| float sinf ( float  x ) | Y | Y |
| float sinhf ( float  x ) | Y | Y |
| float sinpif ( float  x ) | Y | Y |
| float sqrtf ( float  x ) | Y | Y |
| float tanf ( float  x ) | Y | Y |
| float tanhf ( float  x ) | Y | Y |
| float tgammaf ( float  x ) | Y | Y |
| float truncf ( float  x ) | Y | Y |
| float y0f ( float  x ) | Y | Y |
| float y1f ( float  x ) | Y | Y |
| float ynf ( int  n, float  x ) | Y | Y |
| void sincosf ( float  x, float* sptr, float* cptr ) | Y | Y |
| void sincospif ( float  x, float* sptr, float* cptr ) | Y | Y |

## Half precision intrinsics + math library

|   **CUDA**                                                |   **HIP**                         |  **chipStar**|
|-----------------------------------------------------------|-----------------------------------|:----------------:|
|  \_\_device\_\_ \_\_half \_\_habs | Y | Y |
|  \_\_device\_\_ \_\_half \_\_hadd | Y | Y |
|  \_\_device\_\_ \_\_half \_\_hadd_rn | Y | N |
|  \_\_device\_\_ \_\_half \_\_hadd_sat | Y | Y |
|  \_\_device\_\_ \_\_half \_\_hdiv | Y | Y |
|  \_\_device\_\_ \_\_half \_\_hfma | Y | Y |
|  \_\_device\_\_ \_\_half \_\_hfma_relu | Y | N |
|  \_\_device\_\_ \_\_half \_\_hfma_sat | Y | Y |
|  \_\_device\_\_ \_\_half \_\_hmul | Y | Y |
|  \_\_device\_\_ \_\_half \_\_hmul_rn | Y | N |
|  \_\_device\_\_ \_\_half \_\_hmul_sat | Y | Y |
|  \_\_device\_\_ \_\_half \_\_hneg | Y | Y |
|  \_\_device\_\_ \_\_half \_\_hsub | Y | Y |
|  \_\_device\_\_ \_\_half \_\_hsub_rn | Y | N |
|  \_\_device\_\_ \_\_half \_\_hsub_sat | Y | Y |
|  \_\_device\_\_ \_\_half atomicAdd | Y | N |
|  \_\_device\_\_ bool \_\_heq | Y | Y |
|  \_\_device\_\_ bool \_\_hequ | Y | Y |
|  \_\_device\_\_ bool \_\_hge | Y | Y |
|  \_\_device\_\_ bool \_\_hgeu | Y | Y |
|  \_\_device\_\_ bool \_\_hgt | Y | Y |
|  \_\_device\_\_ bool \_\_hgtu | Y | Y |
|  \_\_device\_\_ bool \_\_hne | Y | Y |
|  \_\_device\_\_ bool \_\_hneu | Y | Y |
|  \_\_device\_\_ bool \_\_hle | Y | Y |
|  \_\_device\_\_ bool \_\_hleu | Y | Y |
|  \_\_device\_\_ bool \_\_hlt | Y | Y |
|  \_\_device\_\_ bool \_\_hltu | Y | Y |
|  \_\_device\_\_ int \_\_hisinf | Y | Y |
|  \_\_device\_\_ bool \_\_hisnan | Y | Y |
|  \_\_device\_\_ \_\_half \_\_hmax | Y | Y |
|  \_\_device\_\_ \_\_half \_\_hmax_nan | Y | N |
|  \_\_device\_\_ \_\_half \_\_hmin | Y | Y |
|  \_\_device\_\_ \_\_half \_\_hmin_nan | Y | N |
|  \_\_host\_\_   \_\_device\_\_    \_\_half  \_\_double2half | Y | N |
|  \_\_host\_\_   \_\_device\_\_    \_\_half  \_\_float2half | Y | Y |
|  \_\_host\_\_   \_\_device\_\_    \_\_half  \_\_float2half_rd | Y | Y |
|  \_\_host\_\_   \_\_device\_\_    \_\_half  \_\_float2half_rn | Y | Y |
|  \_\_host\_\_   \_\_device\_\_    \_\_half  \_\_float2half_ru | Y | Y |
|  \_\_host\_\_   \_\_device\_\_    \_\_half  \_\_float2half_rz | Y | Y |
|  \_\_device\_\_    short int \_\_half_as_short | Y | Y |
|  \_\_device\_\_    unsigned short int   \_\_half_as_ushort | Y | Y |
|  \_\_device\_\_    \_\_half \_\_int2half_rd | Y | Y |
|  \_\_host\_\_   \_\_device\_\_    \_\_half  \_\_int2half_rn | Y | Y |
|  \_\_device\_\_    \_\_half \_\_int2half_ru | Y | Y |
|  \_\_device\_\_    \_\_half \_\_int2half_rz | Y | Y |
|  \_\_device\_\_    \_\_half \_\_ldca | Y | Y |
|  \_\_device\_\_    \_\_half \_\_ldcs | Y | Y |
|  \_\_device\_\_    \_\_half \_\_ldcv | Y | N |
|  \_\_device\_\_    \_\_half \_\_ldg | Y | Y |
|  \_\_device\_\_    \_\_half \_\_ldlu | Y | N |
|  \_\_device\_\_    \_\_half \_\_ll2half_rd | Y | Y |
|  \_\_host\_\_   \_\_device\_\_    \_\_half  \_\_ll2half_rn | Y | Y |
|  \_\_device\_\_    \_\_half \_\_ll2half_ru | Y | Y |
|  \_\_device\_\_    \_\_half \_\_ll2half_rz | Y | Y |
|  \_\_device\_\_    \_\_half \_\_shfl_down_sync | Y | N |
|  \_\_device\_\_    \_\_half \_\_shfl_sync | Y | N |
|  \_\_device\_\_    \_\_half \_\_shfl_up_sync | Y | N |
|  \_\_device\_\_    \_\_half \_\_shfl_xor_sync | Y | N |
|  \_\_device\_\_    \_\_half \_\_short2half_rd | Y | Y |
|  \_\_host\_\_   \_\_device\_\_    \_\_half  \_\_short2half_rn | Y | Y |
|  \_\_device\_\_    \_\_half \_\_short2half_ru | Y | Y |
|  \_\_device\_\_    \_\_half \_\_short2half_rz | Y | Y |
|  \_\_device\_\_    \_\_half \_\_short_as_half | Y | Y |
|  \_\_device\_\_    void \_\_stcg | Y | Y |
|  \_\_device\_\_    void \_\_stcs | Y | Y |
|  \_\_device\_\_    void \_\_stwb | Y | N |
|  \_\_device\_\_    void \_\_stwt | Y | N |
|  \_\_device\_\_    \_\_half \_\_uint2half_rd | Y | Y |
|  \_\_host\_\_   \_\_device\_\_    \_\_half  \_\_uint2half_rn | Y | Y |
|  \_\_device\_\_    \_\_half \_\_uint2half_ru | Y | Y |
|  \_\_device\_\_    \_\_half \_\_uint2half_rz | Y | Y |
|  \_\_device\_\_    \_\_half \_\_ull2half_rd | Y | Y |
|  \_\_host\_\_   \_\_device\_\_    \_\_half  \_\_ull2half_rn | Y | Y |
|  \_\_device\_\_    \_\_half \_\_ull2half_ru | Y | Y |
|  \_\_device\_\_    \_\_half \_\_ull2half_rz | Y | Y |
|  \_\_device\_\_    \_\_half \_\_ushort2half_rd | Y | Y |
|  \_\_host\_\_   \_\_device\_\_    \_\_half  \_\_ushort2half_rn | Y | Y |
|  \_\_device\_\_    \_\_half \_\_ushort2half_ru | Y | Y |
|  \_\_device\_\_    \_\_half \_\_ushort2half_rz | Y | Y |
|  \_\_device\_\_    \_\_half \_\_ushort_as_half | Y | Y |
| \_\_device\_\_ \_\_half hceil | Y | Y |
| \_\_device\_\_ \_\_half hcos | Y | Y |
| \_\_device\_\_ \_\_half hexp | Y | Y |
| \_\_device\_\_ \_\_half hexp10 | Y | Y |
| \_\_device\_\_ \_\_half hexp2 | Y | Y |
| \_\_device\_\_ \_\_half hfloor | Y | Y |
| \_\_device\_\_ \_\_half hlog | Y | Y |
| \_\_device\_\_ \_\_half hlog10 | Y | Y |
| \_\_device\_\_ \_\_half hlog2 | Y | Y |
| \_\_device\_\_ \_\_half hrcp | Y | Y |
| \_\_device\_\_ \_\_half hrint | Y | Y |
| \_\_device\_\_ \_\_half hrsqrt | Y | Y |
| \_\_device\_\_ \_\_half hsin | Y | Y |
| \_\_device\_\_ \_\_half hsqrt | Y | Y |
| \_\_device\_\_ \_\_half htrunc | Y | Y |

## Half2 precision intrinsics + math library

|   **CUDA**                                                |   **HIP**                         |  **chipStar**|
|-----------------------------------------------------------|-----------------------------------|:----------------:|
|  \_\_device\_\_    \_\_half2 \_\_h2div | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_habs2 | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_hadd2 | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_hadd2_rn | Y | N |
|  \_\_device\_\_    \_\_half2 \_\_hadd2_sat | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_hcmadd | Y | N |
|  \_\_device\_\_    \_\_half2 \_\_hfma2 | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_hfma2_relu | Y | N |
|  \_\_device\_\_    \_\_half2 \_\_hfma2_sat | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_hmul2 | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_hmul2_rn | Y | N |
|  \_\_device\_\_    \_\_half2 \_\_hmul2_sat | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_hneg2 | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_hsub2 | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_hsub2_rn | Y | N |
|  \_\_device\_\_    \_\_half2 \_\_hsub2_sat | Y | Y |
|  \_\_device\_\_    \_\_half2 atomicAdd | Y | Y |
|  \_\_device\_\_    bool \_\_hbeq2  | Y | Y |
|  \_\_device\_\_    bool \_\_hbequ2 | Y | Y |
|  \_\_device\_\_    bool \_\_hbge2  | Y | Y |
|  \_\_device\_\_    bool \_\_hbgeu2 | Y | Y |
|  \_\_device\_\_    bool \_\_hbgt2  | Y | Y |
|  \_\_device\_\_    bool \_\_hbgtu2 | Y | Y |
|  \_\_device\_\_    bool \_\_hble2  | Y | Y |
|  \_\_device\_\_    bool \_\_hbleu2 | Y | Y |
|  \_\_device\_\_    bool \_\_hblt2  | Y | Y |
|  \_\_device\_\_    bool \_\_hbltu2 | Y | Y |
|  \_\_device\_\_    bool \_\_hbne2  | Y | Y |
|  \_\_device\_\_    bool \_\_hbneu2 | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_heq2  | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_hequ2 | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_hge2  | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_hgeu2 | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_hgt2  | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_hgtu2 | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_hle2  | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_hleu2 | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_hlt2  | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_hltu2 | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_hne2  | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_hneu2 | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_hisnan2 | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_hmax2 | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_hmax2_nan | Y | N |
|  \_\_device\_\_    \_\_half2 \_\_hmin2 | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_hmin2_nan | Y | N |
| \_\_device\_\_ \_\_half2 h2ceil | Y | Y |
| \_\_device\_\_ \_\_half2 h2cos | Y | Y |
| \_\_device\_\_ \_\_half2 h2exp | Y | Y |
| \_\_device\_\_ \_\_half2 h2exp10 | Y | Y |
| \_\_device\_\_ \_\_half2 h2exp2 | Y | Y |
| \_\_device\_\_ \_\_half2 h2floor | Y | Y |
| \_\_device\_\_ \_\_half2 h2log | Y | Y |
| \_\_device\_\_ \_\_half2 h2log10 | Y | Y |
| \_\_device\_\_ \_\_half2 h2log2 | Y | Y |
| \_\_device\_\_ \_\_half2 h2rcp | Y | Y |
| \_\_device\_\_ \_\_half2 h2rint | Y | Y |
| \_\_device\_\_ \_\_half2 h2rsqrt | Y | Y |
| \_\_device\_\_ \_\_half2 h2sin | Y | Y |
| \_\_device\_\_ \_\_half2 h2sqrt | Y | Y |
| \_\_device\_\_ \_\_half2 h2trunc | Y | Y |
|  \_\_host\_\_   \_\_device\_\_    \_\_half2   \_\_float22half2_rn | Y | Y |
|  \_\_host\_\_   \_\_device\_\_    \_\_half2   \_\_float2half2_rn | Y | Y |
|  \_\_host\_\_   \_\_device\_\_    \_\_half2   \_\_floats2half2_rn | Y | Y |
|  \_\_host\_\_   \_\_device\_\_    float2  \_\_half22float2 | Y | Y |
|  \_\_host\_\_   \_\_device\_\_    float   \_\_half2float | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_half2half2 | Y | Y |
|  \_\_device\_\_    int \_\_half2int_rd | Y | Y |
|  \_\_device\_\_    int \_\_half2int_rn | Y | Y |
|  \_\_device\_\_    int \_\_half2int_ru | Y | Y |
|  \_\_host\_\_   \_\_device\_\_    int   \_\_half2int_rz | Y | Y |
|  \_\_device\_\_    long long int  \_\_half2ll_rd | Y | Y |
|  \_\_device\_\_    long long int  \_\_half2ll_rn | Y | Y |
|  \_\_device\_\_    long long int  \_\_half2ll_ru | Y | Y |
|  \_\_host\_\_   \_\_device\_\_    long long int   \_\_half2ll_rz | Y | Y |
|  \_\_device\_\_    short int \_\_half2short_rd | Y | Y |
|  \_\_device\_\_    short int \_\_half2short_rn | Y | Y |
|  \_\_device\_\_    short int \_\_half2short_ru | Y | Y |
|  \_\_host\_\_   \_\_device\_\_    short int   \_\_half2short_rz | Y | Y |
|  \_\_device\_\_    unsigned int   \_\_half2uint_rd | Y | Y |
|  \_\_device\_\_    unsigned int   \_\_half2uint_rn | Y | Y |
|  \_\_device\_\_    unsigned int   \_\_half2uint_ru | Y | Y |
|  \_\_host\_\_   \_\_device\_\_    unsigned int  \_\_half2uint_rz | Y | Y |
|  \_\_device\_\_    unsigned long long int   \_\_half2ull_rd | Y | Y |
|  \_\_device\_\_    unsigned long long int   \_\_half2ull_rn | Y | Y |
|  \_\_device\_\_    unsigned long long int   \_\_half2ull_ru | Y | Y |
|  \_\_host\_\_   \_\_device\_\_    unsigned long long int  \_\_half2ull_rz | Y | Y |
|  \_\_device\_\_    unsigned short int   \_\_half2ushort_rd | Y | Y |
|  \_\_device\_\_    unsigned short int   \_\_half2ushort_rn | Y | Y |
|  \_\_device\_\_    unsigned short int   \_\_half2ushort_ru | Y | Y |
|  \_\_host\_\_   \_\_device\_\_    unsigned short int  \_\_half2ushort_rz | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_halves2half2 | Y | Y |
|  \_\_host\_\_   \_\_device\_\_    float   \_\_high2float | Y | Y |
|  \_\_device\_\_    \_\_half \_\_high2half | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_high2half2 | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_highs2half2 | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_ldca | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_ldcg | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_ldcs | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_ldcv | Y | N |
|  \_\_device\_\_    \_\_half2 \_\_ldg | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_ldlu | Y | N |
|  \_\_host\_\_   \_\_device\_\_    float   \_\_low2float | Y | Y |
|  \_\_device\_\_    \_\_half \_\_low2half | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_low2half2 | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_lowhigh2highlow | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_lows2half2 | Y | Y |
|  \_\_device\_\_    \_\_half2 \_\_shfl_down_sync | Y | N |
|  \_\_device\_\_    \_\_half2 \_\_shfl_sync | Y | N |
|  \_\_device\_\_    \_\_half2 \_\_shfl_up_sync | Y | N |
|  \_\_device\_\_    \_\_half2 \_\_shfl_xor_sync | Y | N |
|  \_\_device\_\_    void \_\_stcg | Y | Y |
|  \_\_device\_\_    void \_\_stcs | Y | Y |
|  \_\_device\_\_    void \_\_stwb | Y | N |
|  \_\_device\_\_    void \_\_stwt | Y | N |
