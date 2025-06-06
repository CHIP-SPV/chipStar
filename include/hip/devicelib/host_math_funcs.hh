#ifndef HIP_INCLUDE_DEVICELIB_HOST_MATH_FUNCS_H
#define HIP_INCLUDE_DEVICELIB_HOST_MATH_FUNCS_H

#include <hip/devicelib/macros.hh>
#include <algorithm>

extern "C" {
    extern float __ocml_cospi_f32(float x);
    extern float __ocml_sinpi_f32(float x);
    extern double __ocml_erfcinv_f64(double x);
    extern float __ocml_erfcinv_f32(float x);
    extern double __ocml_erfcx_f64(double x);
    extern float __ocml_erfcx_f32(float x);
    extern double __ocml_erfinv_f64(double x);
    extern float __ocml_erfinv_f32(float x);
    extern double __ocml_ncdf_f64(double x);
    extern float __ocml_ncdf_f32(float x);
    extern double __ocml_ncdfinv_f64(double x);
    extern float __ocml_ncdfinv_f32(float x);
    extern double __ocml_rcbrt_f64(double x);
    extern float __ocml_rcbrt_f32(float x);
    extern double __ocml_sincospi_f64(double x, double* pcos);
    extern float __ocml_sincospi_f32(float x, float* pcos);
    extern int __ocml_signbit_f64(double x);
    extern double __ocml_rsqrt_f64(double x);
    extern float __ocml_rsqrt_f32(float x);
}

// Trigonometric functions
inline float cospi(float x) { return __ocml_cospi_f32(x); }
inline float cospif(float x) { return __ocml_cospi_f32(x); }

inline float sinpi(float x) { return __ocml_sinpi_f32(x); }
inline float sinpif(float x) { return __ocml_sinpi_f32(x); }

// Error functions
inline double erfcinv(double x) { return __ocml_erfcinv_f64(x); }

inline float erfcinvf(float x) { return __ocml_erfcinv_f32(x); }

inline double erfcx(double x) { return __ocml_erfcx_f64(x); }

inline float erfcxf(float x) { return __ocml_erfcx_f32(x); }

inline double erfinv(double x) { return __ocml_erfinv_f64(x); }

inline float erfinvf(float x) { return __ocml_erfinv_f32(x); }

// Normal distribution functions
inline double normcdf(double x) { return __ocml_ncdf_f64(x); }

inline float normcdff(float x) { return __ocml_ncdf_f32(x); }

inline double normcdfinv(double x) { return __ocml_ncdfinv_f64(x); }

inline float normcdfinvf(float x) { return __ocml_ncdfinv_f32(x); }

// Reciprocal cube root
inline double rcbrt(double x) { return __ocml_rcbrt_f64(x); }

inline float rcbrtf(float x) { return __ocml_rcbrt_f32(x); }

// Sine and cosine of pi times x
inline void sincospi(double x, double* psin, double* pcos) { 
    *psin = __ocml_sincospi_f64(x, pcos);
}

inline void sincospif(float x, float* psin, float* pcos) { 
    *psin = __ocml_sincospi_f32(x, pcos);
}

// Sign bit
inline int signbit(double x) { return __ocml_signbit_f64(x); }

inline double rsqrt(double x) { return __ocml_rsqrt_f64(x); }

inline float rsqrtf(float x) { return __ocml_rsqrt_f32(x); }


#endif // HIP_INCLUDE_DEVICELIB_HOST_MATH_FUNCS_H