// chipStar must ship a rocm_version.h. Applications probe ROCM_VERSION_* to
// gate on HIP-implementation capabilities, and without the header they fail to
// compile outright rather than degrading gracefully.
//
// This mirrors the exact pattern Trilinos STK uses in
// packages/stk/stk_ngp_test/stk_ngp_test/GlobalReporter.hpp: try the rocm-core
// path first, fall back to the include root. Both must resolve.
//
// Deliberately does NOT compare against HIP_VERSION_*. Both headers are
// generated from the same CMake variables in adjacent lines of CMakeLists.txt,
// so they cannot drift by construction, while <hip/hip_version.h> is ambiguous
// at build time: the bundled HIP submodule ships its own copy which shadows the
// generated one depending on -I ordering, so such a comparison tests include
// order rather than this feature.
#include <hip/hip_runtime.h>

#if __has_include(<rocm-core/rocm_version.h>)
#include <rocm-core/rocm_version.h>
#else
#error "<rocm-core/rocm_version.h> not found"
#endif

// Re-check the bare form independently; STK falls back to it.
#if !__has_include(<rocm_version.h>)
#error "<rocm_version.h> not found"
#endif

#if !defined(ROCM_VERSION_MAJOR) || !defined(ROCM_VERSION_MINOR) || \
    !defined(ROCM_VERSION_PATCH)
#error "ROCM_VERSION_MAJOR/MINOR/PATCH not defined"
#endif

// The capability gate STK actually writes. chipStar supports registering device
// static globals without RDC, so this branch must not fire.
#if (ROCM_VERSION_MAJOR < 6) || (ROCM_VERSION_MAJOR == 6 && ROCM_VERSION_MINOR < 3)
#error "reported ROCm level would force apps down the RDC-required path"
#endif
