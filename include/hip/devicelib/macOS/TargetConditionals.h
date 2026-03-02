/*
 * Stub TargetConditionals.h for SPIR-V device compilation on macOS.
 *
 * The real macOS TargetConditionals.h uses __is_target_arch() which fails
 * for the spirv64 target, causing a compilation error. This stub provides
 * safe defaults so device code that transitively includes
 * TargetConditionals.h (e.g., via Catch2) can compile.
 */
#ifndef __TARGETCONDITIONALS_SPIRV_STUB__
#define __TARGETCONDITIONALS_SPIRV_STUB__

#define TARGET_OS_MAC     1
#define TARGET_OS_OSX     1
#define TARGET_OS_IPHONE  0
#define TARGET_OS_IOS     0
#define TARGET_OS_WATCH   0
#define TARGET_OS_TV      0
#define TARGET_OS_VISION  0
#define TARGET_OS_SIMULATOR 0
#define TARGET_OS_EMBEDDED  0
#define TARGET_CPU_ARM64  1
#define TARGET_CPU_ARM    0
#define TARGET_CPU_X86_64 0
#define TARGET_CPU_X86    0
#define TARGET_RT_64_BIT  1
#define TARGET_RT_LITTLE_ENDIAN 1
#define TARGET_RT_BIG_ENDIAN    0

#endif /* __TARGETCONDITIONALS_SPIRV_STUB__ */
