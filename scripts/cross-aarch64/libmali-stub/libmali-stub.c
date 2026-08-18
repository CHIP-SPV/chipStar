/* Link-time stand-in for salami's OpenCL implementation.
 *
 * On salami libOpenCL.so is a symlink chain ending at libmali.so.0.40.0,
 * whose SONAME is libmali.so.0, so every chipStar binary records
 * NEEDED libmali.so.0 rather than libOpenCL.so.1. Shared objects tolerate
 * undefined cl* symbols, but the host tools and a few samples/tests are
 * executables that call the API directly, and those need every symbol
 * defined at link time. So the stub defines each cl* the real driver
 * exports (cl-symbols.txt, taken with `nm -D --defined-only` from
 * libmali.so.0 on salami) as an empty function. Nothing here ever runs:
 * on the target the dynamic loader binds to the real driver. Built with
 * -Wl,-soname,libmali.so.0 and installed as libOpenCL.so so
 * find_library(OpenCL) picks it up. Never shipped.
 */
#define STUB(name) void name(void) {}
#include "cl-symbols.h"
