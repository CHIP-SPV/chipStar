#!/bin/bash
# Cross-compile chipStar and its unit tests for salami (aarch64-linux-gnu,
# Ubuntu 22.04, Mali-G52) on an x86_64 host, inside the image built from the
# Dockerfile next to this script.
#
# chipStar builds hipcc.bin during its own build and then uses it as the C++
# compiler for every Catch2 test. In a cross build that would be an aarch64
# binary that cannot run on the x86 builder, so:
#   1. build chipStar natively for x86 first, only to obtain an x86 hipcc.bin
#      and hipconfig.bin;
#   2. cross-build chipStar + build_tests for aarch64, then overwrite
#      bin/hipcc.bin in that tree with the x86 one from (1).
# hipcc locates everything else via the .hipInfo beside it, which pass 2
# generated for the aarch64 tree, so it links against the aarch64 libCHIP
# while itself running on x86. The pass plugin is loaded by opt on the
# builder and stays x86 throughout.
#
# The x86-hosted clang used for both passes was built with AArch64 enabled
# by image-llvm.sh; the runners' own LLVM 23 cannot emit aarch64.
#
# The only non-stock library the aarch64 link needs is libmali.so.0, salami's
# OpenCL implementation, satisfied by the empty stub from libmali-stub/.
#
# Usage (inside the container):  cross-chipstar.sh <tree> [jobs]
# Environment:
#   CHIPSTAR_DIR   chipStar checkout                 (default /chipstar)
#   WORK_DIR       scratch                           (default /work)
#   X86_LLVM       x86-hosted LLVM 23 with AArch64   (default /opt/llvm-x86-aarch64)
#   STAGE_PREFIX   install prefix ON SALAMI, must be identical to where the
#                  tree will be rsync'd: it is baked into RPATH and .hipInfo
#                  (default /home/pvelesko/ci-stage/<tree>)
#   CROSS_IMAGE    image ID or tag, part of the stamp that forces a clean rebuild
# Output: $WORK_DIR/cross-<tree>, ready for ship-to-salami.sh.
set -e
SHA="${1:?usage: cross-chipstar.sh <tree> [jobs]}"
JOBS="${2:-$(nproc)}"
CHIPSTAR_DIR="${CHIPSTAR_DIR:-/chipstar}"
WORK_DIR="${WORK_DIR:-/work}"
X86_LLVM="${X86_LLVM:-/opt/llvm-x86-aarch64}"
STAGE_PREFIX="${STAGE_PREFIX:-/home/pvelesko/ci-stage/$SHA}"
HERE="$(dirname "$(readlink -f "$0")")"

echo "sha=$SHA jobs=$JOBS chipstar=$CHIPSTAR_DIR work=$WORK_DIR llvm=$X86_LLVM prefix=$STAGE_PREFIX"
git config --global --add safe.directory '*' || true
"$X86_LLVM/bin/llvm-config" --targets-built | grep -q AArch64 \
  || { echo "ERROR: $X86_LLVM cannot emit AArch64"; exit 1; }

# The checkout is bind-mounted read-only; build out of tree.
SRC="$WORK_DIR/src-$SHA"
rm -rf "$SRC"; cp -a "$CHIPSTAR_DIR" "$SRC"

# ninja cannot see the compiler image, a header that newly shadows another, nor a
# dropped -D, toolchain *_INIT or CMake default, which only seed a fresh cache;
# any change wipes both builds.
STAMP=$(set -eo pipefail; { echo "${CROSS_IMAGE:-}"; cat "$HERE/cross-chipstar.sh" "$HERE/chipstar-aarch64-toolchain.cmake"
  git -C "$SRC" ls-files --recurse-submodules | grep -E '(\.(h|hh|hpp|cuh|inc)|(^|/)[^./]+)$'
  git -C "$SRC" ls-files -z --recurse-submodules -- '*CMakeLists.txt' '*.cmake' | (cd "$SRC" && xargs -0 cat); } | sha256sum)
[ "$(cat "$WORK_DIR/stamp-$SHA" 2>/dev/null)" = "$STAMP" ] || rm -rf "$WORK_DIR/native-$SHA" "$WORK_DIR/cross-$SHA" "$WORK_DIR/stamp-$SHA"

# --- pass 1: native x86 build of the tools the aarch64 build must RUN -----
# chipStar builds four things and then executes them during its own build:
#   hipcc.bin / hipconfig.bin   the C++ compiler for every Catch2 test
#   libLLVMHipSpvPasses.so      the pass plugin, dlopen'ed by opt inside clang
#   prepare-builtins            run on every ROCm-Device-Libs bitcode library
# Cross-built they would be aarch64 and unrunnable on the x86 builder, so
# configure a native chipStar here purely to build those four targets. It
# needs an OpenCL library to configure at all; the same empty stub serves,
# compiled for x86 this time. Nothing from this tree is shipped.
NATIVE="$WORK_DIR/native-$SHA"
"$HERE/libmali-stub/make-stub.sh" "$WORK_DIR/x86-stub" gcc
cmake -S "$SRC" -B "$NATIVE" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_CONFIG_BIN="$X86_LLVM/bin/llvm-config" \
  -DCHIP_LLVM_USE_INTERGRATED_SPIRV=ON \
  -DCHIP_BUILD_SAMPLES=OFF -DCHIP_BUILD_TESTS=OFF \
  -DCHIP_ATOMICS_CACHE_BYPASS_WORKAROUND=ON \
  -DOpenCL_LIBRARY="$WORK_DIR/x86-stub/libOpenCL.so" \
  -DOpenCL_INCLUDE_DIR="$SRC/include" \
  -DCMAKE_PREFIX_PATH=/opt/spirv-tools-aarch64
ninja -C "$NATIVE" -j"$JOBS" hipcc.bin hipconfig.bin LLVMHipPasses prepare-builtins

# --- pass 2: aarch64 chipStar + tests -----------------------------------
"$HERE/libmali-stub/make-stub.sh" "$WORK_DIR/mali-stub"
CROSS="$WORK_DIR/cross-$SHA"
# Reused build dir: the stamp above wipes it when a fresh configure is needed.
rm -rf "$WORK_DIR/stage"
cmake -S "$SRC" -B "$CROSS" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$HERE/chipstar-aarch64-toolchain.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$STAGE_PREFIX" \
  -DCMAKE_INSTALL_RPATH="$STAGE_PREFIX/lib" \
  -DLLVM_CONFIG_BIN="$X86_LLVM/bin/llvm-config" \
  -DCHIP_LLVM_USE_INTERGRATED_SPIRV="${INTEGRATED_SPIRV:-ON}" \
  -DCHIP_MALI_GPU_WORKAROUNDS=ON \
  -DCHIP_ATOMICS_CACHE_BYPASS_WORKAROUND=ON \
  -DCHIP_SKIP_TESTS_WITH_DOUBLES=ON \
  -DOpenCL_LIBRARY="$WORK_DIR/mali-stub/libOpenCL.so" \
  -DOpenCL_INCLUDE_DIR="$SRC/include" \
  -DPREPARE_BUILTINS="$NATIVE/bitcode/ROCm-Device-Libs/utils/prepare-builtins/prepare-builtins" \
  -DCMAKE_PREFIX_PATH=/opt/spirv-tools-aarch64
# CMAKE_PREFIX_PATH lets find_package(SPIRV-Tools) find the aarch64 build
# baked into the image, so chipStar's ExternalProject fallback (which
# hardcodes gcc/g++ and would emit x86 archives) never runs.
# hipcc's own compile and link lines are assembled from .hipInfo, not from
# CMake's toolchain, so a test that hipcc links directly (custom commands in
# tests/runtime/CMakeLists.txt) would otherwise reach for the host's
# /usr/bin/ld. hipcc honours these two environment hooks
# (HIPCC/src/hipBin_base.h) for exactly this: they are the standard way to
# retarget hipcc, ROCm included.
export HIPCC_COMPILE_FLAGS_APPEND="--target=aarch64-linux-gnu --sysroot=/ --gcc-toolchain=/usr"
export HIPCC_LINK_FLAGS_APPEND="--target=aarch64-linux-gnu --sysroot=/ --gcc-toolchain=/usr -B/usr/aarch64-linux-gnu/bin -fuse-ld=/usr/bin/aarch64-linux-gnu-ld -L/opt/spirv-tools-aarch64/lib"
# Build the runtime first so .hipInfo and bin/ exist, then swap in the x86
# hipcc before anything tries to compile a test with it.
ninja -C "$CROSS" -j"$JOBS" CHIP hipcc.bin hipconfig.bin devicelib_bc LLVMHipPasses
# Copy only when different so an unchanged tool keeps its mtime.
for f in hipcc.bin hipconfig.bin; do cmp -s "$NATIVE/bin/$f" "$CROSS/bin/$f" || cp -f "$NATIVE/bin/$f" "$CROSS/bin/$f"; done
# hipcc drops HIPCC_LINK_FLAGS_APPEND on its -no-hip-rt link path: the
# no-hip-rt copy of the link flags is taken (hipBin_spirv.h:845) before the
# append is applied (:878), so a test that hipcc links directly still
# reaches for the host's /usr/bin/ld. Until that is fixed in HIPCC, wrap
# hipcc so the retarget flags are on every invocation regardless of path.
# bin/hipcc is a symlink to hipcc.bin, so write hipcc.wrap and mv it over the link.
cat > "$CROSS/bin/hipcc.wrap" <<WRAP
#!/bin/sh
exec "$CROSS/bin/hipcc.bin" "\$@" $HIPCC_LINK_FLAGS_APPEND
WRAP
chmod +x "$CROSS/bin/hipcc.wrap"
cmp -s "$CROSS/bin/hipcc.wrap" "$CROSS/bin/hipcc" && rm "$CROSS/bin/hipcc.wrap" || mv -f "$CROSS/bin/hipcc.wrap" "$CROSS/bin/hipcc"
# The plugin that compiles every test is this x86 one, so the build-time
# lowering choice must be made in the NATIVE configure above; the flag on the
# cross configure only affects the aarch64 plugin that is overwritten here.
# Getting this wrong shipped SPV_INTEL_cache_controls modules to Mali, which
# rejects the OpExtension at clCreateProgramWithIL with CL_INVALID_VALUE.
cmp -s "$NATIVE/lib/libLLVMHipSpvPasses.so" "$CROSS/lib/libLLVMHipSpvPasses.so" \
  || cp -f "$NATIVE/lib/libLLVMHipSpvPasses.so" "$CROSS/lib/libLLVMHipSpvPasses.so"
ninja -C "$CROSS" -j"$JOBS" all
ninja -C "$CROSS" -j"$JOBS" build_tests

arch() { aarch64-linux-gnu-readelf -h "$1" 2>/dev/null | awk '/Machine:/{print $2}'; }
[ "$(arch "$CROSS/libCHIP.so")" = AArch64 ]  || { echo "ERROR: libCHIP is not aarch64"; exit 1; }
[ "$(arch "$CROSS/bin/hipcc.bin")" = Advanced ]  || { echo "ERROR: hipcc.bin is not x86"; exit 1; }
[ "$(arch "$CROSS/lib/libLLVMHipSpvPasses.so")" = Advanced ] || { echo "ERROR: pass plugin is not x86"; exit 1; }
[ "$(arch "$CROSS/tests/runtime/TestKernelArgs")" = AArch64 ] || { echo "ERROR: test binaries are not aarch64"; exit 1; }
echo "$STAMP" > "$WORK_DIR/stamp-$SHA"
echo "CROSS-CHIPSTAR-OK build=$CROSS"
