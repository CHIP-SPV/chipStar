#!/bin/bash
# Build the x86-hosted LLVM 23 that cross-chipstar.sh compiles with.
#
# The LLVM 23 on the runners cannot emit aarch64 (targets-built "SPIRV X86"),
# and the aarch64-hosted one from cross-build.sh runs only on salami. So the
# chipStar cross build needs a third toolchain: x86-hosted, targets
# X86 + AArch64 + SPIRV, carrying the same llvm-patches/llvm-23 series so
# device code is identical to what the runners emit. Runs inside the base
# image; the result lands in $WORK_DIR/x86-llvm and is committed into the
# final image by the Dockerfile's second stage.
#
# Sources come from configure_llvm.sh --source-only, exactly as
# cross-build.sh does, so the pinned refs and patch series stay in one place.
set -e
JOBS="${1:-$(nproc)}"
CHIPSTAR_DIR="${CHIPSTAR_DIR:-/chipstar}"
WORK_DIR="${WORK_DIR:-/work}"
PREFIX=/opt/llvm-x86-aarch64
git config --global --add safe.directory '*' || true
mkdir -p "$WORK_DIR" && cd "$WORK_DIR"
"$CHIPSTAR_DIR/scripts/configure_llvm.sh" --version 23 \
  --install-dir "$PREFIX" --variant native --source-only
cmake -S "$WORK_DIR/llvm-project/llvm" -B "$WORK_DIR/x86-build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  "-DLLVM_ENABLE_PROJECTS=clang" \
  "-DLLVM_TARGETS_TO_BUILD=X86;AArch64;SPIRV" \
  -DLLVM_ENABLE_ASSERTIONS=Off \
  -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_DOCS=OFF \
  -DLLVM_LINK_LLVM_DYLIB=ON -DLLVM_BUILD_LLVM_DYLIB=ON \
  -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_LIBEDIT=OFF
ninja -C "$WORK_DIR/x86-build" -j"$JOBS"
DESTDIR="$WORK_DIR/x86-stage" ninja -C "$WORK_DIR/x86-build" install
"$WORK_DIR/x86-stage$PREFIX/bin/llvm-config" --targets-built
echo "X86-LLVM-OK"
