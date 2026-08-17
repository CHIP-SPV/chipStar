#!/bin/bash
# Cross-build chipStar's LLVM 23 toolchain for salami (aarch64-linux-gnu,
# Ubuntu 22.04) on an x86_64 host, inside the container built from the
# Dockerfile next to this script.
#
# The source tree (clone of llvmorg-23.1.0-rc2 + SPIRV-LLVM-Translator
# llvm_release_230 + the llvm-patches/llvm-23 series) is produced by
# scripts/configure_llvm.sh --version 23 --source-only, so the pinned refs and
# the patch series live in exactly one place. Only the cmake configure is
# duplicated here, because the cross build genuinely differs from the native
# one: an explicit target list (host detection would pick x86), the host/default
# triples, a native tablegen sub-build, and no host-gcc rpath flags.
#
# The openmp runtime cannot be built in-tree (LLVM_ENABLE_RUNTIMES) on a cross
# build: that mode compiles the runtime with the freshly built clang, which is
# an aarch64 binary the x86 host cannot execute. It is built standalone from
# runtimes/ with the same cross toolchain instead.
#
# Usage (inside the container):
#   cross-build.sh [jobs]
# Environment:
#   CHIPSTAR_DIR  chipStar checkout            (default /chipstar)
#   WORK_DIR      scratch: sources+build+stage (default /work)
#   LLVM_PREFIX   install prefix ON SALAMI     (default /home/pvelesko/install/llvm/23.0)
#
# Output: $WORK_DIR/stage$LLVM_PREFIX, ready to rsync to salami:$LLVM_PREFIX.

set -e

JOBS="${1:-$(nproc)}"
CHIPSTAR_DIR="${CHIPSTAR_DIR:-/chipstar}"
WORK_DIR="${WORK_DIR:-/work}"
# Must match the path the toolchain will live at on salami: it is baked into
# CMAKE_INSTALL_RPATH, so a mismatch yields binaries that cannot find libLLVM.
LLVM_PREFIX="${LLVM_PREFIX:-/home/pvelesko/install/llvm/23.0}"

echo "jobs=$JOBS chipstar=$CHIPSTAR_DIR work=$WORK_DIR prefix=$LLVM_PREFIX"

# The checkout is bind-mounted from the host and owned by another uid; git
# refuses to operate on it otherwise.
git config --global --add safe.directory '*' || true

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Clone/checkout the pinned refs and apply llvm-patches/llvm-23/. --install-dir
# is unused in --source-only mode but is a required argument.
"$CHIPSTAR_DIR/scripts/configure_llvm.sh" --version 23 \
  --install-dir "$LLVM_PREFIX" --variant native --source-only

rm -rf "$WORK_DIR/stage"

cmake -S "$WORK_DIR/llvm-project/llvm" -B "$WORK_DIR/build" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$(dirname "$(readlink -f "$0")")/aarch64-toolchain.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$LLVM_PREFIX" \
  -DCMAKE_INSTALL_RPATH="$LLVM_PREFIX/lib" \
  -DLLVM_HOST_TRIPLE=aarch64-unknown-linux-gnu \
  -DLLVM_DEFAULT_TARGET_TRIPLE=aarch64-unknown-linux-gnu \
  "-DCROSS_TOOLCHAIN_FLAGS_NATIVE=-DCMAKE_C_COMPILER=gcc;-DCMAKE_CXX_COMPILER=g++" \
  "-DLLVM_ENABLE_PROJECTS=clang;clang-tools-extra" \
  "-DLLVM_TARGETS_TO_BUILD=AArch64;SPIRV" \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_INCLUDE_DOCS=OFF \
  -DCLANG_DEFAULT_PIE_ON_LINUX=off \
  -DLLVM_LINK_LLVM_DYLIB=ON \
  -DLLVM_BUILD_LLVM_DYLIB=ON \
  -DLLVM_ENABLE_ZLIB=OFF \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_ENABLE_LIBXML2=OFF \
  -DLLVM_ENABLE_LIBEDIT=OFF

if [ "${CONFIGURE_ONLY:-off}" = "on" ]; then
  echo "CONFIGURE-ONLY-OK"
  exit 0
fi

ninja -C "$WORK_DIR/build" -j"$JOBS"
DESTDIR="$WORK_DIR/stage" ninja -C "$WORK_DIR/build" install

# openmp runtime, standalone cross build (libomp only; libomptarget needs a
# running clang and GPU plugins, neither applicable on salami).
#
# CMAKE_INSTALL_INCLUDEDIR is redirected at the clang resource dir on purpose.
# openmp/CMakeLists.txt only asks GetClangResourceDir for the header location
# when it sees an LLVM tree; standalone it falls back to CMAKE_INSTALL_INCLUDEDIR
# and drops omp.h in $PREFIX/include, where `clang -fopenmp` does not look --
# chipStar's TestHipccFopenmp then fails with "'omp.h' file not found". The
# variable it sets internally (LIBOMP_HEADERS_INSTALL_PATH) is a plain set(),
# not a cache entry, so overriding that one from the command line does nothing.
# This build installs no headers other than omp.h/omp-tools.h/ompt.h, so
# repointing the whole include dir is safe.
LLVM_MAJOR=$(basename "$(ls -d "$WORK_DIR/stage$LLVM_PREFIX"/lib/clang/*/ | head -1)")
cmake -S "$WORK_DIR/llvm-project/runtimes" -B "$WORK_DIR/build-openmp" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$(dirname "$(readlink -f "$0")")/aarch64-toolchain.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$LLVM_PREFIX" \
  -DCMAKE_INSTALL_RPATH="$LLVM_PREFIX/lib" \
  -DCMAKE_INSTALL_INCLUDEDIR="lib/clang/$LLVM_MAJOR/include" \
  -DLLVM_ENABLE_RUNTIMES=openmp \
  -DOPENMP_ENABLE_LIBOMPTARGET=OFF

ninja -C "$WORK_DIR/build-openmp" -j"$JOBS"
DESTDIR="$WORK_DIR/stage" ninja -C "$WORK_DIR/build-openmp" install

test -x "$WORK_DIR/stage$LLVM_PREFIX/bin/clang" \
  || { echo "ERROR: no clang in the stage"; exit 1; }
test -f "$WORK_DIR/stage$LLVM_PREFIX/lib/clang/$LLVM_MAJOR/include/omp.h" \
  || { echo "ERROR: omp.h is not in the clang resource dir"; exit 1; }
echo "BUILD-AND-STAGE-OK"
