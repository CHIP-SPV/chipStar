#!/bin/bash
# Cross-build SPIRV-Tools for aarch64 into the image.
#
# chipStar's CMakeLists.txt builds SPIRV-Tools through ExternalProject_Add
# when find_package(SPIRV-Tools) fails, and hardcodes gcc/g++ as that
# sub-build's compiler (CMakeLists.txt:577-578), so in a cross build it
# silently produces x86 static libraries that the aarch64 libCHIP link then
# rejects. Providing a prebuilt aarch64 SPIRV-Tools makes find_package
# succeed and the ExternalProject never runs. Same repo and tag chipStar
# would have fetched itself.
set -e
JOBS="${1:-$(nproc)}"
WORK_DIR="${WORK_DIR:-/work}"
PREFIX=/opt/spirv-tools-aarch64
git config --global --add safe.directory '*' || true
mkdir -p "$WORK_DIR" && cd "$WORK_DIR"
[ -d SPIRV-Tools ] || git clone --depth 1 https://github.com/CHIP-SPV/SPIRV-Tools.git
cd SPIRV-Tools && python3 utils/git-sync-deps && cd ..
cmake -S SPIRV-Tools -B spirv-tools-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
  -DSPIRV_SKIP_TESTS=ON -DSPIRV_TOOLS_BUILD_STATIC=ON \
  -DSPIRV_TOOLS_INSTALL_HEADERS=ON -DCMAKE_INSTALL_LIBDIR=lib
ninja -C spirv-tools-build -j"$JOBS"
DESTDIR="$WORK_DIR/spirv-stage" ninja -C spirv-tools-build install
file "$WORK_DIR/spirv-stage$PREFIX/lib/libSPIRV-Tools.a" | head -1
aarch64-linux-gnu-readelf -h "$WORK_DIR/spirv-stage$PREFIX/lib/libSPIRV-Tools.a" 2>/dev/null | grep -m1 Machine
echo "SPIRV-TOOLS-OK"
