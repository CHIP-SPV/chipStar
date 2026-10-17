#!/bin/bash
# Build chipStar plus every downstream library with its tests, then run each
# library's ctest suite and the zeroRK and OpenSn checks on one Intel GPU per
# backend. Waits while another process holds the GPU and stops all GPU work at
# the first reset or wedge the kernel logs for it. Results go to summary.txt.
#
#   scripts/ci/libs-apps.sh -w WORK_DIR [-s CHIPSTAR_SRC] [-j JOBS] [-b "level0 opencl"] [-p build,libs,apps]
set -o pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SRC=$HERE WRK="" JOBS="" BACKENDS="level0 opencl" PHASES=build,libs,apps
INTEL_GPUS=$(grep -l 0x8086 /sys/bus/pci/devices/*/vendor | xargs -n1 dirname |
  while read -r d; do grep -q '^0x03' "$d/class" && echo "${d##*/}"; done | sort)
# First Intel display device off bus 00, where the integrated GPU sits.
GPU_PCI=${GPU_PCI:-$(grep -v '^0000:00:' <<< "$INTEL_GPUS" | head -1)}
DEPS_CACHE=${DEPS_CACHE:-$HOME/.cache/chipstar-libs-apps}
OPENSN_COMMIT=d0644cd9c633c6ae5e7110d3f8721641e5ef2982
while getopts "s:w:j:b:p:" o; do
  case $o in
    s) SRC=$(cd "$OPTARG" && pwd) || exit 2 ;; w) WRK=$OPTARG ;; j) JOBS=$OPTARG ;;
    b) BACKENDS=$OPTARG ;; p) PHASES=$OPTARG ;; *) sed -n '2,7p' "$0"; exit 2 ;;
  esac
done
[ -n "$WRK" ] && mkdir -p "$WRK/logs" && WRK=$(cd "$WRK" && pwd) || { sed -n '2,7p' "$0"; exit 2; }
LOGS=$WRK/logs SUMMARY=$WRK/summary.txt FAILED=0 GPU_DEAD=0
# idle cores minus one, at most one job per 3 GB of available RAM
: "${JOBS:=$(awk -v n="$(nproc)" -v m="$(free -g | awk '/^Mem:/ {print $7}')" \
  '{ j = int(n - $1 - 1); if (m / 3 < j) j = int(m / 3); print (j < 1 ? 1 : j) }' /proc/loadavg)}"
RENDER=/dev/dri/$(basename "$(ls -d /sys/bus/pci/devices/"$GPU_PCI"/drm/renderD*)")
DRV=$(basename "$(readlink /sys/bus/pci/devices/"$GPU_PCI"/driver)")
# Level Zero and the Intel OpenCL platform both hide every GPU but this one, indexed in PCI order.
ZE_IDX=$(($(grep -nx "$GPU_PCI" <<< "$INTEL_GPUS" | cut -d: -f1) - 1))
[ "$ZE_IDX" -ge 0 ] || { echo "$GPU_PCI is not an Intel display device"; exit 2; }
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 ZE_AFFINITY_MASK=$ZE_IDX CHIP_LOGLEVEL=err
unset CHIP_PLATFORM CHIP_DEVICE CHIP_DEVICE_TYPE
# Arc A-series GPUs have no FP64 hardware; the apps and most libraries need double.
export OverrideDefaultFP64Settings=1 IGC_EnableDPEmulation=1
source "$HERE/scripts/module-env.sh"
module load oneapi/2025.0.4 llvm/23.0 || exit 1
# zeroRK and OpenSn bundle projects that declare a CMake minimum below 3.5.
export CMAKE_POLICY_VERSION_MINIMUM=3.5

trap '[ -n "$STEP" ] && { kill -KILL "$STEP"; pkill -KILL -s "$STEP"; } 2>/dev/null; exit 143' INT TERM
wants() { case ,$PHASES, in *,$1,*) return 0 ;; esac; return 1; }
note() { echo "$*" | tee -a "$SUMMARY"; }
result() { [ "$2" -eq 0 ] && note "PASS  $1" || { note "FAIL  $1 (rc=$2, $3)"; FAILED=$((FAILED + 1)); }; }
# A journal read error counts as a fault: the GPU cannot be watched. Sticky, so a transient error is not forgotten.
gpu_faults() {
  local j p="$DRV $GPU_PCI.*(reset|wedged|timedout|hang)|^Failed|^No journal"
  [ -e "$LOGS/fault" ] && { echo 1; return; }
  j=$(journalctl -k -b -q --no-pager --after-cursor="$CURSOR" 2>&1) || j="Failed: journalctl exit $?
$j"
  j=$(grep -E "$DRV $GPU_PCI|^Failed|^No journal" <<< "$j")
  grep -qiE "$p" <<< "$j" || { echo 0; return; }
  { grep -iE "$p" <<< "$j" | head -100; head -900 <<< "$j"; } > "$LOGS/fault"; echo 1
}

# gpu_step NAME LOG CMD...; CMD gets its own session so a fault kills all of it.
gpu_step() {
  local name=$1 log=$2 pid="" f rc; shift 2
  [ $GPU_DEAD -eq 0 ] || { note "SKIP  $name"; return; }
  while f=$(gpu_faults) && [ "$f" -eq 0 ] && fuser "$RENDER" > /dev/null 2>&1; do echo "[gpu] $RENDER busy, waiting"; sleep 60; done
  if [ "$f" -eq 0 ]; then
    setsid "$@" > "$log" 2>&1 < /dev/null &
    pid=$! STEP=$!
    while kill -0 $pid 2>/dev/null && f=$(gpu_faults) && [ "$f" -eq 0 ]; do sleep 1; done
    [ "$f" -gt 0 ] || f=$(gpu_faults)
  fi
  if [ "$f" -gt 0 ]; then
    [ -n "$pid" ] && { kill -KILL $pid; pkill -KILL -s $pid; } 2>/dev/null; GPU_DEAD=1
    note "STOP  GPU reset, wedge or journal error at $name $(grep -so 'Start *[0-9]*: [^ ]*' "$log" | tail -1), see logs/fault"
    result "$name" 137 "$log"; STEP=""; return
  fi
  wait $pid; rc=$?
  pkill -KILL -s $pid 2>/dev/null
  result "$name" $rc "$log"; STEP=""
}

# Same .ci/exclude.txt and .ci/ctest-labels.txt convention as libraries-presubmit.yml.
run_libs() {
  local lib s ex lb
  for s in "$WRK"/staging/*/; do
    lib=$(basename "$s")
    [ -f "$s/build/CTestTestfile.cmake" ] || continue
    ex=$(grep -Evhs '^\s*(#|$)' "$s/.ci/exclude.txt" | paste -sd '|' -)
    lb=$(grep -Evhs '^\s*(#|$)' "$s/.ci/ctest-labels.txt" | paste -sd '|' -)
    rm -f "$LOGS/$1/$lib.log"
    gpu_step "$1 $lib" "$LOGS/$1/$lib.log" ctest --test-dir "$s/build" -j1 --output-on-failure \
      --timeout "${CTEST_TIMEOUT:-900}" ${lb:+-L "$lb"} ${ex:+-E "$ex"}
    [ $GPU_DEAD -eq 0 ] && grep -hs 'tests passed' "$LOGS/$1/$lib.log" | sed 's/^/      /' | tee -a "$SUMMARY"
  done
}

# Built trees are cached per runner by app commit; the libraries they load come from this run's PREFIX.
lock() { mkdir -p "$DEPS_CACHE" && exec 9> "$DEPS_CACHE/lock" && flock 9; }

# zeroRK's scripts/build_aurora.sh without its module loads and unbounded make -j.
# The -I does not beat zero-rk's hardcoded rocPRIM and rocThrust include dirs, see https://github.com/CHIP-SPV/chipStar/issues/2006
build_zerork() {
  lock || return 1
  [ -e "$ZRK/.built" ] && { echo "reusing $ZRK"; return 0; }
  rm -rf "$ZRK" && git clone -q --depth 1 -b hip-chipStar git@github.com:CHIP-SPV/zero-rk.git "$ZRK" &&
    mkdir "$ZRK/build_aurora" && cd "$ZRK/build_aurora" &&
    cmake -DCMAKE_CXX_FLAGS="-I$PREFIX/include" -DCMAKE_EXE_LINKER_FLAGS=-lintlc -DENABLE_GPU=ON -DENABLE_MAGMA=OFF -DCMAKE_CXX_COMPILER=hipcc \
      -DCMAKE_C_COMPILER=hipcc -DZERORK_TESTS=OFF -DZERORK_HAVE_MKL=OFF .. && make -j"$JOBS" && make install &&
    touch "$ZRK/.built"
}

# OpenSn's dependencies do not use chipStar, so each runner builds them once.
build_opensn() {
  local deps=$DEPS_CACHE/opensn-deps-$OPENSN_COMMIT
  lock || return 1
  [ -e "$OSN/.built" ] && { echo "reusing $OSN"; return 0; }
  rm -rf "$OSN" && git clone -q https://github.com/Open-Sn/opensn.git "$OSN" && cd "$OSN" &&
    git checkout -q $OPENSN_COMMIT || return 1
  [ -s "$deps/bin/set_opensn_env.sh" ] || {
    rm -rf "$deps" && cmake -S tools/dependencies -B build_deps -DCMAKE_INSTALL_PREFIX="$deps" -DJOBS="$JOBS" \
      -DCMAKE_DISABLE_FIND_PACKAGE_Boost=ON &&
      cmake --build build_deps -j1; } || { rm -rf "$deps"; return 1; }
  # CMAKE_HIP_PLATFORM=spirv needs CMake 4.3.
  local cm=$DEPS_CACHE/cmake-4.3.2-linux-x86_64 tgz=$DEPS_CACHE/cmake-4.3.2.tar.gz
  [ -x "$cm/bin/cmake" ] || {
    curl -sSfL -o "$tgz" https://github.com/Kitware/CMake/releases/download/v4.3.2/cmake-4.3.2-linux-x86_64.tar.gz &&
      echo "791ae3604841ca03cb3889a3ad89165346e4b180ae3448efd4b0caa9ef46d245  $tgz" | sha256sum -c - &&
      tar xzf "$tgz" -C "$DEPS_CACHE" && rm -f "$tgz"; } || return 1
  source "$deps/bin/set_opensn_env.sh" &&
    CC=hipcc CXX=hipcc "$cm/bin/cmake" -S . -B build -DOPENSN_WITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=spirv \
      -DCMAKE_HIP_PLATFORM=spirv -DCMAKE_BUILD_TYPE=RelWithDebInfo &&
    "$cm/bin/cmake" --build build -j"$JOBS" && touch "$OSN/.built"
}

run_apps() {
  note "SKIP  $1 zeroRK box_reactor (zerork_correctness_box_reactor.sh needs Aurora's PBS and MPI launcher)"
  rm -rf "$LOGS/$1/zerork-out"
  gpu_step "$1 zeroRK GPU paths" "$LOGS/$1/zerork-paths.log" timeout -k 60 1800 bash -c 'cd "$0" && bash "$1" "$2"' \
    "$ZRK" "$HERE/tests/apps/ZeroRK/zerork_gpu_paths.sh" "$WRK/apps/zerork-out"
  [ $GPU_DEAD -eq 0 ] && grep -hsE '^(PASS|FAIL) ' "$LOGS/$1/zerork-paths.log" | sed "s/^/      $1 /" | tee -a "$SUMMARY"
  cp -r "$WRK/apps/zerork-out" "$LOGS/$1/" 2>/dev/null
  rm -rf "$WRK/apps/opensn-run" && mkdir "$WRK/apps/opensn-run" && cp "$SRC"/tests/apps/OpenSn/* "$WRK/apps/opensn-run"
  gpu_step "$1 OpenSn strong_scaling" "$LOGS/$1/opensn.log" timeout -k 60 1800 bash -c \
    'cd "$0" && source "$1" && CHIP_LOGLEVEL=off mpiexec -n 1 "$2" -i strong_scaling.py' "$WRK/apps/opensn-run" \
    "$DEPS_CACHE/opensn-deps-$OPENSN_COMMIT/bin/set_opensn_env.sh" "$OSN/build/python/opensn"
}

note "=== $(date -u +%FT%TZ) $(hostname) GPU $GPU_PCI $DRV ($RENDER) chipStar $(git -C "$SRC" log -1 --format='%h %s') -j$JOBS"
if wants build; then
  rm -rf "$WRK/install" "$WRK/staging"
  (cd "$SRC" && ./install_chipstar.py --all --with-tests -j "$JOBS" --install-dir "$WRK/install" \
    --staging-dir "$WRK/staging" --module-dir "$WRK/modulefiles") > "$LOGS/build.log" 2>&1
  result "build: install_chipstar.py --all --with-tests" $? "$LOGS/build.log"
  sed 's/\x1b\[[0-9;]*m//g' "$LOGS/build.log" | grep -E '^\[(SUCCESS|FAILED)\]' | cut -c1-150 | sed 's/^/      /' | tee -a "$SUMMARY"
  for s in "$WRK"/staging/*/; do note "      $(basename "$s") $(git -C "$s" rev-parse --short HEAD)"; done
fi
PREFIX=$(ls -d "$WRK"/install/chipStar/*/ | tail -1) && PREFIX=${PREFIX%/}
[ -x "$PREFIX/bin/hipcc" ] || { note "FAIL  no chipStar install under $WRK/install"; exit 1; }
export HIP_PATH=$PREFIX HIP_PLATFORM=spirv PATH=$PREFIX/bin:$PATH CMAKE_PREFIX_PATH=$PREFIX:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH LIBRARY_PATH=$PREFIX/lib:$LIBRARY_PATH
if wants apps; then
  rm -rf "$WRK/apps" && mkdir "$WRK/apps"
  ZRK=$DEPS_CACHE/zerork-$(git ls-remote git@github.com:CHIP-SPV/zero-rk.git hip-chipStar | cut -c1-40)-$(find "$PREFIX"/include/rocprim "$PREFIX"/include/thrust -type f | sort | xargs cat | cksum | cut -d' ' -f1)
  OSN=$DEPS_CACHE/opensn-$OPENSN_COMMIT
  (cd "$WRK/apps" && build_zerork) > "$LOGS/zerork-build.log" 2>&1; result "build zeroRK" $? "$LOGS/zerork-build.log"
  (cd "$WRK/apps" && build_opensn) > "$LOGS/opensn-build.log" 2>&1; result "build OpenSn" $? "$LOGS/opensn-build.log"
fi
wants libs || wants apps || BACKENDS=""
rm -f "$LOGS/fault"
CURSOR=$(journalctl -k -b -n0 --show-cursor -q --no-pager | sed -n 's/^-- cursor: //p')
[ -z "$BACKENDS" ] || [ -n "$CURSOR" ] || { note "FAIL  kernel journal unreadable, cannot watch the GPU"; exit 1; }
[ -z "$BACKENDS" ] || { n=$(journalctl -k -b -q --no-pager | grep -ciE "$DRV $GPU_PCI.*wedged"; echo "${PIPESTATUS[0]}"); [ "$(echo $n)" = "0 0" ]; } ||
  { note "FAIL  $GPU_PCI was declared wedged earlier in this boot, or the kernel journal is unreadable"; exit 1; }
[ -z "$BACKENDS" ] || command -v fuser > /dev/null || { note "FAIL  fuser not found, cannot see other GPU users"; exit 1; }
for be in $BACKENDS; do
  mkdir -p "$LOGS/$be" && export CHIP_BE=$be
  wants libs && run_libs "$be"
  wants apps && run_apps "$be"
done
note "=== $FAILED failed"
[ "$FAILED" -eq 0 ] && [ "$GPU_DEAD" -eq 0 ]
