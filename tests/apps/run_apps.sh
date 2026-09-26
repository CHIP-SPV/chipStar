#!/bin/bash
#
# Build and run the chipStar application tests: zeroRK, OpenSn, hoMusic.
# Needs an installed chipStar module. The run phase needs a compute node.
#
#   ./run_apps.sh                              # zeroRK + OpenSn
#   ./run_apps.sh zerork                       # one app
#   ./run_apps.sh --module-path $WRK/modulefiles --module chipStar/llvm21 all
#   ./run_apps.sh --phases build zerork opensn
#   ./run_apps.sh homusic
#
# "all" and the default are zeroRK and OpenSn. hoMusic runs only when named.
#
# Options:
#   -m, --module NAME        chipStar module to load (default: chipStar/llvm21)
#   -M, --module-path DIR    extra "module use" directory (repeatable)
#   -w, --work-dir DIR       clones, builds and outputs (default: ./chipstar_apps)
#   -p, --phases LIST        comma separated: build,run,report (default: all three)
#   -j, --jobs N             build parallelism (default: 20)
#   -n, --runs N             timed runs per app; the minimum is scored (default: 4)
#       --genesis-dir DIR    hoMusic scripts (default: $GENESIS_CI_PROJECT_DIR)
#
# Exits nonzero if any app failed to build, failed to run, or missed its
# performance baseline.

set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

CHIP_MODULE=chipStar/llvm21
MODULE_PATHS=(/soft/modulefiles)
WRK_DIR=$PWD/chipstar_apps
PHASES=build,run,report
BUILD_JOBS=20
RUNS=4
GENESIS_DIR=${GENESIS_CI_PROJECT_DIR:-/home/bertoni/projects/p01.chipStar/GENESIS-Share-CI}

# seconds; the minimum of the timed runs must be <= baseline * (1 + PERF_MARGIN)
BASELINE_zerork=77
BASELINE_opensn=31
BASELINE_homusic=81
PERF_MARGIN=0.06

OPENSN_COMMIT=d0644cd9c633c6ae5e7110d3f8721641e5ef2982
OPENSN_LAPACK_TARBALL=/lus/flare/projects/chipStar_test/chipStar/dependencies/f2cblaslapack-3.8.0.q2.tar.gz
OPENSN_CMAKE_MODULE_PATH=/lus/flare/projects/chipStar_test/chipStar/dependencies/modulefiles
OPENSN_CMAKE_MODULE=cmake/4.3.2.lua

die() { echo "run_apps.sh: $*" >&2; exit 2; }
banner() { echo; echo "=== $* ==="; }

APPS=()
while [ $# -gt 0 ]; do
  case $1 in
    -m|--module)      CHIP_MODULE=$2; shift 2 ;;
    -M|--module-path) MODULE_PATHS+=("$2"); shift 2 ;;
    -w|--work-dir)    WRK_DIR=$2; shift 2 ;;
    -p|--phases)      PHASES=$2; shift 2 ;;
    -j|--jobs)        BUILD_JOBS=$2; shift 2 ;;
    -n|--runs)        RUNS=$2; shift 2 ;;
    --genesis-dir)    GENESIS_DIR=$2; shift 2 ;;
    -h|--help)        sed -n '2,40p' "$0"; exit 0 ;;
    -*)               die "unknown option $1" ;;
    all)              APPS+=(zerork opensn); shift ;;
    zerork|zeroRK)    APPS+=(zerork); shift ;;
    opensn|OpenSn)    APPS+=(opensn); shift ;;
    homusic|hoMusic)  APPS+=(homusic); shift ;;
    *)                die "unknown app $1 (zerork, opensn, homusic, all)" ;;
  esac
done
[ ${#APPS[@]} -gt 0 ] || APPS=(zerork opensn)

case ,$PHASES, in *,build,*|*,run,*|*,report,*) ;; *) die "no valid phase in '$PHASES'" ;; esac
wants() { case ,$PHASES, in *,$1,*) return 0 ;; *) return 1 ;; esac; }

WRK_DIR=$(mkdir -p "$WRK_DIR" && cd "$WRK_DIR" && pwd)

# hoMusic runs "module restore", so the driver re-adds these itself
export HOMUSIC_MODULE_PATHS="${MODULE_PATHS[*]}"

load_chipstar() {
  local d
  for d in "${MODULE_PATHS[@]}"; do
    module use "$d"
  done
  module load cmake "$CHIP_MODULE"
}

#
# Timing. Each times_* function prints one seconds value per timed run.
#

times_zerork()  { awk -F': ' '/Job time in seconds/ {print $2 + 0}' "$1"; }
times_opensn()  { awk '/Elapsed execution time/ {split($NF, f, ":"); print (f[1] * 3600) + (f[2] * 60) + f[3]}' "$1"; }
times_homusic() { awk '/^HOMUSIC_ELAPSED_SECONDS/ {print $2 + 0}' "$1"; }

timing_spread() {
  awk '{t[n++] = $1 + 0}
       END {
         if (!n) exit
         min = max = t[0]
         for (i = 1; i < n; i++) {
           if (t[i] < min) min = t[i]
           if (t[i] > max) max = t[i]
         }
         printf "runs=%d  min=%.3f  max=%.3f  spread=%.3f (%.1f%%)\n", \
           n, min, max, max - min, (min ? 100 * (max - min) / min : 0)
         printf "values:"
         for (i = 0; i < n; i++) printf " %.3f", t[i]
         print ""
       }'
}

perf_check() {
  local app=$1 output=$2
  local baseline_var=BASELINE_$app
  local baseline=${!baseline_var}

  [ -r "$output" ] || { echo "no output file $output"; return 1; }

  local times
  times=$("times_$app" "$output")
  [ -n "$times" ] || { echo "performance check failed: no timing lines in $output"; return 1; }

  local actual nruns upper
  actual=$(printf '%s\n' "$times" | sort -g | head -1)
  nruns=$(printf '%s\n' "$times" | wc -l)
  upper=$(echo "$baseline * (1 + $PERF_MARGIN)" | bc -l)

  printf 'minimum of %s runs: %s s | target: <= %.2f s (baseline %s s + %g%%)\n' \
    "$nruns" "$actual" "$upper" "$baseline" "$(echo "$PERF_MARGIN * 100" | bc -l)"
  printf '%s\n' "$times" | timing_spread

  if [ "$(echo "$actual <= $upper" | bc -l)" -eq 1 ]; then
    echo "performance check passed"
    return 0
  fi
  echo "performance check failed: ${actual}s > ${upper}s"
  return 1
}

#
# zeroRK
#

build_zerork() {
  cd "$WRK_DIR"
  [ -d zero-rk ] && rm -rf zero-rk
  git clone git@github.com:CHIP-SPV/zero-rk.git -b hip-chipStar
  cd zero-rk
  load_chipstar
  mkdir -p build_aurora
  cd build_aurora
  MAKEFLAGS=-j$BUILD_JOBS sh ../scripts/build_aurora.sh --no-module-loads
}

run_zerork() {
  cd "$WRK_DIR/zero-rk"
  load_chipstar
  local output=$WRK_DIR/zero-rk/output failed="" i lu run_dir
  rm -f "$output"

  for i in $(seq 1 "$RUNS"); do
    run_dir=$WRK_DIR/zero-rk/running_test_aurora_run$i
    rm -rf "$run_dir"
    cp -a "$WRK_DIR/zero-rk/running_test_aurora" "$run_dir"
    cd "$run_dir"
    mkdir -p logs
    chmod a+x ./submit_job.sh
    banner "zeroRK timed run $i/$RUNS ($run_dir)"
    ./submit_job.sh |& tee -a "$output" || failed="$failed timing-run-$i"
    cd "$WRK_DIR/zero-rk"
  done

  for lu in 1 0; do
    banner "zeroRK correctness: hydrogen, ZERORK_REACTOR_USE_LU=$lu"
    cd "$WRK_DIR/zero-rk"
    bash "$SCRIPT_DIR/ZeroRK/zerork_correctness_hydrogen.sh" $lu || failed="$failed hydrogen-lu$lu"
  done

  for lu in 1 0; do
    banner "zeroRK correctness: box reactor, ZERORK_REACTOR_USE_LU=$lu"
    cd "$WRK_DIR/zero-rk"
    bash "$SCRIPT_DIR/ZeroRK/zerork_correctness_box_reactor.sh" $lu || failed="$failed box-reactor-lu$lu"
  done

  if [ -n "$failed" ]; then
    echo "zeroRK FAILED:$failed"
    return 1
  fi
  echo "zeroRK: all checks passed ($RUNS timing runs, hydrogen lu1/lu0, box reactor lu1/lu0)"
}

report_zerork() { perf_check zerork "$WRK_DIR/zero-rk/output"; }

#
# OpenSn
#

build_opensn() {
  cd "$WRK_DIR"
  module load frameworks/2025.3.1
  export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
  export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
  load_chipstar

  local base=$WRK_DIR
  [ -d opensn ] && rm -rf opensn
  git clone git@github.com:Open-Sn/opensn.git
  cd opensn
  git checkout $OPENSN_COMMIT

  # use the staged f2cblaslapack tarball instead of downloading it
  if grep -q -- '--download-f2cblaslapack=yes' tools/dependencies/CMakeLists.txt; then
    sed -i "s|--download-f2cblaslapack=yes|--download-f2cblaslapack=$OPENSN_LAPACK_TARBALL|" \
      tools/dependencies/CMakeLists.txt
  fi

  mkdir -p build_deps
  cd build_deps
  cmake -DCMAKE_INSTALL_PREFIX="$base/opensn/dependencies" ../tools/dependencies
  make -j"$BUILD_JOBS"
  cd ..

  # the main build needs this cmake, not the default 3.x
  module use "$OPENSN_CMAKE_MODULE_PATH"
  module load "$OPENSN_CMAKE_MODULE"

  source "$base/opensn/dependencies/bin/set_opensn_env.sh"
  mkdir -p build_debug
  cd build_debug
  CC=hipcc CXX=hipcc cmake \
    -DOPENSN_WITH_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=spirv \
    -DCMAKE_HIP_PLATFORM=spirv \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
  make -j"$BUILD_JOBS"
}

run_opensn() {
  load_chipstar
  module load hdf5
  export CHIP_MODULE_CACHE_DIR=$WRK_DIR/module_cache

  local bin=$WRK_DIR/opensn/build_debug/python/opensn
  local inputs=$SCRIPT_DIR/OpenSn
  local output=$WRK_DIR/opensn/output rc=0 i run_dir
  rm -f "$output"

  for i in $(seq 1 "$RUNS"); do
    run_dir=$WRK_DIR/opensn/run_$i
    banner "OpenSn timed run $i/$RUNS ($run_dir)"
    rm -rf "$run_dir"
    mkdir -p "$run_dir"
    cp "$inputs"/strong_scaling.* "$inputs/xs_168g.xs" "$run_dir/"
    cd "$run_dir"
    CHIP_LOGLEVEL=off mpiexec --cpu-bind=list:1-8 --env OMP_NUM_THREADS=8 --env OMP_PLACES=cores -n 1 \
      gpu_tile_compact.sh "$bin" -i strong_scaling.py |& tee -a "$output" || rc=1
  done
  return $rc
}

report_opensn() { perf_check opensn "$WRK_DIR/opensn/output"; }

#
# hoMusic
#

build_homusic() {
  [ -x "$GENESIS_DIR/hoMusic_ci_improved.sh" ] || die "no hoMusic driver in $GENESIS_DIR"
  "$GENESIS_DIR/hoMusic_ci_improved.sh" build "$CHIP_MODULE"
}

run_homusic() {
  local output=$WRK_DIR/homusic_output rc=0 i
  rm -f "$output"
  for i in $(seq 1 "$RUNS"); do
    banner "hoMusic timed run $i/$RUNS"
    "$GENESIS_DIR/hoMusic_ci_improved.sh" run "$CHIP_MODULE" "$i" |& tee -a "$output" || rc=1
  done
  "$GENESIS_DIR/hoMusic_ci_improved.sh" clean "$CHIP_MODULE"
  return $rc
}

report_homusic() { perf_check homusic "$WRK_DIR/homusic_output"; }

#
# Driver
#

echo "work dir:   $WRK_DIR"
echo "module:     $CHIP_MODULE  (module use: ${MODULE_PATHS[*]})"
echo "apps:       ${APPS[*]}"
echo "phases:     $PHASES   runs: $RUNS   jobs: $BUILD_JOBS"
echo "host:       $(hostname)"

FAILED=()
for app in "${APPS[@]}"; do
  for phase in build run report; do
    wants $phase || continue
    banner "$app: $phase"
    if (set -eo pipefail; "${phase}_${app}"); then
      echo "$app $phase: ok"
    else
      echo "$app $phase: FAILED"
      FAILED+=("$app-$phase")
      break   # skip the later phases of this app
    fi
  done
done

banner "summary"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED: ${FAILED[*]}"
  exit 1
fi
echo "all passed: ${APPS[*]} (${PHASES})"
