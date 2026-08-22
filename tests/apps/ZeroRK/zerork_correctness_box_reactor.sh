#!/bin/bash
# ZeroRK box reactor correctness test: run the GPU box reactor case and
# check that the temperature trend in box_reactor_results.csv is sane:
#   - all values finite (no NaN/inf)
#   - T_max never decreases (beyond a tiny relative tolerance)
#   - final T_max clearly above the initial 1800 K (ignition progressed)
#   - T_min stays near its initial 600 K
#
# Thresholds are calibrated to the reference run with ZERORK_TEST_N_STEPS=750:
# T_max rises from 1800 K to ~2086 K, T_min holds at 600 K.
#
# Usage: zerork_correctness_box_reactor.sh <0|1>
#   The argument is the value for ZERORK_REACTOR_USE_LU.
#
# Must be invoked from the zero-rk source root (build_aurora/ must already
# exist and be built/installed, i.e. after the zeroRK_build CI job), inside
# a PBS batch job (uses PBS_NODEFILE and mpiexec).

if [ $# -ne 1 ]; then
  echo "Usage: $0 <0|1>  (value for ZERORK_REACTOR_USE_LU)"
  exit 1
fi

export ZERORK_REACTOR_USE_LU=$1
export CHIP_JIT_FLAGS_OVERRIDE="-ze-opt-enable-auto-large-GRF-mode"
export CHIP_LOGLEVEL=off
export ZERORK_TEST_N_STEPS=750

# trend-check parameters
TMAX_DIP_TOL=1e-6     # max allowed relative decrease of T_max between steps
TMAX_FINAL_MIN=2000   # K; final T_max must exceed this (reference reaches ~2086)
TMIN_LOW=570          # K; allowed band for T_min (reference holds at 600)
TMIN_HIGH=630

cd box_reactor || exit 1
mkdir -p logs

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS=12       # Number of MPI ranks per node
NDEPTH=1        # Number of hardware threads per rank, spacing between MPI ranks on a node
NTHREADS=1      # Number of OMP threads per rank, given to OMP_NUM_THREADS
NTOTRANKS=$((NNODES*NRANKS))

export INST_DIR=$PWD/../build_aurora/inst_dir/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${INST_DIR}/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${INST_DIR}/lib64

ZERORK_TEST=${INST_DIR}/bin/zerork_box_reactor_test_gpu.x

results_csv=box_reactor_results_lu${ZERORK_REACTOR_USE_LU}.csv
rm -f box_reactor_results.csv ${results_csv}

echo "Running box reactor (ZERORK_REACTOR_USE_LU=${ZERORK_REACTOR_USE_LU}) with ${NTOTRANKS} ranks on ${NNODES} node(s)"
mpiexec --np ${NTOTRANKS} -ppn ${NRANKS} -d ${NDEPTH} -env OMP_NUM_THREADS=${NTHREADS} gpu_tile_compact.sh $ZERORK_TEST

if [ ! -s box_reactor_results.csv ]; then
  echo "FAIL: box_reactor_results.csv missing or empty"
  exit 1
fi
# keep per-LU-mode results separate
mv box_reactor_results.csv ${results_csv}

echo "Checking temperature trend in ${results_csv} (ZERORK_REACTOR_USE_LU=${ZERORK_REACTOR_USE_LU})"
# CSV columns: Step,Time,T_min,T_max
awk -F, -v dip_tol=${TMAX_DIP_TOL} -v final_min=${TMAX_FINAL_MIN} \
    -v tmin_low=${TMIN_LOW} -v tmin_high=${TMIN_HIGH} '
  NR == 1 { next }
  {
    for (i = 1; i <= 4; i++) {
      if (tolower($i) ~ /nan|inf/) {
        if (msgs++ < 10) printf "FAIL: non-finite value at line %d: %s\n", NR, $0
        fail = 1
        next
      }
    }
    rows++
    tmin = $3 + 0
    tmax = $4 + 0
    if (rows == 1) first_tmax = tmax
    if (tmin < tmin_low || tmin > tmin_high) {
      if (msgs++ < 10) printf "FAIL: T_min %g outside [%g, %g] at step %s\n", tmin, tmin_low, tmin_high, $1
      fail = 1
    }
    if (rows > 1 && tmax < prev_tmax * (1 - dip_tol)) {
      if (msgs++ < 10) printf "FAIL: T_max decreased at step %s: %g -> %g\n", $1, prev_tmax, tmax
      fail = 1
    }
    prev_tmax = tmax
  }
  END {
    if (rows == 0) {
      print "FAIL: no data rows in results file"
      exit 1
    }
    printf "steps: %d, T_max: %g -> %g K\n", rows, first_tmax, prev_tmax
    if (prev_tmax < final_min) {
      printf "FAIL: final T_max %g K below %g K (temperature did not increase as expected)\n", prev_tmax, final_min
      fail = 1
    }
    if (fail) exit 1
    print "PASS: temperature trend OK (finite, T_max non-decreasing and reached " prev_tmax " K, T_min in band)"
  }' "${results_csv}"
