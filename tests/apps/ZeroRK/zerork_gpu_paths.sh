#!/bin/bash
# ZeroRK GPU code path check: run the h2 mechanism on the GPU through five
# integrator and reactor configurations and compare each with a CPU reference.
#
# Usage: zerork_gpu_paths.sh <outputs_dir>
#   CPU references already in <outputs_dir> are reused, so the CPU runs happen
#   once for any number of invocations (one per backend).
#
# Must be invoked from the zero-rk source root with build_aurora/ built and installed.

[ $# -eq 1 ] || { echo "Usage: $0 <outputs_dir>"; exit 1; }
out=$(mkdir -p "$1" && cd "$1" && pwd)
export CHIP_JIT_FLAGS_OVERRIDE="-ze-opt-enable-auto-large-GRF-mode" CHIP_LOGLEVEL=off
TOLERANCE=1e-4   # max relative difference in temperature and pressure

cd build_aurora/inst_dir || exit 1
exe=$PWD/bin/zerork_cfd_plugin_tester_gpu.x
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/lib:$PWD/lib64
cd share/zerork/examples/cfd_plugin_tester || exit 1

# Replace KEY in FILE, or append it: the base ymls lack some keys.
setYML() {
  grep -q "^ *$2 *:" "$1" && sed -i "s|^ *$2 *:.*$|$2: $3|" "$1" || echo "$2: $3" >> "$1"
}

# run NAME GPU CV INTEGRATOR ANALYTIC ALWAYS_SOLVE_T N_REACTORS N_REACTORS_MAX LU
run() {
  local name=$1 in=$out/$1_input.yml zrk=$out/$1_zerork.yml
  local fuel oxid
  fuel=$(grep -v '^#' inputs/h2_fracs.log | grep -Eiv '^(O2|N2)\b' | sed 's/$/,/' | tr '\n' ' ')
  oxid=$(grep -v '^#' inputs/h2_fracs.log | grep -Ei '^(O2|N2)\b' | sed 's/$/,/' | tr '\n' ' ')
  cp inputs/base_tester.yml "$in"; cp inputs/base_plugin.yml "$zrk"
  setYML "$in" mechanism_file "$PWD/../../mechanisms/hydrogen/h2_v1b_mech.txt"
  setYML "$in" thermo_file "$PWD/../../mechanisms/hydrogen/h2_v1a_therm.txt"
  setYML "$in" solution_time "${SOLT:-0.00000165131996}"
  setYML "$in" fuel_composition "{ $fuel }"
  setYML "$in" oxidizer_composition "{ $oxid }"
  setYML "$in" zerork_cfd_plugin_input "$zrk"
  setYML "$in" n_reactors "$7"
  setYML "$in" log_species "[h, h2, o, o2, oh, h2o, n2, ho2, h2o2, ar]"
  setYML "$in" n_steps "${STEPS:-4}"
  [ -n "$STATE" ] && setYML "$in" state_files "[$STATE]"
  setYML "$in" constant_volume "$3"
  setYML "$in" reactor_history_file_prefix "$out/$name"
  setYML "$zrk" mechanism_parsing_log "$out/$name.cklog"
  setYML "$zrk" reactor_timing_log "$out/$name.log"
  setYML "$zrk" preconditioner_threshold 3.20e-5
  setYML "$zrk" gpu "$2"
  setYML "$zrk" always_solve_temperature "$6"
  setYML "$zrk" dense 1
  setYML "$zrk" iterative 1
  setYML "$zrk" integrator "$4"
  setYML "$zrk" analytic "$5"
  setYML "$zrk" n_reactors_max "$8"
  rm -f "$out/$name"_*.hist "$out/$name.log" "$out/$name.stdout"
  ZERORK_REACTOR_USE_LU=$9 "$exe" "$in" > "$out/$name.stdout" 2>&1
}

# Temperature (col 3) and pressure (col 4) row by row against the CPU reference;
# $3 = leading rows of the GPU history to skip.
compare() {
  local gpu
  gpu=$(awk -v s="${3:-0}" '/^#/ || ++n > s' "$2")
  [ -s "$1" ] && [ -n "$gpu" ] && [ "$(wc -l < "$1")" -eq "$(echo "$gpu" | wc -l)" ] || return 1
  paste "$1" <(echo "$gpu") | LC_ALL=C awk -v tol=$TOLERANCE '/^#/ { next } {
    h = NF / 2; dt = ($(h+3) - $3) / $3; dp = ($(h+4) - $4) / $4
    if (dt < 0) dt = -dt; if (dp < 0) dp = -dp
    if (!(dt <= tol && dp <= tol) || $3 $4 $(h+3) $(h+4) ~ /[nN][aA][nN]|[iI][nN][fF]/) { printf "step %s: T %g vs %g, P %g vs %g\n", $1, $3, $(h+3), $4, $(h+4); bad = 1 } }
    END { exit bad }'
}

# Every timing log row must show all reactors on the GPU (zeroRK retries failed GPU reactors on the CPU);
# with $2 = 1 also some row with n_gpu_notemp > 0 (zeroRK counts them only on some solves).
# Columns: 3 n_cpu, 5 n_gpu, 6 n_gpu_notemp.
onGpu() {
  awk -v nt="$2" '/^#/ { next } { n++; if ($6 > 0) any = 1
    if ($3 != 0 || $5 <= 0) { print "row " $1 ": n_cpu " $3 ", n_gpu " $5; bad = 1 } }
    END { if (nt && !any) { print "n_gpu_notemp is 0 on every row"; bad = 1 } exit (bad || !n) }' "$1"
}

# n_reactors must stay >= the plugin's n_reactors_min (256) or the GPU is skipped silently.
N=512
# One CPU reference per constant_volume and always_solve_temperature pair.
for ref in cv1_t1 cv0_t1; do
  [ -s "$out/ref_${ref}_000.hist" ] || run ref_$ref 0 ${ref:2:1} 1 1 ${ref:5:1} 32 $N 0 || rm -f "$out/ref_${ref}_000.hist"
done
# zeroRK solves temperature on the first GPU call whatever always_solve_temperature says, so the
# no-temperature reference restarts from step 0 of the temperature reference without it.
# One reactor: the tester reads one state file per reactor (species, T, P).
if [ ! -s "$out/ref_cv1_t0_000.hist" ]; then
  awk '!/^#/ { print $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $3, $4; exit }' "$out/ref_cv1_t1_000.hist" > "$out/ref_s0.state" &&
    STEPS=3 SOLT=0.00000123848997 STATE=$out/ref_s0.state run ref_cv1_t0 0 1 1 1 0 1 $N 0 || rm -f "$out/ref_cv1_t0_000.hist"
fi
rc=0
# name  constant_volume integrator analytic always_solve_temperature n_reactors_max ZERORK_REACTOR_USE_LU
while read -r name cv integ analytic ast nmax lu; do
  if run "$name" 1 "$cv" "$integ" "$analytic" "$ast" $N "$nmax" "$lu" &&
    diff=$(onGpu "$out/$name.log" $((1 - ast)) && compare "$out/ref_cv${cv}_t${ast}_000.hist" "$out/${name}_000.hist" $((1 - ast))); then echo "PASS  zeroRK $name"
  else echo "FAIL  zeroRK $name ($out/$name.stdout)"; echo "$diff" | head -5; rc=1; fi
done << 'CASES'
cvode_analytic_cv      1 0 1 1 512 1
seulex_fd_cv           1 1 0 1 512 0
radau_analytic_cv      1 3 1 1 512 1
cvode_analytic_cp      0 0 1 1 512 0
seulex_fd_notemp_batch 1 1 0 0 256 1
CASES
exit $rc
