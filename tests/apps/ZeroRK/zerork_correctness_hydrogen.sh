#!/bin/bash
# ZeroRK GPU correctness test: run the h2 dense case on CPU and GPU and
# compare the temperature and pressure histories.
#
# Usage: zerork_correctness.sh <0|1>
#   The argument is the value for ZERORK_REACTOR_USE_LU.
#
# Must be invoked from the zero-rk source root (build_aurora/ must already
# exist and be built/installed, i.e. after the zeroRK_build CI job).

if [ $# -ne 1 ]; then
  echo "Usage: $0 <0|1>  (value for ZERORK_REACTOR_USE_LU)"
  exit 1
fi

export ZERORK_REACTOR_USE_LU=$1
export CHIP_JIT_FLAGS_OVERRIDE="-ze-opt-enable-auto-large-GRF-mode"
export CHIP_LOGLEVEL=off

# max allowed relative difference in temperature and pressure (CPU vs GPU)
TOLERANCE=1e-4

cd build_aurora || exit 1

zerork_exe=$PWD/inst_dir/bin/zerork_cfd_plugin_tester_gpu.x
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PWD}/inst_dir/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PWD}/inst_dir/lib64

cd inst_dir/share/zerork/examples/cfd_plugin_tester || exit 1

outputs_dir=outputs_gpu_lu${ZERORK_REACTOR_USE_LU}
if [ -d ${outputs_dir} ]
then
  rm -r ${outputs_dir}
fi
mkdir ${outputs_dir}


setYMLScalar()
{
  file=$1
  scalar_name=$2
  scalar_val=$3
  sed "s|^\ *${scalar_name}\ *:\ *.*$|${scalar_name}: ${scalar_val}|" $file > ${outputs_dir}/tmp.yml
  mv ${outputs_dir}/tmp.yml $file
}

runSims() {
  name=$1
  mechFile=$2
  thermFile=$3
  sparseThresh=$4
  ignitionDelayTime=$5
  dodense=$6

  ignitionDelayTime=`echo ${ignitionDelayTime} | sed 's/[eE]/\\*10\\^/' | sed 's/+//'`
  simTime=$(echo "scale=20; ${ignitionDelayTime} * 2" | bc)


  initFuelMassFracs=`grep -v "^#" inputs/${name}_fracs.log | egrep -iv '^\<O2\>|^\<N2\>' | sed 's/$/,/'`
  initFuelMassFracs=`echo $initFuelMassFracs`  #strip newlines
  initOxidMassFracs=`grep -v "^#" inputs/${name}_fracs.log | egrep -i  '^\<O2\>|^\<N2\>' | sed 's/$/,/'`
  initOxidMassFracs=`echo $initOxidMassFracs`  #strip newlines

  infile=${outputs_dir}/${name}_input.yml
  zrkfile=${outputs_dir}/${name}_zerork.yml
  cp inputs/base_tester.yml $infile
  cp inputs/base_plugin.yml $zrkfile

  setYMLScalar $infile mechanism_file $mechFile
  setYMLScalar $infile thermo_file $thermFile
  setYMLScalar $infile solution_time $simTime
  setYMLScalar $infile fuel_composition "{ $initFuelMassFracs }"
  setYMLScalar $infile oxidizer_composition "{ $initOxidMassFracs }"
  setYMLScalar $infile zerork_cfd_plugin_input $zrkfile
  setYMLScalar $infile n_reactors 32
  setYMLScalar $zrkfile integrator 1

  setYMLScalar $zrkfile preconditioner_threshold $sparseThresh
  setYMLScalar $zrkfile mechanism_parsing_log ${outputs_dir}/${name}.cklog


  #Dense
  if [ "x"$dodense == "xY" ]
  then
    echo "Running ${name} Dense"
    setYMLScalar $zrkfile gpu 0
    setYMLScalar $zrkfile dense 1
    setYMLScalar $zrkfile analytic 1
    setYMLScalar $zrkfile iterative 1
    setYMLScalar $zrkfile reactor_timing_log ${outputs_dir}/${name}_dense.log
    setYMLScalar $infile reactor_history_file_prefix ${outputs_dir}/${name}_dense
    $zerork_exe $infile >& ${outputs_dir}/${name}_dense.stdout
  fi

  ## GPU
  setYMLScalar $infile n_reactors 2048

  #Dense
  if [ "x"$dodense == "xY" ]
  then
    echo "Running ${name} Dense GPU"
    setYMLScalar $zrkfile gpu 1
    setYMLScalar $zrkfile dense 1
    setYMLScalar $zrkfile analytic 0
    setYMLScalar $zrkfile iterative 1
    setYMLScalar $zrkfile reactor_timing_log ${outputs_dir}/${name}_dense_gpu.log
    setYMLScalar $infile reactor_history_file_prefix ${outputs_dir}/${name}_dense_gpu
    $zerork_exe $infile >& ${outputs_dir}/${name}_dense_gpu.stdout
  fi

  rm $infile
  rm $zrkfile
}

# Compare temperature (col 3) and pressure (col 4) of two .hist files
# row by row; fail if any relative difference exceeds TOLERANCE.
compareHist() {
  cpu_file=$1
  gpu_file=$2

  for f in "$cpu_file" "$gpu_file"
  do
    if [ ! -s "$f" ]; then
      echo "FAIL: missing or empty history file: $f"
      return 1
    fi
  done

  n_cpu=$(wc -l < "$cpu_file")
  n_gpu=$(wc -l < "$gpu_file")
  if [ "$n_cpu" -ne "$n_gpu" ]; then
    echo "FAIL: row count mismatch: $cpu_file has $n_cpu lines, $gpu_file has $n_gpu lines"
    return 1
  fi

  paste "$cpu_file" "$gpu_file" | awk -v tol=${TOLERANCE} '
    /^#/ { next }
    {
      half = NF / 2
      t1 = $3
      p1 = $4
      t2 = $(half + 3)
      p2 = $(half + 4)
      dt = (t2 - t1) / t1
      dp = (p2 - p1) / p1
      if (dt < 0) dt = -dt
      if (dp < 0) dp = -dp
      printf "step %4s  T: %g vs %g (rel %.3e)  P: %g vs %g (rel %.3e)\n", $1, t1, t2, dt, p1, p2, dp
      if (dt > tol || dp > tol) fail = 1
    }
    END {
      if (fail) {
        print "FAIL: temperature/pressure relative difference exceeds " tol
        exit 1
      }
      print "PASS: temperature and pressure agree within " tol
    }'
}


nm=h2
mf=$PWD/../../mechanisms/hydrogen/h2_v1b_mech.txt
tf=$PWD/../../mechanisms/hydrogen/h2_v1a_therm.txt
st=3.20e-5
idt=8.2565998e-07

runSims $nm $mf $tf $st $idt Y

echo "Comparing CPU vs GPU h2 dense histories (ZERORK_REACTOR_USE_LU=${ZERORK_REACTOR_USE_LU})"
compareHist ${outputs_dir}/${nm}_dense_000.hist ${outputs_dir}/${nm}_dense_gpu_000.hist
