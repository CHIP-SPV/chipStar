#!/bin/bash

set -e

echo '#include <sys/mman.h>
#include <stdlib.h>
int main() { void *p = malloc(4096); return mlock(p, 4096); }' | gcc -x c - -o /tmp/mlocktest && /tmp/mlocktest && echo "Page locking works" || { echo "Page locking failed"; exit 1; }

export CHIP_MODULE_CACHE_DIR=""
host=`hostname`
echo "Running on ${host}"
# If not on Salami read the file /opt/actions-runner/num-threads.txt and set the number of threads to the value in the file
# if the file does not exist, set the number of threads to 24
if [ -f "/opt/actions-runner/num-threads.txt" ]; then
  num_threads=$(cat /opt/actions-runner/num-threads.txt)
else
  num_threads=$(nproc)
fi

num_tries=1
# deafult timeout is 30 minutes
timeout=1800
build_only=false

rm -rf ~/.cache/chipStar

# Check if at least one argument is provided
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <debug|release> <llvm-16|llvm-17|llvm-18|llvm-19|llvm-20|llvm-21> [--skip-build] [--build-only] [--num-tries=$num_tries] [--num-threads=$num_threads] [--timeout=$timeout]"
  exit 1
fi

# Check if the first argument is either "debug" or "release"
if [ "$1" != "debug" ] && [ "$1" != "release" ]; then
  echo "Error: Invalid argument. Must be either 'debug' or 'release'."
  exit 1
fi

# Set the build type based on the argument
build_type=$(echo "$1" | tr '[:lower:]' '[:upper:]')

# Check if the second argument starts with "llvm-" and is followed by a valid version number
if [[ ! "$2" =~ ^llvm-(1[6-9]|[2-9][0-9])$ ]]; then
  echo "Error: Invalid LLVM version. Must be llvm-16, llvm-17, llvm-18, llvm-19, llvm-20, llvm-21, or higher."
  exit 1
fi

CLANG="llvm/${2#llvm-}"

shift
shift

for arg in "$@"
do
  case $arg in
    --num-threads=*)
      num_threads="${arg#*=}"
      shift
      ;;
    --num-threads)
      shift
      num_threads="$1"
      shift
      ;;
    --num-tries=*)
      num_tries="${arg#*=}"
      shift
      ;;
    --num-tries)
      shift
      num_tries="$1"
      shift
      ;;
    --skip-build)
      skip_build=true
      shift
      ;;
    --build-only)
      build_only=true
      shift
      ;;
    --timeout=*)
      timeout="${arg#*=}"
      shift
      ;;
    --timeout)
      shift
      timeout="$1"
      shift
      ;;
    *)
      ;;
  esac
done

# Function to detect CMake generator and set build tool
detect_build_tool() {
  if [ -f "CMakeCache.txt" ]; then
    generator=$(grep "CMAKE_GENERATOR:INTERNAL=" CMakeCache.txt | cut -d'=' -f2)
    case "$generator" in
      "Ninja")
        BUILD_TOOL="ninja"
        ;;
      "Unix Makefiles")
        BUILD_TOOL="make"
        ;;
      *)
        echo "Warning: Unknown generator '$generator', defaulting to make"
        BUILD_TOOL="make"
        ;;
    esac
  else
    echo "Warning: CMakeCache.txt not found, defaulting to make"
    BUILD_TOOL="make"
  fi
  echo "Detected CMake generator: $generator, using build tool: $BUILD_TOOL"
}

# Print out the arguments
echo "build_type  = ${build_type}"
echo "CLANG       = ${CLANG}"
echo "num_tries   = ${num_tries}"
echo "num_threads = ${num_threads}"
echo "skip_build  = ${skip_build}"
echo "build_only  = ${build_only}"
echo "timeout     = ${timeout}"

# source /opt/intel/oneapi/setvars.sh intel64 &> /dev/null
# Source Environment Modules init. Prefer user-local install (5.4.0, supports
# pushenv) over system apt package (5.0.1, does not).
if [ -f "$HOME/.local/init/bash" ]; then
  export MODULESHOME=$HOME/.local
  source "$MODULESHOME/init/bash"
elif [ -f /etc/profile.d/lmod.sh ]; then
  source /etc/profile.d/lmod.sh &> /dev/null
else
  source /etc/profile.d/modules.sh &> /dev/null
fi
export IGC_EnableDPEmulation=1
export OverrideDefaultFP64Settings=1
export CHIP_LOGLEVEL=err
export POCL_KERNEL_CACHE=0

# set event timeout to be 10 less than timeout
export CHIP_L0_EVENT_TIMEOUT=$(($timeout - 10))

# Use OpenCL for building/test discovery to prevent Level Zero from being used in multi-thread/multi-process environment
if [ "$host" = "salami" ]; then
  module use  ~/modulefiles
  module load $CLANG
else
  module use ~/modulefiles
  module load oneapi/2025.0.4 $CLANG opencl/dgpu 
  module list
fi

unset CHIP_PLATFORM
unset CHIP_DEVICE

output=$(clinfo -l 2>&1 | grep "Platform #0")
echo $output
if [ $? -ne 0 ]; then
    echo "clinfo failed to execute."
    exit 1
fi

# check if the output is empty
if [ -z "$output" ]; then
    echo "No OpenCL devices detected."
    exit 1
fi

if [ $skip_build ]; then
  echo "Skipping build step"
  cd build
  detect_build_tool
else
  if [ "$host" = "salami" ]; then
    CHIP_OPTIONS="-DCHIP_MALI_GPU_WORKAROUNDS=ON -DCHIP_SKIP_TESTS_WITH_DOUBLES=ON -DCHIP_BUILD_SAMPLES=ON -DCHIP_BUILD_TESTS=ON"
  else
    CHIP_OPTIONS="-DCHIP_BUILD_SAMPLES=ON -DCHIP_BUILD_TESTS=ON"
  fi
  # Build the project
  echo "Building project..."
  rm -rf HIPCC
  rm -rf HIP
  rm -rf bitcode/ROCm-Device-Libs
  rm -rf hip-tests
  rm -rf hip-testsuite

  git submodule update --init
  rm -rf build
  rm -rf *_result.txt
  mkdir build
  cd build

  echo "building with $CLANG"
  LLVM_CONFIG_BIN=$(module show $CLANG 2>&1 | grep -E 'prepend-path\s+PATH' | awk '{print $NF}' | head -1)/llvm-config
  cmake ../ -DLLVM_CONFIG_BIN=$LLVM_CONFIG_BIN -DCMAKE_BUILD_TYPE="$build_type"  ${CHIP_OPTIONS}
  detect_build_tool
  $BUILD_TOOL all install -j $(nproc) #&> /dev/null
  $BUILD_TOOL build_tests install -j $(nproc) #&> /dev/null
  echo "chipStar build complete." 
fi

if [ "$build_only" = true ]; then
  echo "Build-only mode. Exiting."
  exit 0
fi

module unload opencl/dgpu

# Function to run tests
run_tests() {
    local device=$1
    local backend=$2
    echo "begin ${device}_${backend}_failed_tests"
    ../scripts/check.py ./ $device $backend --num-threads=${num_threads} --timeout=$timeout --num-tries=$num_tries | tee ${device}_${backend}_make_check_result.txt
    echo "end ${device}_${backend}_failed_tests"
}

function check_tests {
  file="$1"
  if ! grep -q "The following tests FAILED" "$file"; then
    return 0
  else
    echo "Checking $file"
    grep -E "The following tests FAILED:" -A 1000 "$file" | sed '/^$/q' | tail -n +2
    return 1
  fi
}

set +e # disable exit on error

module unload opencl level-zero
module load opencl/igpu

# Run tests for different configurations
run_tests igpu opencl
module unload opencl/igpu
if [ "$host" = "salami" ]; then
  check_tests igpu_opencl_make_check_result.txt
  igpu_opencl_exit_code=$?
  exit $igpu_opencl_exit_code
fi

module load level-zero/igpu
run_tests igpu level0
module unload level-zero/igpu
module load level-zero/dgpu
run_tests dgpu level0
module unload level-zero/dgpu
module load opencl/dgpu
run_tests dgpu opencl
module unload opencl/dgpu
module load opencl/cpu
run_tests cpu opencl
module unload opencl/cpu

check_tests igpu_opencl_make_check_result.txt
igpu_opencl_exit_code=$?
check_tests cpu_opencl_make_check_result.txt
cpu_opencl_exit_code=$?
check_tests dgpu_opencl_make_check_result.txt
dgpu_opencl_exit_code=$?
check_tests dgpu_level0_make_check_result.txt
dgpu_level0_exit_code=$?
check_tests igpu_level0_make_check_result.txt
igpu_level0_exit_code=$?

exit $((igpu_opencl_exit_code || dgpu_opencl_exit_code || igpu_level0_exit_code || dgpu_level0_exit_code || cpu_opencl_exit_code))