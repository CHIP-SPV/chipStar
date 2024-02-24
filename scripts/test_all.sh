#!/bin/bash
num_tries=100
num_threads=20
source /usr/local/Modules/init/bash

# OpenCL dGPU
module purge && module load opencl/dgpu && ctest --timeout 200 -j $num_threads --repeat until-fail:$num_tries -O alltest_opencl_dgpu.log
awk '/The following tests FAILED/,/\[ERROR_MESSAGE\]/' alltest_opencl_dgpu.log | sed '/\[ERROR_MESSAGE\]/d' > alltest_opencl_dgpu_failed.log

# OpenCL iGPU
module purge && module load opencl/igpu && ctest --timeout 200 -j $num_threads --repeat until-fail:$num_tries -O alltest_opencl_igpu.log
awk '/The following tests FAILED/,/\[ERROR_MESSAGE\]/' alltest_opencl_igpu.log | sed '/\[ERROR_MESSAGE\]/d' > alltest_opencl_igpu_failed.log

# Level Zero dGPU ICL
module purge && module load level-zero/dgpu && ctest --timeout 200 -j $num_threads --repeat until-fail:$num_tries -O alltest_level0icl_dgpu.log
awk '/The following tests FAILED/,/\[ERROR_MESSAGE\]/' alltest_level0_dgpu.log | sed '/\[ERROR_MESSAGE\]/d' > alltest_level0icl_dgpu_failed.log

# Level Zero iGPU ICL
module purge && module load level-zero/igpu && ctest --timeout 200 -j $num_threads --repeat until-fail:$num_tries -O alltest_level0icl_igpu.log
awk '/The following tests FAILED/,/\[ERROR_MESSAGE\]/' alltest_level0_igpu.log | sed '/\[ERROR_MESSAGE\]/d' > alltest_level0icl_igpu_failed.log

# Level Zero dGPU RCL
module purge && module load level-zero/dgpu && CHIP_L0_IMM_CMD_LISTS=OFF ctest --timeout 200 -j $num_threads --repeat until-fail:$num_tries -O alltest_level0rcl_dgpu.log
awk '/The following tests FAILED/,/\[ERROR_MESSAGE\]/' alltest_level0_dgpu.log | sed '/\[ERROR_MESSAGE\]/d' > alltest_level0rcl_dgpu_failed.log

# Level Zero iGPU RCL
module purge && module load level-zero/igpu && CHIP_L0_IMM_CMD_LISTS=OFF ctest --timeout 200 -j $num_threads --repeat until-fail:$num_tries -O alltest_level0rcl_igpu.log
awk '/The following tests FAILED/,/\[ERROR_MESSAGE\]/' alltest_level0_igpu.log | sed '/\[ERROR_MESSAGE\]/d' > alltest_level0rcl_igpu_failed.log

# OpenCL CPU
module purge && module load oneapi/compiler/latest && CHIP_BE=opencl CHIP_DEVICE_TYPE=cpu CHIP_PLATFORM=1 ctest --timeout 200 -j $num_threads --repeat until-fail:$num_tries -O alltest_opencl_cpu.log
awk '/The following tests FAILED/,/\[ERROR_MESSAGE\]/' alltest_opencl_cpu.log | sed '/\[ERROR_MESSAGE\]/d' > alltest_opencl_cpu_failed.log

# OpenCL PoCL
module purge && module load pocl/5.0 && ctest --timeout 200 -j $num_threads --repeat until-fail:$num_tries -O alltest_pocl.log
awk '/The following tests FAILED/,/\[ERROR_MESSAGE\]/' alltest_pocl.log | sed '/\[ERROR_MESSAGE\]/d' > alltest_pocl_failed.log
