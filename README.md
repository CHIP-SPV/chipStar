# CHIP-SPV

CHIP-SPV is a HIP implementation that abstracts HIP API, providing a set of base classes that can be derived from to implement an additional, SPIR-V capable backend on which to execute HIP calls. Currently CHIP-SPV OpenCL and Level Zero as backends.
This project is a result of [HIPCL](https://github.com/cpc/hipcl) and [HIPLZ](https://github.com/jz10/anl-gt-gpu/) projects.

## Prerequisites

* Compiled HIPLZ required for LLVM passes (temporary)
* Cmake > 3.4.3
* Clang with SPIR-V patches: [hipcl-clang](https://github.com/cpc/hipcl-clang)
* For Level Zero Backend
  * [Intel Compute Runtime](https://github.com/intel/compute-runtime)
* For OpenCL Backend
  * An OpenCL implementation with (at least partial) 2.x support; HIPCL requires Shared Virtual Memory and clCreateProgramWithIL() support

## Building

````bash
# export PATH=${PATH_TO_CLANG_SPIRV}:$PATH
cd CHIP-SPV
mkdir build

cd build

cmake ../ \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DHIP_SPIRV_DIR=/home/pvelesko/install/HIPLZ
```
