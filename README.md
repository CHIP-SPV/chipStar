# CHIP-SPV

CHIP-SPV is a HIP implementation that abstracts HIP API, providing a
set of base classes that can be derived from to implement an
additional, SPIR-V capable backend on which to execute HIP
calls.

Currently CHIP-SPV supports OpenCL and Level Zero as backend alternatives.

This project is an integration of [HIPCL](https://github.com/cpc/hipcl) and
[HIPLZ](https://github.com/jz10/anl-gt-gpu/) projects.

## Prerequisites

* Cmake >= 3.18.0
* Clang >= 14.0
* SPIRV-LLVM-Translator from a branch matching to the clang version:
  (e.g. llvm\_release\_140 for Clang 14.0)
  [llvm-spirv](https://github.com/KhronosGroup/SPIRV-LLVM-Translator)
* For Level Zero Backend
  * [Intel Compute Runtime](https://github.com/intel/compute-runtime)
* For OpenCL Backend
  * An OpenCL implementation with (at least partial) 2.x support;
    HIPCL requires Shared Virtual Memory and clCreateProgramWithIL()
    support

## Downloading and Building Clang

Downloading:

```bash
git clone -b release/14.x https://github.com/llvm/llvm-project.git
cd llvm-project/llvm/projects
git clone -b llvm_release_140 https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
```

Building:

```bash
cd llvm-project/llvm
mkdir build
cd build
cmake .. -DLLVM_ENABLE_PROJECTS="clang;openmp" \ # openmp is optional
  -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_DIR} \
  -DLLVM_TARGETS_TO_BUILD=host # Also optional, but if "All" is selected, potential issues can arise. Furthermore, speeds up compilation time if limited to just the architectures you actually use.
make
make install
```

## Downloading Sources

```bash
git clone https://github.com/CHIP-SPV/chip-spv.git
cd chip-spv
git submodule update --init --recursive
```

## Building

```bash
# export PATH=${LLVM_INSTALL_DIR}/bin:$PATH
mkdir build
cd build

cmake .. \
 -DCMAKE_CXX_COMPILER=clang++ \
 -DCMAKE_C_COMPILER=clang \
 -DCMAKE_INSTALL_PREFIX=<install location> 
 # optional: -DCMAKE_BUILD_TYPE=<Debug(default), Release, RelWithDebInfo>

make
make install
```

## Building & Running Unit Tests

```bash
make build_tests_standalone
ctest --timeout 120 # currently some tests might hang
```

## Environment Flags

### CHIP_BE

Select the backend to execute on.
Possible values: opencl(default), level0

### CHIP_LOGLEVEL

Select the verbosity of debug info during execution.
Possible values: trace, debug(default), warn, err, crit

Setting this value to `debug` will print information coming from the CHIP-SPV functions which are shared between the backends.

Settings this value to `trace` will print `debug`, as well as debug infomarmation from the backend implementation itself such as results from low-level Level Zero API calls.

### HIP_PLATFORM

Select which HIP implementation to execute on. Possible values: amd, nvidia, spirv.
If you do not provide this value, `hipcc` will check for existance of the following directions and try to guess which implementation is available:

```bash
/usr/local/cuda
/opt/rocm
```

## Compiling a HIP application using CHIP-SPV

Compiling a HIP application with CHIP-SPV will allow you to execute HIP code on any device that supports SPIR-V, such as Intel GPUs. To compile a HIP application, all you need to do is to use the `hipcc` compiler wrapper provided by this project. In case you have AMD implementation installed as well, you can switch between them by using `HIP_PLATFORM` environment variable.

You can find various HIP applications here: <https://github.com/CHIP-SPV/hip-testsuite>

```bash
hipcc ./hip_app.cpp -o hip_app
```

## Using CHIP-SPV With CMake

CHIP-SPV provides a `FindHIP.cmake` module so you can verify that HIP is installed:

```bash
list(APPEND CMAKE_MODULES_PREFIX <CHIP-SPV install location>/cmake)
find_package(HIP REQUIRED)
```
