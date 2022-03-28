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
cmake .. -DLLVM_ENABLE_PROJECTS="clang" \
  -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_DIR}
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

cmake .. -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang
make
```

## Building & Running Unit Tests

```bash
make build_tests_standalone -j
ctest --timeout 120 # currently some tests might hang
```

## Testing

Run tests on the default backend:

```bash
make test
```

Run tests on a specific backend:

```bash
CHIP_BE=<backend> make test
```

Where the `backend` is a backend identification. Possible values for it are
`level0` and `opencl`.
