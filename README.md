# CHIP-SPV

CHIP-SPV is a HIP implementation that abstracts HIP API, providing a
set of base classes that can be derived from to implement an
additional, SPIR-V capable backend on which to execute HIP
calls. Currently CHIP-SPV OpenCL and Level Zero as backends.  This
project is a result of [HIPCL](https://github.com/cpc/hipcl) and
[HIPLZ](https://github.com/jz10/anl-gt-gpu/) projects.

## Prerequisites

* Cmake >= 3.18.0
* Clang with SPIR-V patches: [hipcl-clang](https://github.com/parmance/llvm-project)
* For Level Zero Backend
  * [Intel Compute Runtime](https://github.com/intel/compute-runtime)
* For OpenCL Backend
  * An OpenCL implementation with (at least partial) 2.x support;
    HIPCL requires Shared Virtual Memory and clCreateProgramWithIL()
    support

## Downloading Sources

```bash
git clone https://github.com/CHIP-SPV/chip-spv.git
cd chip-spv
git submodule update --init --recursive
```

## Building

```bash
# export PATH=${PATH_TO_CLANG_SPIRV}:$PATH
mkdir build
cd build

cmake .. -DCMAKE_CXX_COMPILER=clang++
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
