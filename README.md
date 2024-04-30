# chipStar

chipStar enables porting HIP and CUDA applications to platforms which support
SPIR-V as the device intermediate representation. It supports
OpenCL and Level Zero as the low-level runtime alternatives.

* [User documentation](docs/Using.md)
* [Developer documentation](docs/Development.md)
* [A list of (un)supported features](docs/Features.md)

chipStar was initially built by combining the prototyping work done in the (now obsolete) [HIPCL](https://github.com/cpc/hipcl) and
[HIPLZ](https://github.com/jz10/anl-gt-gpu/) projects.

If you wish to cite chipStar in academic publications, please refer to the [HIPCL poster abstract](https://dl.acm.org/doi/10.1145/3388333.3388641) when discussing the OpenCL backend and/or the [HIPLZ conference paper](https://link.springer.com/chapter/10.1007/978-3-031-31209-0_15) when mentioning the Level Zero backend. The core developers of chipStar are writing a proper article of the integrated chipStar project, but it is in progress.

The name chipStar comes from `c`uda and `hip` and the word `Star` which means asterisk, a typical shell wildcard, denoting the intention to make "CUDA and HIP applications run everywhere". The project was previously called CHIP-SPV.

## Development Status and Maturity

While chipStar 1.1 can already be used to run various large HPC applications successfully, it is still heavily in development mode with plenty of known issues and unimplemented features. There are also known low-performance optimizations that are still to be done. However, we consider chipStar ready for wider-range testing and welcome community contributions in form of reproducible bug reports and good quality pull requests.

Release notes for [1.1](docs/release_notes/chipStar_1.1.rst), [1.0](docs/release_notes/chipStar_1.0.rst) and [0.9](docs/release_notes/release-0.9.txt).

## Prerequisites

* Cmake >= 3.20.0
* Clang and LLVM 17 (Clang/LLVM 15 and 16 might also work)
  * Can be installed, for example, by adding the [LLVM's Debian/Ubuntu repository](https://apt.llvm.org/) and installing packages 'clang-17 llvm-17 clang-tools-17'.
  * For the best results, install Clang/LLVM from a chipStar LLVM/Clang [branch](https://github.com/CHIP-SPV/llvm-project/tree/chipStar-llvm-17) which has fixes that are not yet in the LLVM upstream project. See below for a scripted way to build and install the patched versions.
* SPIRV-LLVM-Translator from a branch matching the LLVM major version:
  (e.g. llvm\_release\_170 for LLVM 17)
,  [llvm-spirv](https://github.com/KhronosGroup/SPIRV-LLVM-Translator).
  * Make sure the built llvm-spirv binary is installed into the same path as clang binary, otherwise clang might find and use a different llvm-spirv, leading to errors.

### Compiling Clang, LLVM and SPIRV-LLVM-Translator

It's recommended to use the chipStar fork of LLVM which has a few patches not yet upstreamed.
For this you can use a script included in the chipStar repository:

```bash
# chipStar/scripts/configure_llvm.sh <version 15/16/17> <install_dir> <static/dynamic>
chipStar/scripts/configure_llvm.sh 17 /opt/install/llvm/17.0 dynamic on
cd llvm-project/llvm/build_17
make -j 16
<sudo> make install
```

Or you can do the steps manually:

```bash
git clone --depth 1 https://github.com/CHIP-SPV/llvm-project.git -b chipStar-llvm-17
cd llvm-project/llvm/projects
git clone --depth 1 https://github.com/CHIP-SPV/SPIRV-LLVM-Translator.git -b chipStar-llvm-17
cd ../..

# DLLVM_ENABLE_PROJECTS="clang;openmp" OpenMP is optional but many apps use it
# DLLVM_TARGETS_TO_BUILD Speed up compilation by building only the necessary CPU host target
# CMAKE_INSTALL_PREFIX Where to install LLVM

cmake -S llvm -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;openmp" \
  -DLLVM_TARGETS_TO_BUILD=X86 \
  -DCMAKE_INSTALL_PREFIX=$HOME/local/llvm-17
make -C build -j8 all install
```

### OpenCL Backend

  * An OpenCL 2.0 or 3.0 driver with at least the following features supported:
    * Coarse-grained buffer shared virtual memory (SVM)
    * SPIR-V input
    * Generic address space
    * Program scope variables
  * Further OpenCL extensions or features might be needed depending on the compiled CUDA/HIP application. For example, to support warp-primitives, the OpenCL driver should support also additional subgroup features such as shuffles, ballots and [cl_intel_required_subgroup_size]( https://registry.khronos.org/OpenCL/extensions/intel,/cl_intel_required_subgroup_size.html).

### Level Zero Backend

  * [Intel Compute Runtime](https://github.com/intel/compute-runtime) or [oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
  * [oneAPI Level Zero Loader](https://github.com/oneapi-src/level-zero/releases)
* For HIP-SYCL and HIP-MKL Interoperability: [oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)

## Downloading Sources

You can download and unpack the latest released source package or clone the development branch via git. We aim to keep the `main` development branch stable, but it might have stability issues during the development cycle.

To clone the sources from Github:

```bash
git clone https://github.com/CHIP-SPV/chipStar.git
cd chipStar
git submodule update --init --recursive
```

## Building and Installing

```bash
mkdir build && cd build

# LLVM_CONFIG_BIN is optional if LLVM can be found in PATH or if not using a version-sufficed
# binary (for example, llvm-config-17)

cmake .. \
    -DLLVM_CONFIG_BIN=/path/to/llvm-config
    -DCMAKE_INSTALL_PREFIX=/path/to/install
make all build_tests install -j8
```

NOTE: If you don't have libOpenCL.so (for example from the `ocl-icd-opencl-dev` package), but only libOpenCL.so.1 installed, CMake fails to find it and disables the OpenCL backend. This [issue](https://github.com/CHIP-SPV/chipStar/issues/542) describes a workaround.

### Building on ARM + Mali

To build chipStar for use with an ARM Mali G52 GPU, use these steps:

1) build LLVM and SPIRV-LLVM-Translator as described above

2) build chipStar with -DCHIP_MALI_GPU_WORKAROUNDS=ON cmake option

There are some limitations - kernels using double type will not work,
and kernels using subgroups may not work.

Note that chipStar relies on proprietary OpenCL implementation
provided by ARM. We have successfully managed to compile & run
chipStar with an Odroid N2 device, using Ubuntu 22.04.2 LTS,
with driver version OpenCL 3.0 v1.r40p0-01eac0.

### Building on RISC-V + PowerVR

To build chipStar for use with a PowerVR GPU, build using the normal steps.
There is an automatic workaround applied for PowerVR OpenCL implementation.

There are some limitations - kernels using double type will not work,
kernels using subgroups may not work, you may also run into unexpected
OpenCL errors like CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST.

Note that chipStar relies on proprietary OpenCL implementation
provided by Imagination Technologies. We have successfully managed to
compile & run chipStar with a VisionFive2 device, using VisionFive2's
pre-built Debian image 202403, driver version 1.19. Other SBCs may require
additional workarounds.

## Running Unit Tests

There's a script `check.py` which can be used to run unit tests and which filters out known failing tests for different platforms. Its usage is as follows.

```bash
# BACKEND={opencl/level0-{reg,imm}/pocl}
# ^ Which backend/driver/platform you wish to test:
# "opencl" = Intel OpenCL runtime, "level0" = Intel LevelZero runtime with regular command lists (reg) or immediate command lists (imm), "pocl" = PoCL OpenCL runtime
# DEVICE={cpu,igpu,dgpu}         # What kind of device to test.
# ^ This selects the expected test pass lists.
#   'igpu' is a Intel Iris Xe iGPU, 'dgpu' a typical recent Intel dGPU such as Data Center GPU Max series or an Arc.
# PARALLEL={N}                   # How many tests to run in parallel.
# export CHIP_PLATFORM=N         # If there are multiple OpenCL platforms present on the system, selects which one to use

python3 $SOURCE_DIR/scripts/check.py -m off --num-threads $PARALLEL $BUILD_DIR $DEVICE $BACKEND
```

Please refer to the [user documentation](docs/Using.md) for instructions on how to use the installed chipStar to build CUDA/HIP programs.

## Environment Variables

```bash
CHIP_BE=<opencl/level0>                         # Selects the backend to use. If both Level Zero and OpenCL are available, Level Zero is used by default
CHIP_PLATFORM=<N>                               # If there are multiple platforms present on the system, selects which one to use. Defaults to 0
CHIP_DEVICE=<N>                                 # If there are multiple devices present on the system, selects which one to use. Defaults to 0
CHIP_DEVICE_TYPE=<gpu/cpu/accel/fpga> or empty  # Selects which type of device to use. Defaults to empty.
CHIP_LOGLEVEL=<trace/debug/info/warn/err/crit>  # Sets the log level. If compiled in RELEASE, only err/crit are available
CHIP_DUMP_SPIRV=<ON/OFF(default)>               # Dumps the generated SPIR-V code to a file
CHIP_JIT_FLAGS=<flags>                          # String to override the default JIT flags. Defaults to -cl-kernel-arg-info -cl-std=CL3.0
CHIP_L0_COLLECT_EVENTS_TIMEOUT=<N(30s default)> # Timeout in seconds for collecting Level Zero events
CHIP_SKIP_UNINIT=<ON/OFF(default)>              # If enabled, skips the uninitialization of chipStar's backend objects at program termination
```

Example:
```bash
╭─pvelesko@cupcake ~
╰─$ clinfo -l
Platform #0: Intel(R) OpenCL Graphics
 `-- Device #0: Intel(R) Arc(TM) A380 Graphics
Platform #1: Intel(R) OpenCL Graphics
 `-- Device #0: Intel(R) UHD Graphics 770
```

Based on these values, if we want to run on OpenCL iGPU:
```bash
export CHIP_BE=opencl
export CHIP_PLATFORM=1
export CHIP_DEVICE=0
```

*NOTE: Level Zero doesn't have a clinfo equivalent. Normally if you have more than one Level Zero device, there will only be a single platform so set CHIP_PLATFORM=0 and then CHIP_DEVICE to the device you want to use.*
*You can check the name of the device by running a sample which prints the name such as `build/samples/0_MatrixMultiply/MatrixMultiply`

## Troubleshooting

### Clang++ Cannot Find libstdc++ When Building chipStar

This occurs often when the latest installed GCC version doesn't include libstdc++, and Clang++ by default chooses the latest found one regardless, and ends up failing to link C++ programs. The problem is discussed [here](https://discourse.llvm.org/t/add-gcc-install-dir-deprecate-gcc-toolchain-and-remove-gcc-install-prefix/65091/14).

The issue can be resolved by defining a Clang++ [configuration file](https://clang.llvm.org/docs/UsersManual.html#configuration-files) which forces the GCC to what we want. Example:

```bash
echo --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11 > ~/local/llvm-17/bin/x86_64-unknown-linux-gnu-clang++.cfg
```

### Missing Double Precision Support

When running the tests on OpenCL devices which do not support double precision floats,
there will be multiple tests that will error out.

It might be possible to enable software emulation of double precision floats for
Intel iGPUs by setting [two environment variables](https://github.com/intel/compute-runtime/blob/master/opencl/doc/FAQ.md#feature-double-precision-emulation-fp64) to make kernels using doubles work but with the major
overhead of software emulation:

```bash
export IGC_EnableDPEmulation=1
export OverrideDefaultFP64Settings=1
```
