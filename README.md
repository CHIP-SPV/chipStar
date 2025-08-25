# chipStar

![Unit Tests Intel GPUs](https://github.com/CHIP-SPV/chipStar/workflows/Unit%20Tests%20Intel%20GPUs/badge.svg)
![Unit Tests ARM GPUs](https://github.com/CHIP-SPV/chipStar/workflows/Unit%20Tests%20ARM%20GPUs/badge.svg)
![Docker Build and Publish](https://github.com/CHIP-SPV/chipStar/workflows/Docker%20Build%20and%20Publish/badge.svg)

chipStar enables compiling and running HIP and CUDA applications on platforms which support
SPIR-V as the device intermediate representation. It supports
OpenCL and Level Zero as the low-level runtime alternatives.

* [User documentation](docs/Using.md)
* [Developer documentation](docs/Development.md)
* [A list of (un)supported features](docs/Features.md)

chipStar was initially built by combining the prototyping work done in the (now obsolete) [HIPCL](https://github.com/cpc/hipcl) and
[HIPLZ](https://github.com/jz10/anl-gt-gpu/) projects.

If you wish to cite chipStar in academic publications, please refer to the [HIPCL poster abstract](https://dl.acm.org/doi/10.1145/3388333.3388641) when discussing the OpenCL backend and/or the [HIPLZ conference paper](https://link.springer.com/chapter/10.1007/978-3-031-31209-0_15) when mentioning the Level Zero backend. The core developers of chipStar are writing a proper article of the integrated chipStar project, but it is in progress.

The name chipStar comes from `c`uda and `hip` and the word `Star` which means asterisk, a typical shell wildcard, denoting the intention to make "CUDA and HIP applications run everywhere". The project was previously called CHIP-SPV.

## Library Support

The following libraries have been ported to work on Intel GPUs via MKL:
- [hipBLAS](https://github.com/CHIP-SPV/H4I-HipBLAS) (Can be built as a part of chipStar by adding `-DCHIP_BUILD_HIPBLAS=ON`)
- [hipFFT](https://github.com/CHIP-SPV/H4I-HipFFT) (Can be built as a part of chipStar by adding `-DCHIP_BUILD_HIPFFT=ON`)
- [hipSOLVER](https://github.com/CHIP-SPV/H4I-HipSOLVER)
- [hipCUB](https://github.com/CHIP-SPV/hipCUB)

The following libraries have been ported and should work on any platform:
- [rocRAND](https://github.com/CHIP-SPV/rocRAND)
- [rocPRIM](https://github.com/CHIP-SPV/rocPRIM)

*If there is a library that you need that is not yet supported, please open an issue stating which libraries you require and what application you are trying to build.*

## Applications

chipStar has so far been tested using the following applications:
- [libCEED](https://github.com/CHIP-SPV/libCEED) Our fork includes some workarounds. 
- [GAMESS](https://www.msg.chem.iastate.edu/gamess/) Source code is not public.
- [HeCBench](https://github.com/zjin-lcf/HeCBench) CUDA Benchmarks. 

## Getting Started

Quickest way to get started is by using a prebuilt Docker container. Please refer to [Docker README](docker/docker.md)
If you want to build everything yourself, you can follow a detailed [Getting Started](docs/Getting_Started.md)

## Development Status and Maturity

While chipStar 1.1 can already be used to run various large HPC applications successfully, it is still heavily in development mode with plenty of known issues and unimplemented features. There are also known low-performance optimizations that are still to be done. However, we consider chipStar ready for wider-range testing and welcome community contributions in form of reproducible bug reports and good quality pull requests.

Release notes for [1.1](docs/release_notes/chipStar_1.1.rst), [1.0](docs/release_notes/chipStar_1.0.rst) and [0.9](docs/release_notes/release-0.9.txt).

## Prerequisites

* Cmake >= 3.20.0
* Clang and LLVM 17, 18, or 19 (Clang/LLVM 15 and 16 might also work)
  * Can be installed, for example, by adding the [LLVM's Debian/Ubuntu repository](https://apt.llvm.org/) and installing packages 'clang-17 llvm-17 clang-tools-17'.
  * For the best results, install Clang/LLVM from a chipStar LLVM/Clang [branch](https://github.com/CHIP-SPV/llvm-project/tree/chipStar-llvm-17) which has fixes that are not yet in the LLVM upstream project. See below for a scripted way to build and install the patched versions.
* SPIRV-LLVM-Translator from a branch matching the LLVM major version:
  (e.g. llvm\_release\_170 for LLVM 17, llvm\_release\_180 for LLVM 18, llvm\_release\_190 for LLVM 19)
,  [llvm-spirv](https://github.com/KhronosGroup/SPIRV-LLVM-Translator).
  * Make sure the built llvm-spirv binary is installed into the same path as clang binary, otherwise clang might find and use a different llvm-spirv, leading to errors.
* SPIRV-Tools and SPIRV-Headers:
  * If available on the system, the system installation will be used
  * If not found, the submodules will be automatically initialized and built when needed

### Compiling Clang, LLVM and SPIRV-LLVM-Translator

It's recommended to use the chipStar fork of LLVM which has a few patches not yet upstreamed.
For this you can use a script included in the chipStar repository:

```bash
./scripts/configure_llvm.sh
Usage: ./scripts/configure_llvm.sh --version <version> --install-dir <dir> --link-type static(default)/dynamic --only-necessary-spirv-exts <on|off> --binutils-header-location <path>
--version: LLVM version 17, 18, 19 or 20
--install-dir: installation directory
--link-type: static or dynamic (default: static)
--only-necessary-spirv-exts: on or off (default: off)
--binutils-header-location: path to binutils header (default: empty)

./scripts/configure_llvm.sh --version 17 --install-dir /opt/install/llvm/17.0
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

You can also compile and install hipBLAS by adding `-DCHIP_BUILD_HIPBLAS=ON`

NOTE: If you don't have libOpenCL.so (for example from the `ocl-icd-opencl-dev` package), but only libOpenCL.so.1 installed, CMake fails to find it and disables the OpenCL backend. This [issue](https://github.com/CHIP-SPV/chipStar/issues/542) describes a workaround.

### Building on ARM + Mali

To build chipStar for use with an ARM Mali G52 GPU, use these steps:

1) build LLVM and SPIRV-LLVM-Translator as described above

2) build chipStar with -DCHIP_MALI_GPU_WORKAROUNDS=ON cmake option

There are some limitations - kernels using double type will not work,
and kernels using subgroups may not work.

Note that chipStar relies on the proprietary OpenCL implementation
provided by ARM. We have successfully managed to compile and run
chipStar with an Odroid N2 device, using Ubuntu 22.04.2 LTS,
with driver version OpenCL 3.0 v1.r40p0-01eac0.

### Building on RISC-V + PowerVR

To build chipStar for use with a PowerVR GPU, the default steps can be followed.
There is an automatic workaround applied for an [issue](https://github.com/CHIP-SPV/chipStar/pull/828) in PowerVR's OpenCL implementation.

There are some limitations: kernels using double type will not work,
kernels using subgroups may not work, you may also run into unexpected
OpenCL errors like CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST and
other issues.

Note that chipStar relies on the proprietary OpenCL implementation
provided by Imagination Technologies. We have successfully managed to
compile and run chipStar with a VisionFive2 device, using VisionFive2's
pre-built Debian image 202403, driver version 1.19. Other SBCs may require
additional workarounds.

## Running Unit Tests

There's a script `check.py` which can be used to run unit tests and which filters out known failing tests for different platforms. Its usage is as follows.

```bash
BUILD_DIR={path to build directory. Make sure that build_tests target has been built}

BACKEND={opencl/level0}
^ Which backend/driver/platform you wish to test:
"opencl" = Intel OpenCL runtime, "level0" = Intel LevelZero runtime 

DEVICE={cpu,igpu,dgpu,pocl}         # What kind of device to test.
^ This selects the expected test pass lists.
  'igpu' is a Intel Iris Xe iGPU, 'dgpu' a typical recent Intel dGPU such as Data Center GPU Max series or an Arc.

export CHIP_PLATFORM=N         # If there are multiple OpenCL platforms present on the system, selects which one to use.

You can always verify which device is being used by chipStar by:
CHIP_LOGLEVEL=info ./build/hipInfo
```

```
python3 $SOURCE_DIR/scripts/check.py $BUILD_DIR $DEVICE $BACKEND
```

Please refer to the [user documentation](docs/Using.md) for instructions on how to use the installed chipStar to build CUDA/HIP programs.

## Tools

chipStar includes several utility tools for working with SPIR-V binaries and OpenCL compilation. These tools are built as part of the chipStar build process and installed in the `bin` directory.

### SPIR-V Extractor Tool

The `spirv-extractor` tool extracts and validates SPIR-V binaries from HIP fatbinaries.

#### Usage

```bash
spirv-extractor [--check-for-doubles] [--validate] [-o <output_filename>] [-h] <fatbinary_path> [<additional_args>...]
```

#### Options

- `--check-for-doubles`: Check if SPIR-V uses double precision and skip test if so
- `--validate`: Perform comprehensive SPIR-V verification (syntax) and validation (spec compliance)
- `-o <filename>`: Output SPIR-V to specified file (both binary .spv and text .txt formats)
- `-h`: Show help message

#### Examples

Extract SPIR-V from a fatbinary and display as text:
```bash
./build/bin/spirv-extractor my_kernel.fatbin
```

Extract and save SPIR-V to files:
```bash
./build/bin/spirv-extractor -o kernel.spv my_kernel.fatbin
```

Validate SPIR-V compliance:
```bash
./build/bin/spirv-extractor --validate my_kernel.fatbin
```

Check for double precision usage (used internally by test framework):
```bash
./build/bin/spirv-extractor --check-for-doubles my_kernel.fatbin
```

The validation feature performs both SPIR-V specification validation and HIP-specific constraint checking, including verification of memory models, execution models, capabilities, and other requirements for HIP kernel compatibility.

### OpenCL SPIR-V Compiler Tool

The `opencl-spirv-compiler` tool compiles SPIR-V binaries or assembly files using the OpenCL runtime to generate device-specific binaries.

#### Usage

```bash
opencl-spirv-compiler <file(s) or directory>
```

#### Features

- Accepts both SPIR-V binary files (.spv) and SPIR-V assembly files (.spvasm)
- Automatically detects file type using the `file` command
- Can process individual files or entire directories recursively
- Generates device-specific binary files with `_device.bin` suffix
- Provides detailed error messages including OpenCL build logs

#### Examples

Compile a single SPIR-V binary:
```bash
./build/bin/opencl-spirv-compiler kernel.spv
```

Compile a SPIR-V assembly file:
```bash
./build/bin/opencl-spirv-compiler kernel.spvasm
```

Process all SPIR-V files in a directory:
```bash
./build/bin/opencl-spirv-compiler /path/to/spirv/files/
```

The tool will create output files with the naming pattern `<input_filename>_device.bin` containing the compiled device-specific binary code.

## Environment Variables

```bash
CHIP_BE=<opencl/level0>                         # Selects the backend to use. If both Level Zero and OpenCL are available, Level Zero is used by default
CHIP_PLATFORM=<N>                               # If there are multiple platforms present on the system, selects which one to use. Defaults to 0
CHIP_DEVICE=<N>                                 # If there are multiple devices present on the system, selects which one to use. Defaults to 0
CHIP_DEVICE_TYPE=<gpu/cpu/accel/fpga/pocl> or empty  # Selects which type of device to use. Cannot be used with CHIP_PLATFORM/CHIP_DEVICE. Defaults to empty.
CHIP_LOGLEVEL=<trace/debug/info/warn/err/crit>  # Sets the log level. If compiled in RELEASE, only err/crit are available
CHIP_DUMP_SPIRV=<ON/OFF(default)>               # Dumps the generated SPIR-V code to a file
CHIP_JIT_FLAGS=<flags>                          # Additional JIT flags
CHIP_L0_COLLECT_EVENTS_TIMEOUT=<N(30s default)> # Timeout in seconds for collecting Level Zero events
CHIP_L0_EVENT_TIMEOUT=<N(0 default)             # Timeout in seconds for how long Level Zero should wait on an event before timing out
CHIP_SKIP_UNINIT=<ON/OFF(default)>              # If enabled, skips the uninitialization of chipStar's backend objects at program termination
CHIP_MODULE_CACHE_DIR=/path/to/desired/dir      # Module/Program cache dir. Defaults to $HOME/.cache/chipStar, if caching is undesired, set to empty string i.e. export CHIP_MODULE_CACHE_DIR=
CHIP_VERIFY_MODE=<off/failures/all>             # Controls LLVM IR and SPIR-V verification output during compilation. 'off' disables verification, 'failures' (default) shows table only when SPIR-V validation fails, 'all' always shows the verification table
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

If you device does not support emulation, you can skip these tests providing `-DSKIP_TESTS_WITH_DOUBLES=ON` option at cmake configure time.
