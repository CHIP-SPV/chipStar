# chipStar

chipStar makes HIP and CUDA applications portable to platforms which support
SPIR-V as the device intermediate representation. Currently it supports
OpenCL and Level Zero as the low-level runtime alternatives.

For User documentation, read [this.](docs/Using.md)
For Developer documentation, read [this.](docs/Development.md)
For a list of (un)supported features, read [this.](docs/Features.md)

This project builds on work done in [HIPCL](https://github.com/cpc/hipcl) and
[HIPLZ](https://github.com/jz10/anl-gt-gpu/) projects.

## Prerequisites

* Cmake >= 3.20.0
* Clang 15 or 16
  * Can be installed, for example, by adding the [LLVM's Debian/Ubuntu repository](https://apt.llvm.org/) and installing packages 'clang-15 llvm-15 clang-tools-15'.
* SPIRV-LLVM-Translator from a branch matching to the clang version:
  (e.g. llvm\_release\_150 for Clang 15.0)
  [llvm-spirv](https://github.com/KhronosGroup/SPIRV-LLVM-Translator).
  * For best results, install [chipStar's LLVM 15 branch](https://github.com/CHIP-SPV/SPIRV-LLVM-Translator/tree/chipspv-llvm-15-patches) or [chipStar's LLVM 16 branch](https://github.com/CHIP-SPV/SPIRV-LLVM-Translator/tree/chipspv-llvm-16-patches) which have fixes that are not yet in upstream.
  * make sure the built llvm-spirv binary is installed into the same path as clang binary, otherwise clang might find and use a different llvm-spirv, leading to errors
* For Level Zero Backend
  * [Intel Compute Runtime](https://github.com/intel/compute-runtime) or [oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
  * [oneAPI Level Zero Loader](https://github.com/oneapi-src/level-zero/releases)
* For OpenCL Backend
  * OpenCL 2.0 or 3.0 implementation with coarse grained Shared Virtual Memory and SPIR-V input supported.
* For HIP-SYCL and HIP-MKL Interoperability
  * [oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)

## Compiling Clang

It's recommended to use the latest version of LLVM and use chipStar fork of SPIRV-LLVM-Translator.
```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout -t origin/release/16.x
cd llvm/projects
git clone https://github.com/CHIP-SPV/SPIRV-LLVM-Translator.git
cd SPIRV-LLVM-Translator
git checkout -t origin/chipspv-llvm-16-patches
cd ../../

mkdir build
cd build

# DLLVM_ENABLE_PROJECTS="clang;openmp" OpenMP is optional but many apps use it
# DLLVM_TARGETS_TO_BUILD Speed up compilation by building only the necessary CPU host target
# CMAKE_INSTALL_PREFIX Where to install LLVM
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;openmp" \
  -DLLVM_TARGETS_TO_BUILD=X86 \
  -DCMAKE_INSTALL_PREFIX=/home/pvelesko/install/llvm/16.0 

```

```bash
## Downloading Sources

```bash
git clone https://github.com/CHIP-SPV/chipStar.git
cd chipStar
git submodule update --init --recursive
```

## Building

```bash
mkdir build
cd build
cmake .. \ 
    -DLLVM_CONFIG_BIN=/path/to/llvm-config # optional, if not in PATH or if only versioned binary is available i.e. llvm-config-16
    -DCMAKE_INSTALL_PREFIX=/path/to/install # optional, default is <build_dir>/install
make
make install
```

Useful options:
 * `-DCMAKE_BUILD_TYPE=<Debug(default), Release, RelWithDebInfo>`
 * `-DBUILD_SAMPLES=<ON(default), OFF>` # Samples are built by default, unless you set this to OFF

The documentation will be placed in `doxygen/html`.

## Building

```bash
cd build
make
make build_tests # builds the HIP's Catch2 testsuite
```

## Running Unit Tests

```bash
export BACKEND={opencl/level0/pocl}   # which backend you wish to test, "pocl" = PoCL OpenCL runtime, "opencl" = any other OpenCL runtime
export DEVICE={cpu,igpu,dgpu}         # what kind of device to test
export PARALLEL={N}                   # how many tests to run in parallel
export CHIP_PLATFORM=N                # required only when there are multiple OpenCL platforms present on the system

python3 $SOURCE_DIR/scripts/check.py $BUILD_DIR $DEVICE $BACKEND $PARALLEL 1
```

## Building documentation

Make sure you have doxygen installed. Then:

```bash
....
cd build
cmake .....
make gendocs
```

The documentation will be placed in `doxygen/html`.

## Troubleshooting

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
