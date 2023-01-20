# CHIP-SPV

CHIP-SPV that aims to make HIP and CUDA portable to platforms which support
SPIR-V as the device intermediate representation. Currently CHIP-SPV supports
OpenCL and Level Zero as the low-level runtime alternatives.

This project is an integration of [HIPCL](https://github.com/cpc/hipcl) and
[HIPLZ](https://github.com/jz10/anl-gt-gpu/) projects.

For CHIP-SPV User documentation, read [this.](docs/Using.md)
For CHIP-SPV Developer documentation, read [this.](docs/Development.md)
For a list of (un)supported features in CHIP-SPV, read [this.](docs/Features.md)

## Prerequisites

* Cmake >= 3.16.0
* Clang 14 or 15
  * Can be installed, for example, by adding the [LLVM's Debian/Ubuntu repository](https://apt.llvm.org/) and installing packages 'clang-15 llvm-15 clang-tools-15'. *NOTE*: The Ubuntu clang package does not provide a symlink for `clang++`, only `clang++-14` is availble. If you plan on using `hipcc` you will need to make this symlink manually to ensure that `clang++` is available in `HIP_CLANG_PATH`.
* SPIRV-LLVM-Translator from a branch matching to the clang version:
  (e.g. llvm\_release\_150 for Clang 15.0)
  [llvm-spirv](https://github.com/KhronosGroup/SPIRV-LLVM-Translator)
* For Level Zero Backend
  * [Intel Compute Runtime](https://github.com/intel/compute-runtime) or [oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
  * [oneAPI Level Zero Loader](https://github.com/oneapi-src/level-zero/releases)
* For OpenCL Backend
  * OpenCL 2.0 or 3.0 implementation with coarse grained Shared Virtual Memory and SPIR-V input supported.
* For HIP-SYCL and HIP-MKL Interoperability
  * [oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)

## Downloading Sources

```bash
git clone https://github.com/CHIP-SPV/chip-spv.git
cd chip-spv
git submodule update --init --recursive
```

## Building

```bash
mkdir build
cd build
cmake .. \ 
    -DLLVM_CONFIG=/path/to/llvm-config # optional, if not in PATH or if only versioned binary is available i.e. llvm-config-15
    -DCMAKE_INSTALL_PREFIX=/path/to/install # optional, default is <build_dir>/install
make
make install
```

Useful options:
 * `-DCMAKE_BUILD_TYPE=<Debug(default), Release, RelWithDebInfo>`

The documentation will be placed in `doxygen/html`.

## Building & Running Unit Tests

```bash
make build_tests
make check # runs only tests that are expected to work
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
