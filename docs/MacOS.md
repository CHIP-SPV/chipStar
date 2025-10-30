# Building chipStar on macOS

This guide covers setting up a complete development environment for chipStar on macOS from a fresh installation.

## Prerequisites

macOS with Xcode Command Line Tools installed. The Command Line Tools can be installed by running:

```bash
xcode-select --install
```

## Quick Setup: Automated Installation Script

Copy and execute the following script to install all dependencies to `$HOME/install`:

```bash
#!/bin/bash
set -e

INSTALL_DIR="$HOME/install"
LLVM_VERSION=19
LLVM_DIR="$INSTALL_DIR/llvm-$LLVM_VERSION"

# Detect Homebrew prefix
if [ -d "/opt/homebrew" ]; then
    BREW_PREFIX="/opt/homebrew"
else
    BREW_PREFIX="/usr/local"
fi

echo "Installing dependencies to $INSTALL_DIR..."

# Install Homebrew if not present
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    eval "$($BREW_PREFIX/bin/brew shellenv)"
fi

# Install build tools via Homebrew
echo "Installing build tools..."
brew install cmake ninja git coreutils spirv-tools spirv-headers libtool pkg-config hwloc

# Create install directory
mkdir -p "$INSTALL_DIR"

# Build LLVM from source using chipStar script
echo "Building LLVM $LLVM_VERSION..."
if [ ! -f "./scripts/configure_llvm.sh" ]; then
    echo "Error: Run this script from the chipStar repository root"
    exit 1
fi

./scripts/configure_llvm.sh --version $LLVM_VERSION --install-dir "$LLVM_DIR"
cd llvm-project/llvm/build_$LLVM_VERSION
make -j$(nproc)
make install
cd - > /dev/null

# Note: SPIRV-LLVM-Translator is automatically built as part of the LLVM build
# since it's cloned into llvm-project/llvm/projects/SPIRV-LLVM-Translator
# by the configure script. llvm-spirv will be installed in $LLVM_DIR/bin

# Build OpenCL Headers
echo "Building OpenCL Headers..."
git clone https://github.com/KhronosGroup/OpenCL-Headers.git || (cd OpenCL-Headers && git pull)
cd OpenCL-Headers
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"
make -j$(nproc)
make install
cd "$INSTALL_DIR"

# Build OpenCL ICD Loader
echo "Building OpenCL ICD Loader..."
git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader.git || (cd OpenCL-ICD-Loader && git pull)
cd OpenCL-ICD-Loader
mkdir -p build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  -DOPENCL_HEADERS_DIR="$INSTALL_DIR/include"
make -j$(nproc)
make install
cd "$INSTALL_DIR"

# Build clinfo
echo "Building clinfo..."
git clone https://github.com/Oblomov/clinfo.git || (cd clinfo && git pull)
cd clinfo
make PREFIX="$INSTALL_DIR" install
cd "$INSTALL_DIR"

# Build PoCL with SPIR-V support
echo "Building PoCL with SPIR-V support..."
git clone https://github.com/pocl/pocl.git || (cd pocl && git pull)
cd pocl
mkdir -p build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  -DENABLE_SPIR=ON \
  -DLLVM_DIR="$LLVM_DIR/lib/cmake/llvm" \
  -DLLVM_CONFIG="$LLVM_DIR/bin/llvm-config"
make -j$(nproc)
make install
cd "$INSTALL_DIR"

# Create ICD vendors directory and PoCL ICD file
echo "Setting up OpenCL ICD configuration..."
sudo mkdir -p /etc/OpenCL/vendors
if [ -f "$INSTALL_DIR/lib/pocl/libpocl.dylib" ]; then
    echo "$INSTALL_DIR/lib/pocl/libpocl.dylib" | sudo tee /etc/OpenCL/vendors/pocl.icd
elif [ -f "$INSTALL_DIR/lib/libpocl.dylib" ]; then
    echo "$INSTALL_DIR/lib/libpocl.dylib" | sudo tee /etc/OpenCL/vendors/pocl.icd
fi

# Add to PATH in shell config
echo ""
echo "Adding to PATH..."
if ! grep -q "$INSTALL_DIR/llvm-$LLVM_VERSION/bin" ~/.zshrc 2>/dev/null; then
    echo 'export PATH="$HOME/install/llvm-'$LLVM_VERSION'/bin:$PATH"' >> ~/.zshrc
    echo 'export LLVM_DIR="$HOME/install/llvm-'$LLVM_VERSION'/lib/cmake/llvm"' >> ~/.zshrc
    echo 'export PATH="$HOME/install/bin:$PATH"' >> ~/.zshrc
    echo 'export LD_LIBRARY_PATH="$HOME/install/lib:$LD_LIBRARY_PATH"' >> ~/.zshrc
    echo 'export DYLD_LIBRARY_PATH="$HOME/install/lib:$DYLD_LIBRARY_PATH"' >> ~/.zshrc
fi

echo ""
echo "Installation complete!"
echo "Please run: source ~/.zshrc"
echo "Or restart your terminal, then verify with:"
echo "  clang-$LLVM_VERSION --version"
echo "  llvm-spirv --version"
echo "  clinfo -l"
```

Save this script, make it executable, and run it from the chipStar repository root:

```bash
chmod +x setup-macos.sh
./setup-macos.sh
```

After running the script, add to your shell config (or restart terminal):

```bash
source ~/.zshrc
```

Then verify the installation:

```bash
clang-19 --version
llvm-spirv --version
clinfo -l
```

**Critical Note**: macOS's built-in OpenCL runtime does **not** support SPIR-V and will **not work** with chipStar. Ensure PoCL appears in `clinfo -l` output, not macOS's built-in OpenCL. Always set `CHIP_DEVICE_TYPE=pocl` when running chipStar on macOS.

---

## Detailed Installation Steps

The following sections provide detailed explanations for each component. Use the automated script above for a complete setup, or follow these steps if you prefer manual installation.

## Step 1: Install Homebrew

Homebrew is a package manager for macOS that simplifies installing dependencies.

If Homebrew is not already installed:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the on-screen instructions. After installation, you may need to add Homebrew to your PATH:

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
eval "$(/opt/homebrew/bin/brew shellenv)"
```

For Intel Macs (if using the older `/usr/local` prefix):

```bash
echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zshrc
eval "$(/usr/local/bin/brew shellenv)"
```

## Step 2: Install Basic Build Tools

Install CMake and other essential build tools:

```bash
brew install cmake ninja git coreutils
```

Note: `coreutils` provides the `nproc` command which will be used throughout this guide to determine the number of CPU cores for parallel builds.

## Step 2.5: Install Environment Modules (Optional)

Environment Modules provides a convenient way to manage environment variables for different software versions. If you prefer using `module load llvm/19.0` instead of manually setting PATH variables, install it:

```bash
brew install modules
```

Add the initialization to your `~/.zshrc`:

```bash
echo 'source /opt/homebrew/opt/modules/init/zsh' >> ~/.zshrc
source ~/.zshrc
```

If you have modulefiles in a custom location (e.g., `$HOME/modulefiles`), add it to MODULEPATH:

```bash
echo 'export MODULEPATH=$HOME/modulefiles:$MODULEPATH' >> ~/.zshrc
# Or use module use:
# module use $HOME/modulefiles
```

After installation, verify modules is working:

```bash
module avail
```

## Step 3: Install LLVM and Clang from Source

chipStar requires a specific version of LLVM/Clang (18, 19, or 20) with patches. The recommended approach is to use the chipStar fork of LLVM which includes necessary fixes.

### Using the chipStar configure script:

```bash
./scripts/configure_llvm.sh --version 19 --install-dir $HOME/install/llvm-19
cd llvm-project/llvm/build_19
make -j$(nproc)
make install
```

Add LLVM to your PATH:

```bash
echo 'export PATH="$HOME/install/llvm-19/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Verify installation:

```bash
clang-19 --version
llvm-config-19 --version
```

### Manual installation (alternative):

```bash
git clone --depth 1 https://github.com/CHIP-SPV/llvm-project.git -b chipStar-llvm-19
cd llvm-project/llvm/projects
git clone --depth 1 https://github.com/CHIP-SPV/SPIRV-LLVM-Translator.git -b chipStar-llvm-19
cd ../..

cmake -S llvm -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;openmp" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_INSTALL_PREFIX=$HOME/install/llvm-19
make -C build -j$(nproc) all install
```

## Step 4: Install Additional Dependencies

Install SPIRV-Tools, SPIRV-Headers, and PoCL dependencies via Homebrew:

```bash
brew install spirv-tools spirv-headers libtool pkg-config hwloc
```

Verify SPIRV-Tools installation:

```bash
spirv-as --version
```

Note: The `spirv-extractor` tool in chipStar expects SPIRV-Tools headers at `/opt/homebrew/include/spirv-tools/` (Apple Silicon) or `/usr/local/include/spirv-tools/` (Intel).

**Note on SPIRV-LLVM-Translator**: The `llvm-spirv` tool is automatically built as part of the LLVM build process since the chipStar configure script clones it into `llvm-project/llvm/projects/SPIRV-LLVM-Translator`. It will be installed in `$HOME/install/llvm-19/bin/`.

## Step 5: Build OpenCL Components

### OpenCL Headers

```bash
git clone https://github.com/KhronosGroup/OpenCL-Headers.git
cd OpenCL-Headers
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/install
make -j$(nproc) && make install
cd ../../..
```

### OpenCL ICD Loader

```bash
git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
cd OpenCL-ICD-Loader
mkdir build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$HOME/install \
  -DOPENCL_HEADERS_DIR=$HOME/install/include
make -j$(nproc) && make install
cd ../../..

# Create ICD vendors directory
sudo mkdir -p /etc/OpenCL/vendors
```

### clinfo

```bash
git clone https://github.com/Oblomov/clinfo.git
cd clinfo
make PREFIX=$HOME/install install
cd ..
```

## Step 6: Build PoCL with SPIR-V Support

PoCL (Portable Computing Language) provides an OpenCL implementation with SPIR-V support required for chipStar:

```bash
git clone https://github.com/pocl/pocl.git
cd pocl
mkdir build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$HOME/install \
  -DENABLE_SPIR=ON \
  -DLLVM_DIR=$HOME/install/llvm-19/lib/cmake/llvm \
  -DLLVM_CONFIG=$HOME/install/llvm-19/bin/llvm-config
make -j$(nproc) && make install
cd ../../..

# Configure PoCL ICD
sudo mkdir -p /etc/OpenCL/vendors
if [ -f "$HOME/install/lib/pocl/libpocl.dylib" ]; then
    echo "$HOME/install/lib/pocl/libpocl.dylib" | sudo tee /etc/OpenCL/vendors/pocl.icd
elif [ -f "$HOME/install/lib/libpocl.dylib" ]; then
    echo "$HOME/install/lib/libpocl.dylib" | sudo tee /etc/OpenCL/vendors/pocl.icd
fi
```

## Step 7: Verify Installation

Verify all components are properly installed:

```bash
# Check tools
clang-19 --version
llvm-spirv --version
spirv-as --version

# Check OpenCL platforms - should show PoCL, NOT Apple's OpenCL
clinfo -l
```

**Critical**: Ensure PoCL appears in `clinfo -l`, not Apple's OpenCL. If Apple's OpenCL is listed:

```bash
export OCL_ICD_VENDORS=/etc/OpenCL/vendors/pocl.icd
clinfo -l
```

## Step 8: Build chipStar

```bash
mkdir build && cd build
cmake .. \
  -DLLVM_CONFIG_BIN=$HOME/install/llvm-19/bin/llvm-config \
  -DCMAKE_INSTALL_PREFIX=$HOME/install/chipStar \
  -GNinja
ninja -j$(nproc) all build_tests install
```

**Verify chipStar detects PoCL**:

```bash
CHIP_LOGLEVEL=info CHIP_DEVICE_TYPE=pocl ./bin/hipInfo
```

Output should show PoCL as the OpenCL platform. If you see Apple's OpenCL or errors, check the Troubleshooting section.

## Step 9: Run Unit Tests

```bash
cd /path/to/chipStar  # Repository root
export CHIP_DEVICE_TYPE=pocl
python3 scripts/check.py build pocl opencl
```

## Quick Reference

### Key Points
- **macOS's built-in OpenCL does NOT work with chipStar** - it doesn't support SPIR-V
- Always set `CHIP_DEVICE_TYPE=pocl` when running chipStar on macOS
- All dependencies install to `$HOME/install`

### Installation Locations

All dependencies are installed to `$HOME/install`:
- **LLVM/Clang**: `$HOME/install/llvm-19/`
- **SPIRV-Tools**: `/opt/homebrew/` or `/usr/local/` (via Homebrew)
- **SPIRV-LLVM-Translator**: `$HOME/install/llvm-19/`
- **OpenCL Headers**: `$HOME/install/include/CL/`
- **OpenCL ICD Loader**: `$HOME/install/lib/libOpenCL.dylib`
- **PoCL**: `$HOME/install/lib/pocl/` or `$HOME/install/lib/`
- **clinfo**: `$HOME/install/bin/clinfo`

### Important Paths

- **Install directory**: `$HOME/install`
- **Homebrew prefix (Apple Silicon)**: `/opt/homebrew`
- **Homebrew prefix (Intel)**: `/usr/local`
- **ICD vendors directory**: `/etc/OpenCL/vendors/`

### Environment Variables

Add these to your `~/.zshrc`:

```bash
export PATH="$HOME/install/llvm-19/bin:$PATH"
export PATH="$HOME/install/bin:$PATH"
export LLVM_DIR="$HOME/install/llvm-19/lib/cmake/llvm"
export LD_LIBRARY_PATH="$HOME/install/lib:$LD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH="$HOME/install/lib:$DYLD_LIBRARY_PATH"

# Required for chipStar on macOS - forces use of PoCL instead of Apple's OpenCL
export CHIP_DEVICE_TYPE=pocl
```

## Troubleshooting

### SPIRV-Tools header not found

If you get errors about `spirv-tools/libspirv.h` not found, verify:

```bash
ls /opt/homebrew/include/spirv-tools/libspirv.h
```

If the file doesn't exist, ensure SPIRV-Tools is properly installed:

```bash
brew reinstall spirv-tools
```

### macOS OpenCL Not Working

macOS's built-in OpenCL **does not support SPIR-V** and will **not work** with chipStar. Ensure you're using PoCL:

```bash
export OCL_ICD_VENDORS=/etc/OpenCL/vendors/pocl.icd
export CHIP_DEVICE_TYPE=pocl
CHIP_LOGLEVEL=info ./build/bin/hipInfo  # Should show PoCL
```

### LLVM not found

Ensure LLVM is in your PATH:

```bash
which clang-19
which llvm-config-19
```

If not found, add to your `~/.zshrc`:

```bash
export PATH="$HOME/install/llvm-19/bin:$PATH"
source ~/.zshrc
```

### PoCL build errors

If PoCL fails to build, ensure:
- LLVM is properly installed and accessible
- All dependencies are installed (`libtool`, `pkg-config`, `hwloc`)
- CMake can find LLVM (`-DLLVM_DIR` and `-DLLVM_CONFIG` are set correctly)

---

## Changes in Upstream Projects

### What we changed in LLVM

chipStar uses a forked version of LLVM from the [CHIP-SPV/llvm-project](https://github.com/CHIP-SPV/llvm-project) repository with branches named `chipStar-llvm-19`, `chipStar-llvm-18`, etc. The following macOS-specific commits have been added to the `chipStar-llvm-19` branch:

- **e2da01b8edc4**: `fix(clang): Make spirv64 target compatible with Apple SDK TargetConditionals.h`
  - Hides `__is_target_arch` builtin for spirv targets to prevent Apple SDK TargetConditionals.h from using it
  - Handles spirv architectures gracefully in `isTargetArch()` to prevent assertion errors
  - Fixes compilation error: `'error: unrecognized arch using compiler with __is_target_arch support'`

- **39cf081a623c**: `Update HIP fatbin section names for macOS`
  - Changed fatbin section names in CGCUDANV.cpp, HIPUtility.cpp, and OffloadWrapper.cpp to include macOS-specific prefixes
  - Ensures proper handling of fatbin symbols when targeting macOS systems

- **1093aa91f404**: `Darwin toolchain: handle uninitialized targets`
  - Fixes assertion failure when functions like `isTargetWatchOSBased()` are called before target initialization during offload setup
  - Prevents crash: `Assertion failed: (TargetInitialized && "Target not initialized!")`

### What we changed in POCL

chipStar uses a forked version of PoCL from the [CHIP-SPV/pocl](https://github.com/CHIP-SPV/pocl) repository on the `macos-chipstar` branch. The following macOS-specific commits have been added:

- **c43cb952f**: `Fix SLEEF vectorization issue on macOS by disabling it`
  - Disables `ENABLE_HOST_CPU_VECTORIZE_SLEEF` on macOS ARM64 to avoid undefined symbol linker errors
  - LLVM's LoopVectorizer generates calls to LLVM Vector Function ABI names that are not provided by SLEEF on macOS
  - Math functions use scalar implementations instead, which work correctly on macOS

- **d7cc0cbb1**: `Enable FP16 support for macOS`
  - Changed CMAKE_HOST_SYSTEM_NAME check to include Darwin alongside Linux
  - Enables half-precision floating point operations on macOS when LLVM 19+ and other requirements are met

- **06717fd6b**: `fix(macos): Add SDK sysroot to kernel linker flags`
  - Adds `-isysroot` flag to DEFAULT_HOST_LD_FLAGS for macOS, pointing to the SDK path obtained via `xcrun --show-sdk-path`
  - Fixes JIT compilation errors on macOS 15 (Sequoia) and later where libSystem.tbd cannot be found without sysroot specification
  - Resolves: `ld: library 'System' not found` errors

