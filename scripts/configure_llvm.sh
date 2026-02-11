#!/bin/bash

# if an error is enountered, exit
set -e

# default values for optional arguments
LINK_TYPE="static"
EMIT_ONLY="off"
CONFIGURE_ONLY="off"
WITH_BINUTILS=""

THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
initial_pwd=$(pwd)

# parse named arguments
while [ $# -gt 0 ]; do
  case "$1" in
    --version)
      VERSION="$2"
      shift 2
      ;;
    --install-dir)
      INSTALL_DIR="$2"
      shift 2
      ;;
    --link-type)
      LINK_TYPE="$2"
      shift 2
      ;;
    --with-binutils)
      if [[ -z "$2" ]] || [[ "$2" == --* ]] || [[ "$2" == -* ]]; then
        # No path provided, just enable binutils
        WITH_BINUTILS="on"
        shift 1
      else
        # Path provided
        WITH_BINUTILS="$2"
        shift 2
      fi
      ;;
    -N)
      EMIT_ONLY="on"
      shift 1
      ;;
    --configure-only)
      CONFIGURE_ONLY="on"
      shift 1
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# check mandatory argument version
if [ -z "$VERSION" ]; then
  echo "Usage: $0 --version <version> --install-dir <dir> --link-type static(default)/dynamic [--with-binutils [path]] [--configure-only] [-N]"
  echo "--version: LLVM version 17, 18, 19, 20 or 21"
  echo "--install-dir: installation directory"
  echo "--link-type: static or dynamic (default: static)"
  echo "--with-binutils [path]: enable binutils support with optional path to header directory (default: disabled)"
  echo "--configure-only: only clone, patch, and run cmake configure (skip build and install)"
  echo "-N: only emit the cmake configure command without executing it"
  echo "By default, the script will clone, patch, configure, build, and install LLVM."
  exit 1
fi
# Check if install-dir argument is provided
if [ -z "$INSTALL_DIR" ]; then
  echo "Error: --install-dir argument is required."
  exit 1
fi

# validate version argument
if [ "$VERSION" != "17" ] && [ "$VERSION" != "18" ] && [ "$VERSION" != "19" ] \
       && [ "$VERSION" != "20" ] && [ "$VERSION" != "21" ]; then
  echo "Invalid version. Must be 17, 18, 19, 20 or 21."
  exit 1
fi

# validate LINK_TYPE argument
if [ "$LINK_TYPE" != "static" ] && [ "$LINK_TYPE" != "dynamic" ]; then
  echo "Invalid LINK_TYPE. Must be 'static' or 'dynamic'."
  exit 1
fi

# get the gcc base path to use in cmake flags
gcc_base_path=$( which gcc | sed s+'bin/gcc'++ )

# Map version to upstream LLVM branch and SPIRV-Translator branch
LLVM_BRANCH="release/${VERSION}.x"
TRANSLATOR_BRANCH="llvm_release_${VERSION}0"

export LLVM_DIR=`pwd`/llvm-project/llvm

# If we're only emitting the cmake command, skip the git operations
if [ "$EMIT_ONLY" != "on" ]; then
  # check if llvm-project exists, if not clone it
  if [ ! -d llvm-project ]; then
    echo "Cloning LLVM from upstream..."
    git clone https://github.com/llvm/llvm-project.git
    cd llvm-project
    git checkout ${LLVM_BRANCH}
    
    echo "Cloning SPIRV-LLVM-Translator from upstream..."
    cd llvm/projects
    git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
    cd SPIRV-LLVM-Translator
    git checkout ${TRANSLATOR_BRANCH}
  else
    # Warn the user.
    echo "llvm-project directory already exists. Checking out upstream branches..."
    cd llvm-project
    git fetch origin
    git reset --hard
    git clean -fd
    git checkout ${LLVM_BRANCH}
    
    if [ ! -d llvm/projects/SPIRV-LLVM-Translator ]; then
      echo "Cloning SPIRV-LLVM-Translator from upstream..."
      cd llvm/projects
      git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
      cd SPIRV-LLVM-Translator
    else
      cd llvm/projects/SPIRV-LLVM-Translator
      git fetch origin
      git reset --hard
      git clean -fd
    fi
    git checkout ${TRANSLATOR_BRANCH}
  fi
  cd ${initial_pwd}/llvm-project
  
  # Apply chipStar-specific patches
  echo "Applying chipStar patches..."
  
  # Apply LLVM patches
  LLVM_PATCH_DIR="${THIS_SCRIPT_DIR}/../llvm-patches/llvm"
  if [ -d "$LLVM_PATCH_DIR" ]; then
    echo "Applying LLVM patches..."
    for patch in "$LLVM_PATCH_DIR"/*.patch; do
      if [ -f "$patch" ]; then
        patch_name=$(basename $patch)
        
        # Skip data layout patch for LLVM 17-19 (only needed for 20+)
        if [[ "$patch_name" == *"data-layout"* ]] && [ "$VERSION" -lt 20 ]; then
          echo "  Skipping $patch_name (not needed for LLVM ${VERSION})"
          continue
        fi
        
        # Use LLVM 21-specific data layout patch for version 21, skip LLVM 20 version
        if [[ "$patch_name" == "0002-fix-SPIR-V-data-layout.patch" ]] && [ "$VERSION" -eq 21 ]; then
          echo "  Skipping $patch_name (using LLVM 21-specific patch instead)"
          continue
        fi
        if [[ "$patch_name" == "0002-fix-SPIR-V-data-layout-llvm21.patch" ]] && [ "$VERSION" -ne 21 ]; then
          echo "  Skipping $patch_name (only needed for LLVM 21)"
          continue
        fi
        
        echo "  Applying $patch_name..."
        git apply "$patch" || {
          echo "Error: Failed to apply $patch_name"
          exit 1
        }
      fi
    done
  else
    #if the patch directory doesn't exist, good chance somehow this script is being run outside of its intended project
    echo "Error: LLVM patch directory not found at $LLVM_PATCH_DIR"
    exit 1
  fi
  
  # Apply SPIRV-Translator patches
  cd ${initial_pwd}/llvm-project/llvm/projects/SPIRV-LLVM-Translator
  TRANSLATOR_PATCH_DIR="${THIS_SCRIPT_DIR}/../llvm-patches/spirv-translator"
  if [ -d "$TRANSLATOR_PATCH_DIR" ]; then
    echo "Applying SPIRV-Translator patches..."
    for patch in "$TRANSLATOR_PATCH_DIR"/*.patch; do
      if [ -f "$patch" ]; then
        patch_name=$(basename $patch)
        
        # Skip fp_fast_mode patch for LLVM 17/18/19 (test file structure changed, opaque pointers are default)
        if [[ "$patch_name" == *"fp_fast_mode"* ]] && ([ "$VERSION" -eq 17 ] || [ "$VERSION" -eq 18 ] || [ "$VERSION" -eq 19 ]); then
          echo "  Skipping $patch_name (not needed for LLVM ${VERSION})"
          continue
        fi
        
        # Use version-specific patch for LLVM 17/18, skip the original for LLVM 17/18
        if [[ "$patch_name" == "0002-Pretend-the-SPIR-ver-needed-by-shuffles-is-1.2.patch" ]] && ([ "$VERSION" -eq 17 ] || [ "$VERSION" -eq 18 ]); then
          echo "  Skipping $patch_name (using LLVM 17/18-specific patch instead)"
          continue
        fi
        if [[ "$patch_name" == "0002-Pretend-the-SPIR-ver-needed-by-shuffles-is-1.2-llvm17-18.patch" ]] && [ "$VERSION" -ne 17 ] && [ "$VERSION" -ne 18 ]; then
          echo "  Skipping $patch_name (only needed for LLVM 17/18)"
          continue
        fi
        
        echo "  Applying $patch_name..."
        git apply "$patch" || {
          echo "Error: Failed to apply $patch_name"
          exit 1
        }
      fi
    done
  else
    #if the patch directory doesn't exist, good chance somehow this script is being run outside of its intended project
    echo "Error: SPIRV-Translator patch directory not found at $TRANSLATOR_PATCH_DIR"
    exit 1
  fi
  
  echo "All patches applied successfully"
  
  cd ${LLVM_DIR}

  rm -rf build_$VERSION
  mkdir build_$VERSION
  cd build_$VERSION
fi

# Check if /usr/include/plugin-api.h exists
if [ -n "${WITH_BINUTILS}" ]; then
  if [ "${WITH_BINUTILS}" != "on" ]; then
    # A specific path was provided
    if [ ! -f "${WITH_BINUTILS}/plugin-api.h" ]; then
      echo "Error: plugin-api.h not found in the specified path (${WITH_BINUTILS})"
      exit 1
    else
      echo "plugin-api.h was found at ${WITH_BINUTILS}"
      BINUTILS_HEADER_DIR=${WITH_BINUTILS}
    fi
  elif [ -f /usr/include/plugin-api.h ]; then
    echo "plugin-api.h was found at /usr/include/plugin-api.h"
    BINUTILS_HEADER_DIR=/usr/include
  else
    echo "plugin-api.h was not found at /usr/include/plugin-api.h"
    
    # Check if binutils was installed in a previous attempt
    BINUTILS_INSTALL_DIR=${INSTALL_DIR}/binutils
    if [ -f "${BINUTILS_INSTALL_DIR}/include/plugin-api.h" ]; then
      echo "Found previously installed binutils at ${BINUTILS_INSTALL_DIR}"
      BINUTILS_HEADER_DIR=${BINUTILS_INSTALL_DIR}/include
    else
      if [ "$EMIT_ONLY" != "on" ]; then
        echo "Installing binutils-dev from source..."
        
        # Create the installation directory if it doesn't exist
        mkdir -p ${BINUTILS_INSTALL_DIR}
        
        # Download the binutils source
        BINUTILS_VERSION="2.36.1"
        wget https://ftp.gnu.org/gnu/binutils/binutils-${BINUTILS_VERSION}.tar.gz
        
        # Extract the source
        tar -xzf binutils-${BINUTILS_VERSION}.tar.gz
        cd binutils-${BINUTILS_VERSION}
        
        # Configure, compile, and install binutils
        ./configure --prefix=${BINUTILS_INSTALL_DIR}
        make -j$(nproc)
        make install
        
        # Clean up
        cd ..
        rm -rf binutils-${BINUTILS_VERSION} binutils-${BINUTILS_VERSION}.tar.gz
        
        echo "binutils-dev installed successfully in ${BINUTILS_INSTALL_DIR}"
      fi
      BINUTILS_HEADER_DIR=${BINUTILS_INSTALL_DIR}/include
    fi
  fi
else
  echo "Binutils support is disabled (use --with-binutils to enable)"
  BINUTILS_HEADER_DIR=""
fi

# Add build type condition
# Forcing the use of gcc and g++ to avoid issues with intel compilers
# NOTE: chipStar uses the external SPIRV-LLVM-Translator tool exclusively for SPIRV generation,
# so we don't need LLVM's experimental/native SPIRV target at all
LLVM_TARGETS="host"
if [ "$VERSION" -ge 20 ]; then
  LLVM_TARGETS="${LLVM_TARGETS};SPIRV"
fi

COMMON_CMAKE_OPTIONS=(
  "-DCMAKE_CXX_COMPILER=g++"
  "-DCMAKE_C_COMPILER=gcc"
  "-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}"
  "-DCMAKE_BUILD_TYPE=Release"
  "-DLLVM_ENABLE_PROJECTS=\"clang;openmp;clang-tools-extra\""
  "-DLLVM_TARGETS_TO_BUILD=\"${LLVM_TARGETS}\""
  "-DLLVM_ENABLE_ASSERTIONS=On"
  "-DCMAKE_CXX_LINK_FLAGS=\"-Wl,-rpath,${gcc_base_path}/lib64 -L${gcc_base_path}/lib64\""
  "-DCLANG_DEFAULT_PIE_ON_LINUX=off"
  "-DLLVM_INCLUDE_TESTS=OFF"
  "-DLLVM_INCLUDE_EXAMPLES=OFF"
  "-DLLVM_INCLUDE_BENCHMARKS=OFF"
  "-DLLVM_INCLUDE_DOCS=OFF"
)

if [ -n "${BINUTILS_HEADER_DIR}" ]; then
  COMMON_CMAKE_OPTIONS+=("-DLLVM_BINUTILS_INCDIR=${BINUTILS_HEADER_DIR}")
fi

if [ "$LINK_TYPE" == "static" ]; then
  CMAKE_COMMAND="cmake ../ ${COMMON_CMAKE_OPTIONS[@]}"
elif [ "$LINK_TYPE" == "dynamic" ]; then
  CMAKE_COMMAND="cmake ../ ${COMMON_CMAKE_OPTIONS[@]} \"-DCMAKE_INSTALL_RPATH=${INSTALL_DIR}/lib\" \"-DLLVM_LINK_LLVM_DYLIB=ON\" \"-DLLVM_BUILD_LLVM_DYLIB=ON\""
else
  echo "Invalid link_type. Must be 'static' or 'dynamic'."
  exit 1
fi

if [ "$EMIT_ONLY" == "on" ]; then
  echo "CMAKE COMMAND:"
  echo "$CMAKE_COMMAND"
else
  eval $CMAKE_COMMAND

  if [ "$CONFIGURE_ONLY" == "on" ]; then
    echo "Configure complete. Skipping build and install (--configure-only)."
  else
    echo "Building LLVM (this may take a while)..."
    cmake --build . -j$(nproc)
    echo "Installing LLVM to ${INSTALL_DIR}..."
    cmake --build . --target install
    echo "LLVM ${VERSION} installed to ${INSTALL_DIR}"
  fi
fi
