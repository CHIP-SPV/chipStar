#!/bin/bash

# if an error is enountered, exit
set -e

# Retry wrapper for network operations (handles transient DNS failures)
retry() {
  local max_attempts=5
  local delay=10
  local attempt=1
  while [ $attempt -le $max_attempts ]; do
    if "$@"; then
      return 0
    fi
    echo "Attempt $attempt/$max_attempts failed. Retrying in ${delay}s..."
    sleep $delay
    attempt=$((attempt + 1))
    delay=$((delay * 2))
  done
  echo "All $max_attempts attempts failed."
  return 1
}
# default values for optional arguments
LINK_TYPE="dynamic"
EMIT_ONLY="off"
CONFIGURE_ONLY="off"
WITH_BINUTILS=""
VARIANT="translator"

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
    --variant)
      VARIANT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# check mandatory argument version
if [ -z "$VERSION" ]; then
  echo "Usage: $0 --version <version> --install-dir <dir> --link-type static/dynamic(default) [--variant translator|native] [--with-binutils [path]] [--configure-only] [-N]"
  echo "--version: LLVM version 21, 22, 23, or latest"
  echo "           21/22: upstream release branch plus patches from llvm-patches/llvm-<version>/"
  echo "           23: upstream tag llvmorg-23.1.0-rc2 plus patches from llvm-patches/llvm-23/"
  echo "           latest (experimental): CHIP-SPV/llvm-project branch chipStar-llvm-23, maintained"
  echo "           directly with no patches (patches exist only for the release-pinned versions)"
  echo "--install-dir: installation directory"
  echo "--link-type: static or dynamic (default: dynamic)"
  echo "--variant: translator (host only) or native (host;SPIRV) (default: translator)"
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
if [ "$VERSION" != "21" ] && [ "$VERSION" != "22" ] && [ "$VERSION" != "23" ] \
       && [ "$VERSION" != "latest" ]; then
  echo "Invalid version '$VERSION'. Must be 21, 22, 23, or latest."
  echo "(Support for LLVM 17 through 20 has been dropped.)"
  exit 1
fi

# validate LINK_TYPE argument
if [ "$LINK_TYPE" != "static" ] && [ "$LINK_TYPE" != "dynamic" ]; then
  echo "Invalid LINK_TYPE. Must be 'static' or 'dynamic'."
  exit 1
fi

# validate VARIANT argument
if [ "$VARIANT" != "translator" ] && [ "$VARIANT" != "native" ]; then
  echo "Invalid VARIANT. Must be 'translator' or 'native'."
  exit 1
fi

# Platform-specific compiler selection
if [[ "$(uname)" == "Darwin" ]]; then
  CC=clang
  CXX=clang++
  gcc_base_path=""
else
  CC=gcc
  CXX=g++
  # get the gcc base path to use in cmake flags
  gcc_base_path=$( which gcc | sed s+'bin/gcc'++ )
fi

NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu)

# Map version to LLVM repo/branch and SPIRV-Translator branch.
# latest tracks the maintained chipStar branch and takes no patches.
if [ "$VERSION" == "latest" ]; then
  LLVM_REPO="https://github.com/CHIP-SPV/llvm-project.git"
  LLVM_BRANCH="chipStar-llvm-23"
  TRANSLATOR_BRANCH="llvm_release_230"
elif [ "$VERSION" == "23" ]; then
  # Pinned to a release candidate tag rather than release/23.x: the branch
  # moves, and the llvm-patches/llvm-23 series (in particular the llvm#213052
  # backport) is verified against this exact tree.
  LLVM_REPO="https://github.com/llvm/llvm-project.git"
  LLVM_BRANCH="llvmorg-23.1.0-rc2"
  TRANSLATOR_BRANCH="llvm_release_230"
else
  LLVM_REPO="https://github.com/llvm/llvm-project.git"
  LLVM_BRANCH="release/${VERSION}.x"
  TRANSLATOR_BRANCH="llvm_release_${VERSION}0"
fi

# LLVM 23 removed the LLVM_ENABLE_PROJECTS=openmp build mode; openmp has to be
# built as a runtime there. Earlier versions still expect it as a project, and
# CI hard-fails when omp.h is missing from the install (TestHipccFopenmp), so
# this has to stay version conditional.
if [ "$VERSION" == "23" ] || [ "$VERSION" == "latest" ]; then
  LLVM_PROJECTS="clang;clang-tools-extra"
  LLVM_RUNTIMES="openmp"
else
  LLVM_PROJECTS="clang;openmp;clang-tools-extra"
  LLVM_RUNTIMES=""
fi

export LLVM_DIR=`pwd`/llvm-project/llvm

# If we're only emitting the cmake command, skip the git operations
if [ "$EMIT_ONLY" != "on" ]; then
  # check if llvm-project exists, if not clone it
  if [ ! -d llvm-project ]; then
    echo "Cloning LLVM (${LLVM_BRANCH})..."
    if [ "$VERSION" == "latest" ]; then
      retry git clone --single-branch -b ${LLVM_BRANCH} ${LLVM_REPO}
      cd llvm-project
    else
      retry git clone ${LLVM_REPO}
      cd llvm-project
      git checkout ${LLVM_BRANCH}
    fi

    echo "Cloning SPIRV-LLVM-Translator (${TRANSLATOR_BRANCH})..."
    cd llvm/projects
    retry git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
    cd SPIRV-LLVM-Translator
    git checkout ${TRANSLATOR_BRANCH}
  else
    # Warn the user.
    echo "llvm-project directory already exists. Checking out ${LLVM_BRANCH}..."
    cd llvm-project
    git fetch origin
    git reset --hard
    git clean -fd
    git checkout ${LLVM_BRANCH}

    if [ ! -d llvm/projects/SPIRV-LLVM-Translator ]; then
      echo "Cloning SPIRV-LLVM-Translator (${TRANSLATOR_BRANCH})..."
      cd llvm/projects
      retry git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
      cd SPIRV-LLVM-Translator
    else
      cd llvm/projects/SPIRV-LLVM-Translator
      git fetch origin
      git reset --hard
      git clean -fd
    fi
    git checkout ${TRANSLATOR_BRANCH}
  fi

  if [ "$VERSION" == "latest" ]; then
    echo "Skipping patches: branch ${LLVM_BRANCH} is maintained directly."
  else
    # Apply chipStar-specific patches for this version, in lexicographic
    # (numeric) order. Every patch in the version directory must apply.
    PATCH_DIR="${THIS_SCRIPT_DIR}/../llvm-patches/llvm-${VERSION}"
    if [ ! -d "$PATCH_DIR" ]; then
      echo "Error: patch directory not found: $PATCH_DIR"
      echo "Is this script being run from a complete chipStar checkout?"
      exit 1
    fi

    cd ${initial_pwd}/llvm-project
    echo "Applying LLVM patches from ${PATCH_DIR}/llvm..."
    for patch in "$PATCH_DIR"/llvm/*.patch; do
      [ -f "$patch" ] || continue
      echo "  Applying $(basename "$patch")..."
      git apply "$patch" || {
        echo "Error: Failed to apply $(basename "$patch")"
        exit 1
      }
    done

    cd ${initial_pwd}/llvm-project/llvm/projects/SPIRV-LLVM-Translator
    echo "Applying SPIRV-Translator patches from ${PATCH_DIR}/spirv-translator..."
    for patch in "$PATCH_DIR"/spirv-translator/*.patch; do
      [ -f "$patch" ] || continue
      echo "  Applying $(basename "$patch")..."
      git apply "$patch" || {
        echo "Error: Failed to apply $(basename "$patch")"
        exit 1
      }
    done

    echo "All patches applied successfully"
  fi

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
        make -j${NPROC}
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
# translator: SPIRV-LLVM-Translator only (host targets). native: also enable LLVM's experimental SPIR-V backend.
if [ "$VARIANT" = "native" ]; then
  LLVM_TARGETS="host;SPIRV"
else
  LLVM_TARGETS="host"
fi

COMMON_CMAKE_OPTIONS=(
  "-DCMAKE_CXX_COMPILER=${CXX}"
  "-DCMAKE_C_COMPILER=${CC}"
  "-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}"
  "-DCMAKE_BUILD_TYPE=Release"
  "-DLLVM_ENABLE_PROJECTS=\"${LLVM_PROJECTS}\""
  "-DLLVM_TARGETS_TO_BUILD=\"${LLVM_TARGETS}\""
  "-DLLVM_ENABLE_ASSERTIONS=On"
  "-DLLVM_INCLUDE_TESTS=OFF"
  "-DLLVM_INCLUDE_EXAMPLES=OFF"
  "-DLLVM_INCLUDE_BENCHMARKS=OFF"
  "-DLLVM_INCLUDE_DOCS=OFF"
)

if [ -n "${LLVM_RUNTIMES}" ]; then
  COMMON_CMAKE_OPTIONS+=("-DLLVM_ENABLE_RUNTIMES=\"${LLVM_RUNTIMES}\"")
  if [[ "$(uname)" == "Darwin" ]]; then
    # Runtimes (libomp) are compiled with the just-built clang, which ships no
    # compiler-rt builtins on macOS; libomp's complex division needs __divdc3
    # and friends, so link the runtimes against Apple's builtins archive.
    apple_builtins="$(clang -print-resource-dir)/lib/darwin/libclang_rt.osx.a"
    COMMON_CMAKE_OPTIONS+=(
      "-DRUNTIMES_CMAKE_ARGS=\"-DCMAKE_EXE_LINKER_FLAGS=${apple_builtins};-DCMAKE_SHARED_LINKER_FLAGS=${apple_builtins}\""
    )
  fi
fi

# Linux-specific flags
if [[ "$(uname)" != "Darwin" ]]; then
  COMMON_CMAKE_OPTIONS+=(
    "-DCMAKE_CXX_LINK_FLAGS=\"-Wl,-rpath,${gcc_base_path}/lib64 -L${gcc_base_path}/lib64\""
    "-DCLANG_DEFAULT_PIE_ON_LINUX=off"
  )
fi

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
    cmake --build . -j${NPROC}
    echo "Installing LLVM to ${INSTALL_DIR}..."
    cmake --build . --target install
    echo "LLVM ${VERSION} installed to ${INSTALL_DIR}"
  fi
fi
