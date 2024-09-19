#!/bin/bash

# if an error is enountered, exit
set -e

# default values for optional arguments
LINK_TYPE="static"
ONLY_NECESSARY_SPIRV_EXTS="off"
BINUTILS_HEADER_LOCATION=""

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
    --only-necessary-spirv-exts)
      ONLY_NECESSARY_SPIRV_EXTS="$2"
      shift 2
      ;;
    --binutils-header-location)
      BINUTILS_HEADER_LOCATION="$2"
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
  echo "Usage: $0 --version <version> --install-dir <dir> --link-type static(default)/dynamic --only-necessary-spirv-exts <on|off> --binutils-header-location <path>"
  echo "--version: LLVM version 15, 16, 17, 18, 19"
  echo "--install-dir: installation directory"
  echo "--link-type: static or dynamic (default: static)"
  echo "--only-necessary-spirv-exts: on or off (default: off)"
  echo "--binutils-header-location: path to binutils header (default: empty)"
  exit 1
fi
# Check if install-dir argument is provided
if [ -z "$INSTALL_DIR" ]; then
  echo "Error: --install-dir argument is required."
  exit 1
fi

# validate version argument
if [ "$VERSION" != "15" ] && [ "$VERSION" != "16" ] && [ "$VERSION" != "17" ] \
       && [ "$VERSION" != "18" ] && [ "$VERSION" != "19" ]; then
  echo "Invalid version. Must be 15, 16, 17, 18 or 19."
  exit 1
fi

# validate LINK_TYPE argument
if [ "$LINK_TYPE" != "static" ] && [ "$LINK_TYPE" != "dynamic" ]; then
  echo "Invalid LINK_TYPE. Must be 'static' or 'dynamic'."
  exit 1
fi

# validate only-necessary-spirv-exts argument
if [ "$ONLY_NECESSARY_SPIRV_EXTS" != "on" ] && [ "$ONLY_NECESSARY_SPIRV_EXTS" != "off" ]; then
  echo "Invalid only_necessary_spirv_exts. Must be 'on' or 'off'."
  exit 1
fi

# get the gcc base path to use in cmake flags
gcc_base_path=$( which gcc | sed s+'bin/gcc'++ )

# set the brach name for checkuot based on only-necessary-spirv-exts
if [ "$4" == "on" ]; then
  LLVM_BRANCH="spirv-ext-fixes-${VERSION}"
  TRANSLATOR_BRANCH="llvm_release_${VERSION}0"
else
  LLVM_BRANCH="chipStar-llvm-${VERSION}"
  TRANSLATOR_BRANCH="llvm_release_${VERSION}0"
fi

export LLVM_DIR=`pwd`/llvm-project/llvm

# check if llvm-project exists, if not clone it
if [ ! -d llvm-project ]; then
  git clone https://github.com/CHIP-SPV/llvm-project.git -b ${LLVM_BRANCH} \
    --depth 1
  cd ${LLVM_DIR}/projects
  git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git \
    -b ${TRANSLATOR_BRANCH} --depth 1
  cd ${LLVM_DIR}
else
  # Warn the user.
  echo "llvm-project directory already exists. Assuming it's cloned from chipStar."
  cd ${LLVM_DIR}
  # check if already on the desired branch
  if [ `git branch --show-current` == ${LLVM_BRANCH} ]; then
    echo "Already on branch ${LLVM_BRANCH}"
  else
    echo "Switching to branch ${LLVM_BRANCH}"
    git fetch origin ${LLVM_BRANCH}:${LLVM_BRANCH}
    git checkout ${LLVM_BRANCH}
  fi
  cd ${LLVM_DIR}/projects/SPIRV-LLVM-Translator
  # check if already on the desired branch
  if [ `git branch --show-current` == ${TRANSLATOR_BRANCH} ]; then
    echo "Already on branch ${TRANSLATOR_BRANCH}"
  else
    echo "Switching to branch ${TRANSLATOR_BRANCH}"
    git fetch origin ${TRANSLATOR_BRANCH}:${TRANSLATOR_BRANCH}
    git checkout ${TRANSLATOR_BRANCH}
  fi
  cd ${LLVM_DIR}
fi

# check if the build directory exists
if [ -d build_$VERSION ]; then
  read -p "Build directory build_$VERSION already exists. Do you want to delete it and continue? (y/n) " answer
  case ${answer:0:1} in
    y|Y )
      echo "Deleting existing build directory..."
      rm -rf build_$VERSION
      mkdir build_$VERSION
      cd build_$VERSION
    ;;
    * )
      echo "Build directory not deleted. Exiting."
      exit 1
    ;;
  esac
else
  mkdir build_$VERSION
  cd build_$VERSION
fi

# Check if /usr/include/plugin-api.h exists
if [ -n "${BINUTILS_HEADER_LOCATION}" ]; then
  if [ ! -f "${BINUTILS_HEADER_LOCATION}/plugin-api.h" ]; then
    echo "Error: plugin-api.h not found in the specified --binutils-header-location (${BINUTILS_HEADER_LOCATION})"
    exit 1
  else
    echo "plugin-api.h was found at ${BINUTILS_HEADER_LOCATION}"
    BINUTILS_HEADER_DIR=${BINUTILS_HEADER_LOCATION}
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
    BINUTILS_HEADER_DIR=${BINUTILS_INSTALL_DIR}/include
  fi
fi

# Add build type condition
if [ "$LINK_TYPE" == "static" ]; then
  cmake ../  \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="clang;openmp;clang-tools-extra" \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="SPIRV" \
    -DLLVM_ENABLE_ASSERTIONS=On \
    -DLLVM_BINUTILS_INCDIR=${BINUTILS_HEADER_DIR} \
    -DCMAKE_CXX_LINK_FLAGS="-Wl,-rpath,${gcc_base_path}/lib64 -L${gcc_base_path}/lib64"
elif [ "$LINK_TYPE" == "dynamic" ]; then
  cmake ../ \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DCMAKE_INSTALL_RPATH=${INSTALL_DIR}/lib \
    -DLLVM_ENABLE_PROJECTS="clang;openmp;clang-tools-extra" \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="SPIRV" \
    -DLLVM_LINK_LLVM_DYLIB=ON \
    -DLLVM_BUILD_LLVM_DYLIB=ON \
    -DLLVM_PARALLEL_LINK_JOBS=2 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_LINK_FLAGS="-Wl,-rpath,${gcc_base_path}/lib64 -L${gcc_base_path}/lib64" \
    -DLLVM_BINUTILS_INCDIR=${BINUTILS_HEADER_DIR} \
    -DLLVM_ENABLE_ASSERTIONS=On
else
  echo "Invalid link_type. Must be 'static' or 'dynamic'."
  exit 1
fi
