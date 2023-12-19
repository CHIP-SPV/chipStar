#!/bin/bash

# if an error is enountered, exit
set -e

# check arguments
if [ $# -ne 4 ]; then
  echo "Usage: $0 <version> <install_dir> <link_type> <only_necessary_spirv_exts>"
  echo "version: LLVM version 15, 16, 17"
  echo "link_type: static or dynamic"
  echo "only_necessary_spirv_exts: on or off"
  exit 1
fi

# check version argument
if [ "$1" != "15" ] && [ "$1" != "16" ] && [ "$1" != "17" ]; then
  echo "Invalid version. Must be 15, 16, or 17."
  exit 1
fi

# check link_type argument
if [ "$3" != "static" ] && [ "$3" != "dynamic" ]; then
  echo "Invalid link_type. Must be 'static' or 'dynamic'."
  exit 1
fi

# check only-necessary-spirv-exts argument
if [ "$4" != "on" ] && [ "$4" != "off" ]; then
  echo "Invalid only_necessary_spirv_exts. Must be 'on' or 'off'."
  exit 1
fi

VERSION=$1
INSTALL_DIR=$2
LINK_TYPE=$3

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
  git clone https://github.com/CHIP-SPV/llvm-project.git -b ${LLVM_BRANCH} --depth 1
  cd ${LLVM_DIR}/projects
  git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git -b ${TRANSLATOR_BRANCH} --depth 1
  cd ${LLVM_DIR}
else
  # Warn the user, error out
  echo "llvm-project directory already exists. Assuming it's cloned from chipStar."
  cd ${LLVM_DIR}
  # check if already on the desired branch 
  if [ `git branch --show-current` == "${LLVM_BRANCH}" ]; then
    echo "Already on branch ${LLVM_BRANCH}"
  else
    echo "Switching to branch ${LLVM_BRANCH}"
    git br -D ${LLVM_BRANCH} &> /dev/null
    git fetch origin ${LLVM_BRANCH}:${LLVM_BRANCH}
    git checkout ${LLVM_BRANCH}
  fi
  cd ${LLVM_DIR}/projects/SPIRV-LLVM-Translator
  # check if already on the desired branch
  if [ `git branch --show-current` == "${TRANSLATOR_BRANCH}" ]; then
    echo "Already on branch ${TRANSLATOR_BRANCH}"
  else
    echo "Switching to branch ${TRANSLATOR_BRANCH}"
    git br -D ${TRANSLATOR_BRANCH} &> /dev/null
    git fetch origin ${TRANSLATOR_BRANCH}:${TRANSLATOR_BRANCH}
    git checkout ${TRANSLATOR_BRANCH}
  fi
  cd ${LLVM_DIR}
fi

# check if the build directory exists, if not create it
if [ ! -d build_$VERSION ]; then
  mkdir build_$VERSION
  cd build_$VERSION
else
  # Warn the user, error out
  echo "Build directory build_$VERSION already exists, please remove it and re-run the script"
  exit 1
fi

# Add build type condition
if [ "$LINK_TYPE" == "static" ]; then
  cmake ../  \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="clang;openmp" \
    -DLLVM_TARGETS_TO_BUILD=host
elif [ "$LINK_TYPE" == "dynamic" ]; then
  cmake ../ \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DCMAKE_INSTALL_RPATH=${INSTALL_DIR}/lib \
    -DLLVM_ENABLE_PROJECTS="clang;openmp" \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_LINK_LLVM_DYLIB=ON \
    -DLLVM_BUILD_LLVM_DYLIB=ON \
    -DLLVM_PARALLEL_LINK_JOBS=2 \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=On
else
  echo "Invalid link_type. Must be 'static' or 'dynamic'."
  exit 1
fi
