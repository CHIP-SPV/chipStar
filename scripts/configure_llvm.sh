#!/bin/bash

# check arguments
if [ $# -ne 2 ]; then
  echo "Usage: $0 <version> <install_dir>"
  echo "version: LLVM version 15, 16, 17"
  exit 1
fi

# if an error is enountered, exit
set -e

VERSION=$1
INSTALL_DIR=$2
export LLVM_DIR=`pwd`/llvm-project/llvm

# check if llvm-project exists, if not clone it
if [ ! -d llvm-project ]; then
  git clone git@github.com:CHIP-SPV/llvm-project.git -b chipspv-llvm-${VERSION}-patches --depth 1
  cd ${LLVM_DIR}/projects
  git clone git@github.com:CHIP-SPV/SPIRV-LLVM-Translator.git -b chipspv-llvm-${VERSION}-patches --depth 1
  cd ${LLVM_DIR}
else
  # Warn the user, error out
  echo "llvm-project directory already exists. Assuming it's cloned from chipStar."
  cd ${LLVM_DIR}
  # check if already on the desired branch 
  if [ `git branch --show-current` == "chipspv-llvm-${VERSION}-patches" ]; then
    echo "Already on branch chipspv-llvm-${VERSION}-patches"
  else
    echo "Switching to branch chipspv-llvm-${VERSION}-patches"
    git br -D chipspv-llvm-${VERSION}-patches &> /dev/null
    git fetch origin chipspv-llvm-${VERSION}-patches:chipspv-llvm-${VERSION}-patches
    git checkout chipspv-llvm-${VERSION}-patches
  fi
  cd ${LLVM_DIR}/projects/SPIRV-LLVM-Translator
  # check if already on the desired branch
  if [ `git branch --show-current` == "chipspv-llvm-${VERSION}-patches" ]; then
    echo "Already on branch chipspv-llvm-${VERSION}-patches"
  else
    echo "Switching to branch chipspv-llvm-${VERSION}-patches"
    git br -D chipspv-llvm-${VERSION}-patches &> /dev/null
    git fetch origin chipspv-llvm-${VERSION}-patches:chipspv-llvm-${VERSION}-patches
    git checkout chipspv-llvm-${VERSION}-patches
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

cmake ../  \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;openmp" \
  -DLLVM_TARGETS_TO_BUILD=X86

