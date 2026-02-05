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
  echo "Usage: $0 --version <version> --install-dir <dir> --link-type static(default)/dynamic [--with-binutils [path]] [-N] [--configure-only]"
  echo "--version: LLVM version 17, 18, 19, 20, 21 or 22"
  echo "--install-dir: installation directory"
  echo "--link-type: static or dynamic (default: static)"
  echo "--with-binutils [path]: enable binutils support with optional path to header directory (default: disabled)"
  echo "-N: only emit the cmake configure command without executing it"
  echo "--configure-only: only configure, skip build and install (default: build and install)"
  echo "The llvm-project will be cloned and built under the current working directory.  Patches are first applied from the chipStar project."
  exit 1
fi
# Check if install-dir argument is provided
if [ -z "$INSTALL_DIR" ]; then
  echo "Error: --install-dir argument is required."
  exit 1
fi

# validate version argument
if [ "$VERSION" != "17" ] && [ "$VERSION" != "18" ] && [ "$VERSION" != "19" ] \
       && [ "$VERSION" != "20" ] && [ "$VERSION" != "21" ] && [ "$VERSION" != "22" ]; then
  echo "Invalid version. Must be 17, 18, 19, 20, 21 or 22."
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
    # Ensure clean checkout of the correct branch
    echo "llvm-project directory already exists. Ensuring clean checkout of ${LLVM_BRANCH}..."
    cd llvm-project
    # Fetch latest from upstream
    git fetch origin
    # Checkout the branch (must exist locally or as remote)
    git checkout -f ${LLVM_BRANCH} || git checkout -f origin/${LLVM_BRANCH}
    # Reset hard to upstream branch to ensure clean state
    git reset --hard origin/${LLVM_BRANCH}
    # Clean any untracked files
    git clean -fd
    # Verify we're on the correct branch
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    if [ "$current_branch" != "${LLVM_BRANCH}" ]; then
      echo "Error: Failed to checkout ${LLVM_BRANCH}, currently on $current_branch"
      exit 1
    fi
    
    if [ ! -d llvm/projects/SPIRV-LLVM-Translator ]; then
      echo "Cloning SPIRV-LLVM-Translator from upstream..."
      cd llvm/projects
      git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
      cd SPIRV-LLVM-Translator
      git fetch origin
      git checkout -b ${TRANSLATOR_BRANCH} origin/${TRANSLATOR_BRANCH} 2>/dev/null || git checkout ${TRANSLATOR_BRANCH}
    else
      cd llvm/projects/SPIRV-LLVM-Translator
      git fetch origin
      # Discard any local changes before switching branches
      git reset --hard HEAD
      git clean -fd
      # Ensure clean checkout of translator branch
      git checkout -b ${TRANSLATOR_BRANCH} origin/${TRANSLATOR_BRANCH} 2>/dev/null || git checkout ${TRANSLATOR_BRANCH}
      git reset --hard origin/${TRANSLATOR_BRANCH}
      git clean -fd
    fi
  fi
  cd ${initial_pwd}/llvm-project
  
  # Verify branch is correct before applying patches
  current_branch=$(git rev-parse --abbrev-ref HEAD)
  if [ "$current_branch" != "${LLVM_BRANCH}" ]; then
    echo "Error: Branch mismatch. Expected ${LLVM_BRANCH}, got $current_branch"
    exit 1
  fi
  echo "Verified: On branch ${LLVM_BRANCH} at commit $(git rev-parse --short HEAD)"
  
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
        
        # Skip generic data layout patch for LLVM 21/22 (using version-specific patch instead)
        if [[ "$patch_name" == "0002-fix-SPIR-V-data-layout.patch" ]] && ([ "$VERSION" -eq 21 ] || [ "$VERSION" -eq 22 ]); then
          echo "  Skipping $patch_name (using LLVM ${VERSION}-specific patch instead)"
          continue
        fi
        # Use LLVM 21-specific data layout patch for version 21 only
        if [[ "$patch_name" == "0002-fix-SPIR-V-data-layout-llvm21.patch" ]] && [ "$VERSION" -eq 21 ]; then
          echo "  Applying LLVM 21-specific patch"
        elif [[ "$patch_name" == "0002-fix-SPIR-V-data-layout-llvm21.patch" ]] && [ "$VERSION" -ne 21 ]; then
          echo "  Skipping $patch_name (only needed for LLVM 21)"
          continue
        fi
        # Use LLVM 22-specific data layout patch for version 22
        if [[ "$patch_name" == "0002-fix-SPIR-V-data-layout-llvm22.patch" ]] && [ "$VERSION" -eq 22 ]; then
          echo "  Applying LLVM 22-specific patch"
        elif [[ "$patch_name" == "0002-fix-SPIR-V-data-layout-llvm22.patch" ]] && [ "$VERSION" -ne 22 ]; then
          echo "  Skipping $patch_name (only needed for LLVM 22)"
          continue
        fi
        
        # Skip Unbundle-SDL patch for LLVM 22+ (already upstream)
        if [[ "$patch_name" == "0003-Unbundle-SDL.patch" ]] && [ "$VERSION" -ge 22 ]; then
          echo "  Skipping $patch_name (already upstream in LLVM ${VERSION})"
          continue
        fi
        
        # Skip SPIR-V 1.2 patch for LLVM 22+ (already upstream)
        if [[ "$patch_name" == "0001-Allow-up-to-v1.2-SPIR-V-features.patch" ]] && [ "$VERSION" -ge 22 ]; then
          echo "  Skipping $patch_name (already upstream in LLVM ${VERSION})"
          continue
        fi
        
        # Skip extensions patch for LLVM 22+ (already upstream)
        if [[ "$patch_name" == "0004-only-necessary-exts.patch" ]] && [ "$VERSION" -ge 22 ]; then
          echo "  Skipping $patch_name (already upstream in LLVM ${VERSION})"
          continue
        fi
        
        # HIPSPV new offload driver patches only for LLVM 22+
        if [[ "$patch_name" == "0007-add-chipstar-triple.patch" ]] && [ "$VERSION" -lt 22 ]; then
          echo "  Skipping $patch_name (only needed for LLVM 22+)"
          continue
        fi
        # Skip the LLVM 22-specific version for non-22 versions
        if [[ "$patch_name" == "0007a-add-xoffload-compiler-option-llvm22.patch" ]] && [ "$VERSION" -ne 22 ]; then
          echo "  Skipping $patch_name (only needed for LLVM 22)"
          continue
        fi
        if [[ "$patch_name" == "0007a-add-xoffload-compiler-option.patch" ]]; then
          if [ "$VERSION" -lt 22 ]; then
            echo "  Skipping $patch_name (only needed for LLVM 22+)"
            continue
          elif [ "$VERSION" -eq 22 ]; then
            # Use LLVM 22-specific version that skips Options.td (file moved)
            patch_22_version="${LLVM_PATCH_DIR}/0007a-add-xoffload-compiler-option-llvm22.patch"
            if [ -f "$patch_22_version" ]; then
              echo "  Skipping $patch_name, using LLVM 22-specific version instead"
              continue
            fi
          fi
        fi
        # Apply HIPSPV new offload driver patch for LLVM 22+
        if [[ "$patch_name" == "0008-hipspv-new-offload-driver.patch" ]] && [ "$VERSION" -eq 22 ]; then
          echo "  Skipping $patch_name (using LLVM 22-specific patch instead)"
          continue
        fi
        if [[ "$patch_name" == "0008-hipspv-new-offload-driver.patch" ]] && [ "$VERSION" -lt 22 ]; then
          echo "  Skipping $patch_name (only needed for LLVM 22+)"
          continue
        fi
        # Use LLVM 22-specific new offload driver patch
        if [[ "$patch_name" == "0008-hipspv-new-offload-driver-llvm22.patch" ]] && [ "$VERSION" -eq 22 ]; then
          echo "  Applying LLVM 22-specific new offload driver patch"
        elif [[ "$patch_name" == "0008-hipspv-new-offload-driver-llvm22.patch" ]] && [ "$VERSION" -ne 22 ]; then
          echo "  Skipping $patch_name (only needed for LLVM 22)"
          continue
        fi
        
        # Apply HIP math AMDGCN builtin fix for LLVM 20
        if [[ "$patch_name" == "0009-fix-hip-math-amdgcn-builtins-llvm20.patch" ]] && [ "$VERSION" -ne 20 ]; then
          echo "  Skipping $patch_name (only needed for LLVM 20)"
          continue
        fi
        
        echo "  Applying $patch_name..."
        set +e
        apply_output=$(git apply "$patch" 2>&1)
        apply_status=$?
        set -e
        if [ $apply_status -ne 0 ]; then
          # Try with 3-way merge for data layout patches or 0008 patch that may have conflicts
          if [[ "$patch_name" == *"data-layout"* ]] || [[ "$patch_name" == "0008-hipspv-new-offload-driver-llvm22.patch" ]]; then
            echo "    Attempting 3-way merge for $patch_name..."
            # Stage any modified files that the patch might affect
            git add -u 2>/dev/null || true
            merge_output=$(git apply --3way "$patch" 2>&1)
            merge_status=$?
            if [ $merge_status -eq 0 ] || echo "$merge_output" | grep -q "Applied\|Applied patch\|cleanly"; then
              echo "    Applied with 3-way merge"
              # Check for conflicts and resolve by removing n8:16:32:64
              conflicted_files=$(git diff --name-only --diff-filter=U 2>/dev/null | grep -E "SPIR\.h|SPIRVTargetMachine\.cpp" || true)
              if [ -n "$conflicted_files" ]; then
                echo "    Resolving conflicts in $patch_name..."
                echo "$conflicted_files" | while read file; do
                  if [ -f "$file" ]; then
                    sed -i 's/-n8:16:32:64-G10/-G10/g; s/-n8:16:32:64-G1/-G1/g' "$file"
                    git add "$file" 2>/dev/null || true
                  fi
                done
              fi
              # Also fix any remaining n8:16:32:64 occurrences that weren't conflicted
              for file in clang/lib/Basic/Targets/SPIR.h llvm/lib/Target/SPIRV/SPIRVTargetMachine.cpp; do
                if [ -f "$file" ] && grep -q "n8:16:32:64" "$file"; then
                  sed -i 's/-n8:16:32:64-G10/-G10/g; s/-n8:16:32:64-G1/-G1/g' "$file"
                  git add "$file" 2>/dev/null || true
                fi
              done
            else
              # If 3-way fails, manually apply the fix by removing n8:16:32:64
              echo "    Manually applying fix for $patch_name..."
              for file in clang/lib/Basic/Targets/SPIR.h llvm/lib/Target/SPIRV/SPIRVTargetMachine.cpp; do
                if [ -f "$file" ] && grep -q "n8:16:32:64" "$file"; then
                  sed -i 's/-n8:16:32:64-G10/-G10/g; s/-n8:16:32:64-G1/-G1/g' "$file"
                fi
              done
              # Verify the fix was applied
              if grep -q "n8:16:32:64" clang/lib/Basic/Targets/SPIR.h llvm/lib/Target/SPIRV/SPIRVTargetMachine.cpp 2>/dev/null; then
                echo "Error: Failed to apply $patch_name and manual fix incomplete"
                exit 1
              fi
            fi
          elif echo "$apply_output" | grep -q "does not exist\|No such file"; then
            # For patches that reference moved/deleted files, try 3-way merge
            echo "    File not found, attempting 3-way merge for $patch_name..."
            merge_output=$(git apply --3way "$patch" 2>&1)
            if echo "$merge_output" | grep -q "Applied\|Applied patch"; then
              echo "    Applied with 3-way merge"
              # Check for any remaining conflicts or missing files
              if echo "$merge_output" | grep -q "does not exist"; then
                echo "    Warning: Some files in $patch_name don't exist (may be moved or already upstream)"
              fi
            else
              # Check if the change is already upstream (file moved or change integrated)
              echo "    Warning: $patch_name may have changes already upstream or file moved"
              # For LLVM 22, patch 0007a references Options.td which was moved to Driver/Options.td
              # The important changes (Clang.cpp) should still apply
              if [ "$VERSION" -eq 22 ] && [[ "$patch_name" == "0007a-add-xoffload-compiler-option.patch" ]]; then
                echo "    Continuing - Options.td change not needed for LLVM 22 (file moved)"
              fi
            fi
          else
            echo "Error: Failed to apply $patch_name"
            echo "$apply_output"
            exit 1
          fi
        fi
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
        
        # Use LLVM 21-specific LoopMerge patch, skip original for LLVM 21
        if [[ "$patch_name" == "0003-Fix-LoopMerge-error.patch" ]] && [ "$VERSION" -eq 21 ]; then
          echo "  Skipping $patch_name (using LLVM 21-specific patch instead)"
          continue
        fi
        if [[ "$patch_name" == "0003-Fix-LoopMerge-error-llvm21.patch" ]] && [ "$VERSION" -ne 21 ]; then
          echo "  Skipping $patch_name (only needed for LLVM 21)"
          continue
        fi
        
        # Skip LoopMerge fix for LLVM 22+ (already upstream)
        if [[ "$patch_name" == "0003-Fix-LoopMerge-error.patch" ]] && [ "$VERSION" -ge 22 ]; then
          echo "  Skipping $patch_name (already upstream in LLVM ${VERSION})"
          continue
        fi
        
        # Skip blockMerge fix for LLVM 22+ (already upstream)
        if [[ "$patch_name" == "0004-fix-blockMerge.patch" ]] && [ "$VERSION" -ge 22 ]; then
          echo "  Skipping $patch_name (already upstream in LLVM ${VERSION})"
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

  # check if the build directory exists
  if [ -d build_$VERSION ]; then
    echo "Build directory build_$VERSION already exists. Deleting it..."
    rm -rf build_$VERSION
  fi
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
  
  if [ $? -ne 0 ]; then
    echo "Error: CMake configuration failed"
    exit 1
  fi
  
  echo "CMake configuration completed successfully"
  echo "Build directory: $(pwd)"
  
  # Start building and installing immediately after configuration (unless --configure-only is set)
  if [ "$CONFIGURE_ONLY" != "on" ]; then
    echo "Starting LLVM build..."
    ninja -j $(nproc)
    
    if [ $? -ne 0 ]; then
      echo "Error: LLVM build failed"
      exit 1
    fi
    
    echo "LLVM build completed successfully"
    
    echo "Installing LLVM..."
    ninja install
    
    if [ $? -ne 0 ]; then
      echo "Error: LLVM installation failed"
      exit 1
    fi
    
    echo "LLVM installation completed successfully"
  else
    echo "Configuration-only mode: Skipping build and install"
    echo "To build LLVM, run: ninja -j \$(nproc)"
    echo "To install LLVM, run: ninja install"
  fi
fi
