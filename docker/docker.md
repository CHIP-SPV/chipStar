# Docker Files

## Setting up Docker and Running a chipStar Sample

1. Install Docker:
   - For Ubuntu: `sudo apt-get update && sudo apt-get install docker.io`
   - For other systems, follow the official Docker installation guide: https://docs.docker.com/get-docker/

2. Pull the chipStar 'latest' image:
   ```
   docker pull pveleskopglc/chipstar:latest
   ```

3. Run the container:
   ```
   docker run -it pveleskopglc/chipstar:latest /bin/bash
   ```
   This command starts an interactive session without binding any GPUs to the container - a couple of OpenCL CPU runtimes are  setup already.

4. Execute a matrix multiply sample:
   ```
   CHIP_DEVICE_TYPE=cpu CHIP_BE=opencl ~/chipStar/build/samples/0_MatrixMultiply/MatrixMultiply 
   Device name 13th Gen Intel(R) Core(TM) i9-13900K
   Running 1 iterations 
   hipLaunchKernel 0 time taken: 185.903
   hipLaunchKernel BEST TIME: 185.903
   GPU real time taken(ms): 193.007
   matrixMultiplyCPUReference time taken(ms): 2861.23
   Verification PASSED!
   ```

This process will set up Docker, pull the latest chipStar image, start a container with the necessary environment, and run a sample application to verify the setup.



## DockerfileBase Overview

- Base: Ubuntu latest
- User: 'chipStarUser' with sudo, video, render group access
- Core tools: gcc, g++, cmake, python3, git, OpenCL dev environment
- LLVM/Clang: Customizable version (default 15)
- Lmod: For environment module management
- POCL: Portable OpenCL implementation
- Intel OneAPI: Via Miniconda, includes MKL, TBB, DPC++
- Level Zero API: For low-level device control

Purpose: Comprehensive chipStar development environment, supporting both open-source and Intel proprietary frameworks. 


## DockerfileCPPLinter Overview

This layer builds upon the base image and adds:

- Python virtual environment setup
- C++ linting tools installation

Key components:
- Python3 venv: For isolated Python environment
- clang-tools (version 0.13.0): Provides static analysis and linting capabilities for C++
- cpp-linter (version 1.10.0): A tool for linting C++ code

Purpose: Enhances the development environment with code quality tools specifically for C++ projects, enabling better code analysis and consistency checks.

## DockerfileFull Overview

This layer builds upon the base image and adds:

- Multiple LLVM/Clang versions (16, 17, 18)
- POCL (Portable Computing Language) for each LLVM version

Key components:
- LLVM/Clang versions 16, 17, and 18:
  - Built from source
  - Installed in /apps/llvm/${LLVM_VERSION}
  - Environment modules created for each version
- POCL 4.0:
  - Built for each LLVM version
  - Installed in /apps/pocl/4.0-llvm-${LLVM_VERSION}
  - Environment modules created for each version

Build process for each LLVM/POCL pair:
1. Configure and build LLVM
2. Install LLVM and create its environment module
3. Clone POCL repository
4. Configure and build POCL
5. Install POCL and create its environment module

Purpose: Provides a comprehensive development environment with multiple LLVM toolchains and corresponding POCL installations, allowing for flexible OpenCL development and testing across different LLVM versions, suitable for CI use. 



## DockerfileLatest Overview

This layer builds upon the base image and adds:

- Additional Python packages
- LLVM/Clang 15 environment setup
- Vim common tools
- chipStar build and installation

Key components:
- PyYAML: Python package for YAML parsing
- LLVM/Clang 15: Loaded as a module
- Vim common: Includes 'xxd' utility
- chipStar: 
  - Cloned from GitHub
  - Built with CMake (Release mode, HIPBLAS enabled)
  - Installed system-wide
  - Test suite built (for OpenCL backend, CPU device type)

Purpose: Provides a complete environment for chipStar development and testing.


