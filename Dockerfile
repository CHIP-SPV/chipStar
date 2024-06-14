FROM ubuntu:latest


RUN apt update; apt install -y gcc g++ cmake python3 git software-properties-common; \
    add-apt-repository ppa:ocl-icd/ppa;  \
    apt update; \
    apt install -y python3-dev python3-yaml libpython3-dev build-essential cmake git pkg-config make ninja-build ocl-icd-libopencl1 ocl-icd-dev ocl-icd-opencl-dev libhwloc-dev zlib1g zlib1g-dev clinfo dialog apt-utils libxml2-dev
       
#Install SPIRV tools
RUN git clone --depth 1 --branch v2023.2 https://github.com/KhronosGroup/SPIRV-Tools.git; \
    mkdir -p /SPIRV-Tools/build; \
    cd /SPIRV-Tools; python3 utils/git-sync-deps; \
    cd build; \
    cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/opt/SPIRV-Tools/v2023.2 -DSPIRV_SKIP_TESTS=ON; \
    cmake --build . --parallel 4; \
    cmake --install .

#Install LLVM with Chipstar patch
RUN mkdir -p llvm-15; \
    cd llvm-15; \
    git init; \
    git remote add origin https://github.com/CHIP-SPV/llvm-project; \
    git -c protocol.version=2 fetch --no-tags --prune --progress --no-recurse-submodules --depth=1 origin 7368327d27341d07719772570071ab0caa5b76fc; \
    git checkout --progress --force 7368327d27341d07719772570071ab0caa5b76fc; \
    git log -1 --format='%H'; \
    mkdir -p llvm/projects/SPIRV-LLVM-Translator; \
    cd llvm/projects/SPIRV-LLVM-Translator; \
    git init; \
    git remote add origin https://github.com/KhronosGroup/SPIRV-LLVM-Translator; \
    git -c protocol.version=2 fetch --no-tags --prune --progress --no-recurse-submodules --depth=1 origin 13168c2d1822a6e6ca6c2f36dbfa3a23c49cc5fd; \
    git checkout --progress --force 13168c2d1822a6e6ca6c2f36dbfa3a23c49cc5fd; \
    git log -1 --format='%H'; \
    export PKG_CONFIG_PATH=${HOME}/opt/SPIRV-Tools/v2023.2/lib/pkgconfig/; \
    mkdir -p /llvm-15/build; cd /llvm-15; \
    cmake -S llvm -B build -DLLVM_ENABLE_PROJECTS="clang;openmp" -DCMAKE_INSTALL_PREFIX=$HOME/opt/llvm/15 -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_SPIRV_INCLUDE_TESTS=OFF; \
    cd build; \
    cmake --build . --parallel 4; \
    cmake --install .

#Install POCL
RUN mkdir -p pocl; \
    cd pocl; \
    git init; \
    git remote add origin https://github.com/pocl/pocl; \
    git -c protocol.version=2 fetch --no-tags --prune --progress --no-recurse-submodules --depth=1 origin d6ec42378fe6f618b92170d2be45f47eae22343f; \
    git checkout --progress --force d6ec42378fe6f618b92170d2be45f47eae22343f; \
    git log -1 --format='%H'; \ 
    mkdir -p build; \
    cd build; \
    cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/opt/pocl/4.0-15 -DCMAKE_BUILD_TYPE=Release -DWITH_LLVM_CONFIG=$HOME/opt/llvm/15/bin/llvm-config -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF -DSTATIC_LLVM=ON -DKERNELLIB_HOST_CPU_VARIANTS=distro; \
    cmake --build . --parallel 4; \
    cmake --install .q