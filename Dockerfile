FROM ubuntu:latest

ENV DEBIAN_FRONTEND noninteractive

SHELL ["/bin/bash", "-c"]

# Get the basic stuff
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y \
    sudo

# Create chipStarUser user with sudo privileges
RUN useradd -ms /bin/bash chipStarUser && \
    usermod -aG sudo chipStarUser; \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Set as default user
USER chipStarUser
WORKDIR /home/chipStarUser


RUN sudo apt update; sudo apt install -y gcc g++ cmake python3 git software-properties-common; \
    sudo add-apt-repository ppa:ocl-icd/ppa;  \
    sudo apt update; \
    sudo apt install -y python3-dev python3-yaml libpython3-dev build-essential cmake git pkg-config make ninja-build ocl-icd-libopencl1 ocl-icd-dev ocl-icd-opencl-dev libhwloc-dev zlib1g zlib1g-dev clinfo dialog apt-utils libxml2-dev wget lua5.3 lua-bit32:amd64 lua-posix:amd64 lua-posix-dev liblua5.3-0:amd64 liblua5.3-dev:amd64 tcl tcl-dev tcl8.6 tcl8.6-dev:amd64 libtcl8.6:amd64

ADD configure_llvm.sh /home/chipStarUser/configure_llvm.sh

RUN sudo chown chipStarUser:chipStarUser configure_llvm.sh; \
    wget https://sourceforge.net/projects/lmod/files/Lmod-8.7.tar.bz2; \
    tar -xvf Lmod-8.7.tar.bz2; \
    cd Lmod-8.7; \
    ./configure --prefix=/apps; \
    sudo make install; \
    sudo ln -s /apps/lmod/lmod/init/profile        /etc/profile.d/z00_lmod.sh; \
    sudo mkrid -p /apps/modulefiles/Core ; \
    echo 'if ! shopt -q login_shell; then'  | sudo tee -a /etc/bash.bashrc > /dev/null; \
    echo '  if [ -d /etc/profile.d ]; then'  | sudo tee -a /etc/bash.bashrc > /dev/null; \
    echo '    for i in /etc/profile.d/*.sh; do'  | sudo tee -a /etc/bash.bashrc > /dev/null; \
    echo '      if [ -r $i ]; then'  | sudo tee -a /etc/bash.bashrc > /dev/null; \
    echo '        . $i' | sudo tee -a  /etc/bash.bashrc > /dev/null; \
    echo '      fi'  | sudo tee -a /etc/bash.bashrc > /dev/null; \
    echo '    done'  | sudo tee -a /etc/bash.bashrc > /dev/null; \
    echo '  fi'  | sudo tee -a /etc/bash.bashrc > /dev/null; \
    echo 'fi' | sudo tee -a /etc/bash.bashrc > /dev/null; 

RUN ./configure_llvm.sh 15 /apps/llvm/15 static off; \
    cd llvm-project/llvm/build_15; \
    make -j 16; \
    sudo make install

RUN sudo mkdir -p /apps/modulefiles/Core/llvm/; \
    echo 'prepend_path("CPATH","/apps/llvm/15/include")' | sudo tee  /apps/modulefiles/Core/llvm/15.lua; \
    echo 'prepend_path("LD_LIBRARY_PATH","/apps/llvm/15/lib")' | sudo tee -a /apps/modulefiles/Core/llvm/15.lua; \
    echo 'prepend_path("LIBRARY_PATH","/apps/llvm/15/lib")' | sudo tee -a /apps/modulefiles/Core/llvm/15.lua; \
    echo 'prepend_path("PATH","/apps/llvm/15/libexec")' | sudo tee -a /apps/modulefiles/Core/llvm/15.lua; \
    echo 'prepend_path("PATH","/apps/llvm/15/bin")' | sudo tee -a /apps/modulefiles/Core/llvm/15.lua; \
    echo 'prepend_path("PKG_CONFIG_PATH","/apps/llvm/15/lib/pkgconfig/")' | sudo tee -a /apps/modulefiles/Core/llvm/15.lua

SHELL ["/bin/bash", "-ci"]
RUN ml avail; ml llvm/15; \
    mkdir -p pocl; \
    cd pocl; \
    git init; \
    git remote add origin https://github.com/pocl/pocl; \
    git -c protocol.version=2 fetch --no-tags --prune --progress --no-recurse-submodules --depth=1 origin d6ec42378fe6f618b92170d2be45f47eae22343f; \
    git checkout --progress --force d6ec42378fe6f618b92170d2be45f47eae22343f; \
    git log -1 --format='%H'; \ 
    mkdir -p build; \
    cd build; \
    cmake ..  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/apps/pocl/4.0-llvm-15 -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF -DSTATIC_LLVM=ON -DKERNELLIB_HOST_CPU_VARIANTS=distro; \
    cmake --build . --parallel 4; \
    sudo cmake --install .

RUN sudo mkdir -p /apps/modulefiles/Core/pocl/; \
    echo 'prepend_path("CPATH","/apps/llvm/15/include")' | sudo tee  /apps/modulefiles/Core/pocl/4.0-llvm-15.lua; \
    echo 'prepend_path("LD_LIBRARY_PATH","/apps/pocl/4.0-llvm-15/lib")' | sudo tee -a /apps/modulefiles/Core/pocl/4.0-llvm-15.lua; \
    echo 'prepend_path("LIBRARY_PATH","/apps/llvm/15/lib")' | sudo tee -a /apps/modulefiles/Core/pocl/4.0-llvm-15.lua; \
    echo 'prepend_path("PATH","/apps/pocl/4.0-llvm-15/bin")' | sudo tee -a /apps/modulefiles/Core/pocl/4.0-llvm-15.lua; \
    echo 'prepend_path("XDG_DATA_DIRS","/apps/pocl/4.0-llvm-15/share")' | sudo tee -a /apps/modulefiles/Core/pocl/4.0-llvm-15.lua; \
    echo 'setenv("OPENCL_VENDOR_PATH","/apps/pocl/4.0-llvm-15/etc/OpenCL/vendors")' | sudo tee -a /apps/modulefiles/Core/pocl/4.0-llvm-15.lua;

RUN echo 'load("llvm/15", "pocl/4.0-llvm-15")' | sudo tee  /apps/modulefiles/Core/StdEnv.lua
 
RUN echo 'if [ -z "$__Init_Default_Modules" ]; then' | sudo tee /etc/profile.d/z01_StdEnv.sh; \
    echo '    export __Init_Default_Modules=1;' | sudo tee -a /etc/profile.d/z01_StdEnv.sh; \
    echo '    ## ability to predefine elsewhere the default list' | sudo tee -a /etc/profile.d/z01_StdEnv.sh; \
    echo '    LMOD_SYSTEM_DEFAULT_MODULES=${LMOD_SYSTEM_DEFAULT_MODULES:-"StdEnv"}' | sudo tee -a /etc/profile.d/z01_StdEnv.sh; \
    echo '    export LMOD_SYSTEM_DEFAULT_MODULES' | sudo tee -a /etc/profile.d/z01_StdEnv.sh; \
    echo '    module --initial_load --no_redirect restore' | sudo tee -a /etc/profile.d/z01_StdEnv.sh; \
    echo 'else' | sudo tee -a /etc/profile.d/z01_StdEnv.sh; \
    echo '    module refresh' | sudo tee -a /etc/profile.d/z01_StdEnv.sh; \
    echo 'fi' | sudo tee -a /etc/profile.d/z01_StdEnv.sh

 
RUN echo 'if [ -f /etc/bashrc ]; then' | tee -a  ~/.bashrc; \
    echo '   . /etc/bashrc' | tee -a  ~/.bashrc; \
    echo 'fi' | tee -a  ~/.bashrc

# RUN sudo apt install -y gpg-agent wget; \
#     wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null; \
#     echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list; \
#     sudo apt update; \
#     sudo apt install -y intel-oneapi-compiler-dpcpp-cpp intel-oneapi-runtime-opencl

# RUN git clone https://github.com/intel/compute-runtime neo; \
#     cd neo; mkdir build; cd build;\
#     cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/apps/intel -DNEO_SKIP_UNIT_TESTS=1 ../; \
#     make -j 4; \
#     sudo make install



    
# RUN chmod 744 configure_llvm.sh; configure_llvm.sh 
# #Install SPIRV tools
# RUN git clone --depth 1 --branch v2023.2 https://github.com/KhronosGroup/SPIRV-Tools.git; \
#     mkdir -p /SPIRV-Tools/build; \
#     cd /SPIRV-Tools; python3 utils/git-sync-deps; \
#     cd build; \
#     cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/opt/SPIRV-Tools/v2023.2 -DSPIRV_SKIP_TESTS=ON; \
#     cmake --build . --parallel 4; \
#     cmake --install .

# #Install LLVM with Chipstar patch
# RUN mkdir -p llvm-15; \
#     cd llvm-15; \
#     git init; \
#     git remote add origin https://github.com/CHIP-SPV/llvm-project; \
#     git -c protocol.version=2 fetch --no-tags --prune --progress --no-recurse-submodules --depth=1 origin 7368327d27341d07719772570071ab0caa5b76fc; \
#     git checkout --progress --force 7368327d27341d07719772570071ab0caa5b76fc; \
#     git log -1 --format='%H'; \
#     mkdir -p llvm/projects/SPIRV-LLVM-Translator; \
#     cd llvm/projects/SPIRV-LLVM-Translator; \
#     git init; \
#     git remote add origin https://github.com/KhronosGroup/SPIRV-LLVM-Translator; \
#     git -c protocol.version=2 fetch --no-tags --prune --progress --no-recurse-submodules --depth=1 origin 13168c2d1822a6e6ca6c2f36dbfa3a23c49cc5fd; \
#     git checkout --progress --force 13168c2d1822a6e6ca6c2f36dbfa3a23c49cc5fd; \
#     git log -1 --format='%H'; \
#     export PKG_CONFIG_PATH=${HOME}/opt/SPIRV-Tools/v2023.2/lib/pkgconfig/; \
#     mkdir -p /llvm-15/build; cd /llvm-15; \
#     cmake -S llvm -B build -DLLVM_ENABLE_PROJECTS="clang;openmp" -DCMAKE_INSTALL_PREFIX=$HOME/opt/llvm/15 -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_SPIRV_INCLUDE_TESTS=OFF; \
#     cd build; \
#     cmake --build . --parallel 4; \
#     cmake --install .

# #Install POCL
