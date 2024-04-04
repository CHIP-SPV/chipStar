### Add LLVM repo to apt

```bash
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 
```

### Install Base Packages

```bash

sudo apt install build-essential libzstd1 libzstd-dev libncurses5-dev libzstd-dev zlib1g-dev cmake

sudo apt-get install ocl-icd-libopencl1 
```

### Install LLVM + Clang + SPIR-V Tools

```bash
sudo apt install llvm-17 llvm-17-dev libclang-17-dev

git clone git@github.com:CHIP-SPV/SPIRV-LLVM-Translator.git
cd SPIRV-LLVM-Translator
git checkout -t origin/chipStar-llvm-17
mkdir build && cd build
cmake ../ # Optionally provide -DCMAKE_INSTALL_PREFIX= if you want to install in non-default dir
make -j 
sudo make install # sudo not required if you installed in non-default local path
```

### Verify that necessary dev packages are present

This was found to be an issue on Ubuntu where `libstdc++-11` and `libstdc++-11-dev` packages were installed but `libstdc++-12` was also present so `clang` would pick 12 and then not find the necessary dependencies

```bash
clang-17 --verbose
Ubuntu clang version 17.0.6 (++20231209124227+6009708b4367-1~exp1~20231209124336.77)
Target: x86_64-pc-linux-gnu
Thread model: posix
InstalledDir: /usr/bin
Found candidate GCC installation: /usr/bin/../lib/gcc/x86_64-linux-gnu/11
Found candidate GCC installation: /usr/bin/../lib/gcc/x86_64-linux-gnu/12
Selected GCC installation: /usr/bin/../lib/gcc/x86_64-linux-gnu/12
Candidate multilib: .;@m64
Selected multilib: .;@m64
Found CUDA installation: /usr/local/cuda, version 12.1
```

Here we see that clang chose GCC-12 so make sure that dev packages are installed

```bash
sudo apt install libstdc++-12-dev
```

### Install python

Python is used for unit testing

```bash
sudo apt install python3
python3 -m pip install  PyYAML
```

### Install & Verify OpenCL

Easiest way to check that installation is working and run a simple example would be to use an OpenCL CPU runtime. Two supported options for this are Intel (works on AMD cpus as well) and PoCL (works on a wide range of hardware)

First you need to install an ICD Loader which provides `[libOpenCL.so](http://libOpenCL.so)` and the `clinfo` tool for checking the properties of a given OpenCL installation

```bash
sudo apt install ocl-icd-libopencl1 clinfo
```

Then, install a runtime which provides and .icd file which is then used by the loader. For example, PoCL which runs on a large number of different systems.

```bash
sudo apt install pocl-opencl-icd
```

### Check that `clinfo` is using the OCL ICD Loader

```bash
clinfo | grep "ICD Loader"
  ICD loader Name                                 OpenCL ICD Loader
```

If you don’t see this, and clinfo just exits, it’s likely that some other [libOpenCL.so](http://libOpenCL.so) got picked up such as one from the CUDA installation.

You can which which [libOpenCL.so](http://libOpenCL.so) is being used by `clinfo`

```bash
ldd `which clinfo`
	linux-vdso.so.1 (0x00007ffefffe8000)
	libOpenCL.so.1 => /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 (0x00007f0264ec4000)
	libdl.so.2 => /usr/lib/x86_64-linux-gnu/libdl.so.2 (0x00007f0264ebf000)
	libc.so.6 => /usr/lib/x86_64-linux-gnu/libc.so.6 (0x00007f0264c00000)
	/lib64/ld-linux-x86-64.so.2 (0x00007f0264f10000)
```

And find all `[libOpenCL.so](http://libOpenCL.so)` on your system by:

```bash
sudo find / -name "libOpenCL.so*"
/snap/firefox/3972/usr/lib/x86_64-linux-gnu/libOpenCL.so.1
/snap/firefox/3972/usr/lib/x86_64-linux-gnu/libOpenCL.so.1.0.0
/snap/firefox/4033/usr/lib/x86_64-linux-gnu/libOpenCL.so.1
/snap/firefox/4033/usr/lib/x86_64-linux-gnu/libOpenCL.so.1.0.0
/usr/lib/x86_64-linux-gnu/libOpenCL.so.1.0.0
/usr/lib/x86_64-linux-gnu/libOpenCL.so.1
/usr/share/man/man7/libOpenCL.so.7.gz
/usr/local/cuda-12.1/targets/x86_64-linux/lib/libOpenCL.so.1.0
/usr/local/cuda-12.1/targets/x86_64-linux/lib/libOpenCL.so.1.0.0
/usr/local/cuda-12.1/targets/x86_64-linux/lib/libOpenCL.so
/usr/local/cuda-12.1/targets/x86_64-linux/lib/libOpenCL.so.1
```

Then use `LD_PRELOAD` to `LD_LIBRARY_PATH` to use the one you want

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libOpenCL.so.1 clinfo -l
Platform #0: NVIDIA CUDA
 `-- Device #0: NVIDIA GeForce RTX 3070
Platform #1: Portable Computing Language
 `-- Device #0: pthread-AMD Ryzen 7 5700G with Radeon Graphics
```

Notice above that we have multiple platforms available - one from CUDA and one from PoCL we just installed. Without changing anything and just running a sample:

```bash
/home/pvelesko/chipStar/build/samples/0_MatrixMultiply/MatrixMultiply
CHIP critical [TID 446954] [1712210555.487863702] : Selected OpenCL device 0 is out of range
```

This happens because by default we select platform #0 and device #0 and expect a GPU

```bash
# Defaults
CHIP_PLATFORM=0
CHIP_DEVICE=0
CHIP_DEVICE_TYPE=gpu
```

So we can select to run on the PoCL platform and cpu device as follows:

```bash
CHIP_PLATFORM=1 CHIP_DEVICE_TYPE=cpu /home/pvelesko/chipStar/build/samples/0_MatrixMultiply/MatrixMultiply
CHIP warning [TID 447571] [1712210779.229144953] : The device might not support subgroup size 32, warp-size sensitive kernels might not work correctly.
Device name pthread-AMD Ryzen 7 5700G with Radeon Graphics
CHIP error [TID 447571] [1712210779.371289725] : hipErrorNotInitialized (CL_INVALID_VALUE ) in /home/pvelesko/chipStar/src/backend/OpenCL/CHIPBackendOpenCL.cc:776:compileIL

CHIP error [TID 447571] [1712210779.371373743] : Caught Error: hipErrorNotInitialized
HIP API error
```

The issue here is that we used PoCL from apt which is not compiled with SPIR-V support. Solution: build and install PoCL from source: https://github.com/pocl/pocl

### Find and select the proper .icd file

Since multiple OpenCL platforms can be available at any given time it’s best to select only a single .icd to simplify device selection.

First, find all the .icd files in your system:

```bash
sudo find / -name "*.icd"
/etc/OpenCL/vendors/intel.icd
/etc/OpenCL/vendors/nvidia.icd
/etc/OpenCL/vendors/pocl.icd
/usr/local/etc/OpenCL/vendors/pocl.icd
/home/pvelesko/pocl/build/pocl.icd
/home/pvelesko/pocl/build/ocl-vendors/pocl-tests.icd
```

And use PoCL version we just installed

```bash
OCL_ICD_VENDORS=/usr/local/etc/OpenCL/vendors/pocl.icd clinfo
```