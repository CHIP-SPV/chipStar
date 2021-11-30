#!/bin/bash

module use /home/jyoung/gpfs_share/compilers/modulefiles/oneapi/2020.2.0.2997/
#module avail
module load mkl compiler
module load intel_compute_runtime/release/latest

dpcpp onemkl_gemm_wrapper.cpp -DMKL_ILP64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -o libonemkl_gemm_wrapper.so -fsycl -lze_loader -shared -fPIC

module use /home/bertoni/modulefiles
module load hiplz/wip
rm ./a.out
clang++ -std=c++11 hiplz_sycl_interop.cpp -lhiplz -lOpenCL -lze_loader -lonemkl_gemm_wrapper -L.


export LD_LIBRARY_PATH=/home/bertoni/projects/p051.hiplz_mkl/:$LD_LIBRARY_PATH
./a.out
