/*
 * Copyright (c) 2021-22 CHIP-SPV developers
 * Copyright (c) 2020 zjin@anl.gov
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include<stdio.h>
#include<string>
#include<hip/hip_runtime.h>

#define BLOCKSIZE_x 16
#define BLOCKSIZE_y 16

#define N 128
#define M 64
#define W 16



/*****************/
/* HIP MEMCHECK */
/*****************/

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(hipError_t code, std::string file, int line, bool abort=true)
{

   if (code != hipSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %dn", hipGetErrorString(code), file.c_str(), line);

       if (abort) { exit(code); }
    }
}



/*******************/
/* iDivUp FUNCTION */
/*******************/

int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }


/******************/
/* TEST KERNEL 3D */
/******************/

__global__ void test_kernel_3D(hipPitchedPtr devPitchedPtr)
{

   int tidx = blockIdx.x*blockDim.x+threadIdx.x;
   int tidy = blockIdx.y*blockDim.y+threadIdx.y;

   char* devPtr = (char*) devPitchedPtr.ptr;
   size_t pitch = devPitchedPtr.pitch;
   size_t slicePitch = pitch * N;

   for (int w = 0; w < W; w++)
   {
      char* slice = devPtr + w * slicePitch;
      float* row = (float*)(slice + tidy * pitch);
      row[tidx] = row[tidx] * row[tidx];
   }
}


/********/
/* MAIN */
/********/

int main()
{
   float a[N][M][W];

for (int i=0; i<N; i++)
   for (int j=0; j<M; j++)
      for (int w=0; w<W; w++)
      {
           a[i][j][w] = 3.f;
           //printf("row %i column %i depth %i value %f n",i,j,w,a[i][j][w]);
      }



   // --- 3D pitched allocation and host->device memcopy
   hipExtent extent{M * sizeof(float), N, W};

   hipPitchedPtr devPitchedPtr;

   gpuErrchk(hipMalloc3D(&devPitchedPtr, extent));

   hipMemcpy3DParms p = { 0 };

   p.srcPtr.ptr = a;
   p.srcPtr.pitch = M * sizeof(float);
   p.srcPtr.xsize = M;
   p.srcPtr.ysize = N;
   p.dstPtr.ptr = devPitchedPtr.ptr;
   p.dstPtr.pitch = devPitchedPtr.pitch;
   p.dstPtr.xsize = M;
   p.dstPtr.ysize = N;
   p.extent.width = M * sizeof(float);
   p.extent.height = N;
   p.extent.depth = W;
   p.kind = hipMemcpyHostToDevice;

   gpuErrchk(hipMemcpy3D(&p));

   dim3 GridSize(iDivUp(M,BLOCKSIZE_x),iDivUp(N,BLOCKSIZE_y));

   dim3 BlockSize(BLOCKSIZE_y,BLOCKSIZE_x);

   hipLaunchKernelGGL(test_kernel_3D, dim3(GridSize), dim3(BlockSize), 0, 0, devPitchedPtr);

   gpuErrchk(hipPeekAtLastError());

   gpuErrchk(hipDeviceSynchronize());
   p.srcPtr.ptr = devPitchedPtr.ptr;
   p.srcPtr.pitch = devPitchedPtr.pitch;
   p.dstPtr.ptr = a;
   p.dstPtr.pitch = M * sizeof(float);
   p.kind = hipMemcpyDeviceToHost;

   gpuErrchk(hipMemcpy3D(&p));

   int error = 0;
   for (int i=0; i<N; i++)
      for (int j=0; j<M; j++)
         for (int w=0; w<W; w++)
            if (a[i][j][w] != 9.0f) error++;
            //printf("row %i column %i depth %i value %fn",i,j,w,a[i][j][w]);

   if (error) { 
     printf("FAILED\n");
     return 1;
   }
   else {
     printf("PASSED\n");
     return 0;
   }
}

