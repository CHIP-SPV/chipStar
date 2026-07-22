// Reproducer for chipStar issue #1360: HeCBench snake-hip sneaky_snake
// kernel returns wrong results at -O1/-O2/-O3 and correct results at -O0.
//
// Root cause: __clz() in devicelib was implemented as bare __builtin_clz,
// which lowers to llvm.ctlz(x, is_zero_poison=true). CUDA defines
// __clz(0) == 32, but with the poison variant the optimizer may assume
// ctlz's result is <= 31 and fold away range checks on it. Here, with
// ReadSeq == RefSeq every DiagonalResult is 0, __clz(0) is hit on every
// iteration, and clang -O1+ folds the "(Max_leading_zeros/2) < 16" check
// to an unconditional AccumulatedErrs increment, rejecting a perfect
// match. The bad fold is baked into the SPIR-V, so it reproduces on all
// backends (CPU OpenCL, PoCL, Level Zero).
//
// Kernel body copied verbatim from HeCBench src/snake-cuda/kernel.h,
// helpers (lsl/lsr/set_bit) copied verbatim from src/snake-cuda/main.cu.
//
// With ReadSeq == RefSeq there are zero mismatches, so the kernel must
// accept the read: Ftest_Results[0] == 1. This test is compiled at -O2
// (see tests/runtime/CMakeLists.txt) to exercise the miscompile.

#include <hip/hip_runtime.h>
#include <cstdio>

#define NBytes 8

__host__ __device__
inline uint lsr(uint x, int sa) {
  if(sa > 0 && sa < 32) return (x >> sa);
  return x;
}

__host__ __device__
inline uint lsl(uint x, int sa) {
  if (sa > 0 && sa < 32) return (x << sa);
  return x;
}

__host__ __device__
inline uint set_bit(uint &data, int y) {
  data |= lsl(1, y);
  return data;
}

__global__ void sneaky_snake(
  const uint*__restrict__ F_ReadSeq,
  const uint*__restrict__ F_RefSeq,
  int*__restrict__ Ftest_Results,
  const int NumReads,
  const int F_ErrorThreshold)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= NumReads) return;

  uint ReadsPerThread[NBytes];
  uint RefsPerThread[NBytes];

#pragma unroll
  for (int i = 0; i < NBytes; i++)
  {
    ReadsPerThread[i] = F_ReadSeq[tid*8 + i];
    RefsPerThread[i] = F_RefSeq[tid*8 + i];
  }

  /////////////////////////////////////////////////////////////////////////////
  Ftest_Results[tid] = 1;

  uint ReadCompTmp = 0;
  uint RefCompTmp = 0;
  uint DiagonalResult = 0;

  uint ReadTmp1 = 0;
  uint ReadTmp2 = 0;

  uint RefTmp1 = 0;
  uint RefTmp2 = 0;

  uint CornerCase = 0;

  int localCounter= 0;
  int localCounterMax=0;
  int globalCounter = 0;
  int Max_leading_zeros = 0;
  int AccumulatedErrs = 0;

  int Diagonal = 0;
  int ShiftValue = 0;

  int j = 0; //specifying the j-th int that we are reading in each read-ref comparison (can be from 0 to 7)

  while ( (j < 7) && (globalCounter < 200))
  {
    Diagonal = 0;
    RefTmp1 = lsl(RefsPerThread[j], ShiftValue);
    RefTmp2 = lsr(RefsPerThread[j + 1], 32 - ShiftValue);
    ReadTmp1 = lsl(ReadsPerThread[j], ShiftValue);
    ReadTmp2 = lsr(ReadsPerThread[j + 1], 32 - ShiftValue);

    ReadCompTmp = ReadTmp1 | ReadTmp2;
    RefCompTmp = RefTmp1 | RefTmp2;
    DiagonalResult = ReadCompTmp ^ RefCompTmp;
    localCounterMax = __clz(DiagonalResult);

    //////////////////// Upper diagonals /////////////////////

    for(int e = 1; e <= F_ErrorThreshold; e++)
    {
      Diagonal += 1;
      CornerCase = 0;
      if ( (j == 0) && ( (ShiftValue - (2*e)) < 0 ) )
      {
        ReadTmp1 = lsr(ReadsPerThread[j], 2*e - ShiftValue);
        ReadTmp2 = 0;

        ReadCompTmp = ReadTmp1 | ReadTmp2;
        RefCompTmp = RefTmp1 | RefTmp2;

        DiagonalResult = ReadCompTmp ^ RefCompTmp;

        CornerCase = 0;
        for(int Ci = 0; Ci < (2*e) - ShiftValue; Ci++)
        {
          set_bit(CornerCase, 31 - Ci);
        }

        DiagonalResult  = DiagonalResult | CornerCase;
        localCounter = __clz(DiagonalResult);

      }
      else if ( (ShiftValue - (2*e) ) < 0 )
      {
        ReadTmp1 = lsl(ReadsPerThread[j-1], 32 - (2*e - ShiftValue));
        ReadTmp2 = lsr(ReadsPerThread[j], 2*e - ShiftValue);

        ReadCompTmp = ReadTmp1 | ReadTmp2;
        RefCompTmp = RefTmp1 | RefTmp2;

        DiagonalResult = ReadCompTmp ^ RefCompTmp;

        localCounter = __clz(DiagonalResult);
      }
      else
      {
        ReadTmp1 = lsl(ReadsPerThread[j], ShiftValue - 2*e);
        ReadTmp2 = lsr(ReadsPerThread[j+1], 32 - (ShiftValue - 2*e)) ;

        ReadCompTmp = ReadTmp1 | ReadTmp2;
        RefCompTmp = RefTmp1 | RefTmp2;

        DiagonalResult = ReadCompTmp ^ RefCompTmp;

        localCounter = __clz(DiagonalResult);
      }
      if (localCounter>localCounterMax)
        localCounterMax=localCounter;
    }

    //////////////////// Lower diagonals /////////////////////

    for(int e = 1; e <= F_ErrorThreshold; e++)
    {
      Diagonal += 1;
      CornerCase = 0;
      if (j<5)
      {
        if ((ShiftValue + 2*e) < 32)
        {
          ReadTmp1 = lsl(ReadsPerThread[j], ShiftValue + 2*e);
          ReadTmp2 = lsr(ReadsPerThread[j+1], 32 - (ShiftValue + 2*e));

          ReadCompTmp = ReadTmp1 | ReadTmp2;
          RefCompTmp = RefTmp1 | RefTmp2;

          DiagonalResult = ReadCompTmp ^ RefCompTmp;
          localCounter = __clz(DiagonalResult);
        }
        else
        {
          ReadTmp1 = lsl(ReadsPerThread[j+1], (ShiftValue + 2*e) % 32);
          ReadTmp2 = lsr(ReadsPerThread[j+2], 32 - (ShiftValue + 2*e) % 32);

          ReadCompTmp = ReadTmp1 | ReadTmp2;
          RefCompTmp = RefTmp1 | RefTmp2;

          DiagonalResult = 0xffffffff;//ReadCompTmp ^ RefCompTmp;

          DiagonalResult = ReadCompTmp ^ RefCompTmp;

          localCounter = __clz(DiagonalResult);
        }
      }
      else
      {
        ReadTmp1 = lsl(ReadsPerThread[j], ShiftValue + 2*e);
        ReadTmp2 = lsr(ReadsPerThread[j+1], 32 - (ShiftValue + 2*e));

        ReadCompTmp = ReadTmp1 | ReadTmp2;
        RefCompTmp = RefTmp1 | RefTmp2;
        DiagonalResult = ReadCompTmp ^ RefCompTmp;

        CornerCase = 0;
        if ((globalCounter+32)>200) {

          for(int Ci = globalCounter+32-200; Ci < globalCounter+32-200+2*e; Ci++)
          {
            set_bit(CornerCase, Ci);
          }
        }

        else if ((globalCounter+32)>=(200- (2*e))){

          for(int Ci = 0; Ci < (2*e); Ci++)
          {
            set_bit(CornerCase, Ci);
          }
        }
        DiagonalResult = DiagonalResult | CornerCase;

        localCounter = __clz(DiagonalResult);
      }

      if (localCounter>localCounterMax)
        localCounterMax=localCounter;
    }

    Max_leading_zeros = 0;

    if ( (j == 6) && ( ((localCounterMax/2)*2) >= 8) )
    {
      Max_leading_zeros = 8;
      break;
    }
    else if((localCounterMax/2*2) > Max_leading_zeros)
    {
      Max_leading_zeros = ((localCounterMax/2)*2);
    }

    if (((Max_leading_zeros/2) < 16) && (j < 5))
    {
      AccumulatedErrs += 1;
    }
    else if ((j == 6) && ((Max_leading_zeros/2) < 4))
    {
      AccumulatedErrs += 1;
    }

    if(AccumulatedErrs > F_ErrorThreshold)
    {
      Ftest_Results[tid] = 0;
      break;
    }

    if(ShiftValue + Max_leading_zeros + 2 >= 32)
    {
      j += 1;
    }

    // ShiftValue_2Ref = (ShiftValue_2Ref + Max_leading_zeros + 2) %32;
    if (Max_leading_zeros == 32)
    {
      globalCounter += Max_leading_zeros;
    }
    else
    {
      ShiftValue = ((ShiftValue + Max_leading_zeros + 2) % 32);
      globalCounter += (Max_leading_zeros + 2);
    }
  }
}

int main() {
  const uint Seq[NBytes] = {0u,         0x0a0cf5ffu, 0x528f3ce0u, 0x8f430053u,
                            0xbf3c2ef0u, 0x912e5e91u, 0x30000000u, 0u};
  const int NumReads = 1;
  const int F_ErrorThreshold = 0;

  uint *dRead = nullptr, *dRef = nullptr;
  int *dRes = nullptr;
  (void)hipMalloc(&dRead, sizeof(Seq));
  (void)hipMalloc(&dRef, sizeof(Seq));
  (void)hipMalloc(&dRes, sizeof(int));
  (void)hipMemcpy(dRead, Seq, sizeof(Seq), hipMemcpyHostToDevice);
  (void)hipMemcpy(dRef, Seq, sizeof(Seq), hipMemcpyHostToDevice);
  int minusOne = -1;
  (void)hipMemcpy(dRes, &minusOne, sizeof(int), hipMemcpyHostToDevice);

  hipLaunchKernelGGL(sneaky_snake, dim3(1), dim3(1), 0, 0, dRead, dRef, dRes,
                     NumReads, F_ErrorThreshold);
  (void)hipDeviceSynchronize();

  int result = -2;
  (void)hipMemcpy(&result, dRes, sizeof(int), hipMemcpyDeviceToHost);
  (void)hipFree(dRead);
  (void)hipFree(dRef);
  (void)hipFree(dRes);

  // READ == REF and threshold 0: the read must be accepted (result == 1).
  printf("result=%d -> %s\n", result, result == 1 ? "PASS" : "FAIL");
  return result == 1 ? 0 : 1;
}
