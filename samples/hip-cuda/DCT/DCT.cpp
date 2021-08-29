/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "../include/HIPUtil.hpp"

#define SAMPLE_VERSION "HIP-Examples-Applications-v1.0"

using namespace appsdk;
using namespace std;

#if !defined(M_PI)
#define M_PI (3.14159265358979323846f)
#endif

namespace dct
{
const float a = cosf(M_PI/16)/2;
const float b = cosf(M_PI/8 )/2;
const float c = cosf(3*M_PI/16)/2;
const float d = cosf(5*M_PI/16)/2;
const float e = cosf(3*M_PI/8)/2;
const float f = cosf(7*M_PI/16)/2;
const float g = 1.0f/sqrtf(8.0f);

/**
 * DCT8x8 mask that is used to calculate Discrete Cosine Transform
 * of an 8x8 matrix
 */
float dct8x8[64] =
{
    g,  a,  b,  c,  g,  d,  e,  f,
    g,  c,  e, -f, -g, -a, -b, -d,
    g,  d, -e, -a, -g,  f,  b,  c,
    g,  f, -b, -d,  g,  c, -e, -a,
    g, -f, -b,  d,  g, -c, -e,  a,
    g, -d, -e,  a, -g, -f,  b, -c,
    g, -c,  e,  f, -g,  a, -b,  d,
    g, -a,  b, -c,  g, -d,  e,  -f
};

/**
* DCT
* Class implements HIP Discrete Cosine Transform
*/

class DCT
{
        unsigned int             seed;    /**< Seed value for random number generation */
        double              setupTime;    /**< Time for setting up OpenCL */
        double        totalKernelTime;    /**< Time for kernel execution */
        double       totalProgramTime;    /**< Time for program execution */
        double    referenceKernelTime;    /**< Time for reference implementation */
        int                     width;    /**< Width of the input array */
        int                    height;    /**< height of the input array */
        float                  *input;    /**< Input array */
        float                 *output;    /**< Output array */
        unsigned int       blockWidth;    /**< width of the blockSize */
        unsigned int        blockSize;    /**< size of the block */
        unsigned int          inverse;    /**< flag for inverse DCT */
        float     *verificationOutput;    /**< Input array for reference implementation */
        float*            inputBuffer;    /**< memory buffer */
        float*           outputBuffer;    /**< memory buffer */
        float*              dctBuffer;    /**< memory buffer */
	    float*        dct_transBuffer;    /**< memory buffer */
		float	     dct8x8_trans[64];    /**< trans_DCT8x8 mask */
        int                iterations;    /**< Number of iteration for kernel execution */
        SDKTimer         *sampleTimer;    /**< SDKTimer object */
        float *din, *ddct, *dout, *ddct_trans;
    public:

        HIPCommandArgs    *sampleArgs;    /**< HIPCommand argument class */

        /**
         * Constructor
         * Initialize member variables
         * @param name name of sample (string)
         */
        DCT()
        {
            seed = 123;
            input = NULL;
            verificationOutput = NULL;
            width = 64;
            height = 64;
            blockWidth = 8;
            blockSize  = blockWidth * blockWidth;
            inverse = 0;
            setupTime = 0;
            totalKernelTime = 0;
            iterations  = 1;
            sampleArgs = new HIPCommandArgs() ;
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
        }

        inline long long get_time()
        {
          struct timeval tv;
          gettimeofday(&tv, 0);
          return (tv.tv_sec * 1000000) + tv.tv_usec;
        }

        /**
         * Allocate and initialize host memory array with random values
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setupDCT();

        /**
         * HIP related initialisations.
         * Set up the environment, Memory buffers
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setupHIP();

        /**
         * Set values for kernels' arguments, enqueue calls to the kernels
         * Get kernel start and end time if timing is enabled
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int runKernels();

        /**
         * Given the blockindices and localIndicies this
         * function calculate the global index
         * @param blockIdx index of the block horizontally
         * @param blockIdy index of the block vertically
         * @param localidx index of the element relative to the block horizontally
         * @param localIdy index of the element relative to the block vertically
         * @param blockWidth width of each block which is 8
         * @param globalWidth Width of the input matrix
         * @return ID in x dimension
         */
        unsigned int getIdx(unsigned int blockIdx, unsigned int blockIdy, unsigned int localIdx,
                       unsigned int localIdy, unsigned int blockWidth, unsigned int globalWidth);

        /**
         * Reference CPU implementation of Discrete Cosine Transform
         * for performance comparison
         * @param output output of the DCT8x8 transform
         * @param input  input array
         * @param dct8x8 8x8 cosine function base used to calculate DCT8x8
         * @param width width of the input matrix
         * @param height height of the input matrix
         * @param numBlocksX number of blocks horizontally
         * @param numBlocksY number of blocks vertically
         * @param inverse  flag to perform inverse DCT
         */
        void DCTCPUReference( float * output,
                              const float * input ,
                              const float * dct8x8 ,
                              const unsigned int    width,
                              const unsigned int    height,
                              const unsigned int   numBlocksX,
                              const unsigned int   numBlocksY,
                              const unsigned int    inverse);

        /**
         * Override from SDKSample. Print sample stats.
         */
        void printStats();

        /**
         * Override from SDKSample. Initialize
         * command line parser, add custom options
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int initialize();

        /**
         * Override from SDKSample, adjust width and height
         * of execution domain, perform all sample setup
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setup();

        /**
         * Override from SDKSample
         * Run HIP DCT
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int run();

        /**
         * Override from SDKSample
         * Clean-up memory allocations
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int cleanup();

        /**
         * Override from SDKSample
         * Verify against reference implementation
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int verifyResults();
};
} //namespace DCT


inline __device__ unsigned int
getIdx(unsigned int blockIdx, unsigned int blockIdy, unsigned int localIdx, unsigned int localIdy, unsigned int blockWidth, unsigned int globalWidth)
{
    unsigned int globalIdx = blockIdx * blockWidth + localIdx;
    unsigned int globalIdy = blockIdy * blockWidth + localIdy;

    return (globalIdy * globalWidth  + globalIdx);
}

/**
 * Perform Discrete Cosine Transform for block of size 8x8
 * in the input matrix
 * @param output output of the DCT8x8 transform
 * @param input  input array
 * @param dct8x8 8x8 consine function base used to calculate DCT8x8
 * @param inter  local memory which stores intermediate result
 * @param width  width of the input matrix
 * @param blockWidth width of each block, 8 here
 * @param inverse  flag to perform inverse DCT
 */

__global__ void DCTKernel(
	                      float * output,
                          float * input,
                          float * dct8x8,
	                      float * dct8x8_trans,
                          const    unsigned int    width,
                          const    unsigned int  blockWidth,
                          const    unsigned int    inverse)
{
    /* get global indices of the element */
    unsigned int globalIdx = hipBlockIdx_x*hipBlockDim_x + hipThreadIdx_x;
    unsigned int globalIdy = hipBlockIdx_y*hipBlockDim_y + hipThreadIdx_y;

    /* get indices of the block to which the element belongs to */
    unsigned int groupIdx  = hipBlockIdx_x;
    unsigned int groupIdy  = hipBlockIdx_y;

    /* get indices relative to the block */
    unsigned int i  = hipThreadIdx_x;
    unsigned int j  = hipThreadIdx_y;
    unsigned int idx = globalIdy * width + globalIdx;

    /* initialise the accumulator */
    float acc = 0.0f;

    __shared__ float inter[64];

    /* AT * X  */
    for(unsigned int k=0; k < blockWidth; k++)
    {
        unsigned int index1 = j*blockWidth +k;
        unsigned int index2 = getIdx(groupIdx, groupIdy, i, k, blockWidth, width);

		 if (inverse)
			acc += dct8x8[index1]*input[index2];
		 else
			acc += dct8x8_trans[index1]*input[index2];
    }

    inter[j*blockWidth + i] = acc;

    /*
     * Make sure all the values of inter that belong to a block
     * are calculated before proceeding further
     */
    __syncthreads();

    /* again initalising the accumulator */
    acc = 0.0f;

    /* (AT * X) * A */
    for(unsigned int k=0; k < blockWidth; k++)
    {
        unsigned int index11 = j* blockWidth + k;
        unsigned int index22 = k*blockWidth + i;

		if (inverse)
			acc += inter[index11]*dct8x8_trans[index22];
		else
			acc += inter[index11]*dct8x8[index22];

    }

    output[idx] = acc;

    __syncthreads();

}

using namespace dct;

int
DCT::setupDCT()
{
    unsigned int inputSizeBytes;

    // allocate and init memory used by host
    inputSizeBytes = width * height * sizeof(float);
    input = (float *) malloc(inputSizeBytes);
    CHECK_ALLOCATION(input, "Failed to allocate host memory. (input)");

    unsigned int outputSizeBytes =  width * height * sizeof(float);
    output = (float *)malloc(outputSizeBytes);
    CHECK_ALLOCATION(output, "Failed to allocate host memory. (output)");

    // random initialisation of input
    fillRandom<float>(input, width, height, 0, 255);

	//Get the dct8x8 transpose
	for(unsigned int j=0; j < blockWidth ; ++j)
        for(unsigned int i = 0; i < blockWidth ; ++i)
        {
			dct8x8_trans[j*blockWidth + i] = dct8x8[i*blockWidth + j];
		}

    // Unless sampleArgs->quiet mode has been enabled, print the INPUT array.

    if(!sampleArgs->quiet)
    {
        printArray<float>(
            "Input",
            input,
            width,
            1);
    }

    return SDK_SUCCESS;
}

int
DCT::setupHIP(void)
{
    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    cout << " System minor " << devProp.minor << endl;
    cout << " System major " << devProp.major << endl;
    cout << " agent prop name " << devProp.name << endl;

    // Set input data to matrix A and matrix B
    hipHostMalloc((void**)&inputBuffer, sizeof(float) * width * height, hipHostMallocDefault);
    hipHostMalloc((void**)&outputBuffer, sizeof(float) * width * height, hipHostMallocDefault);
    hipHostMalloc((void**)&dctBuffer, sizeof(float) * blockSize, hipHostMallocDefault);
    hipHostMalloc((void**)&dct_transBuffer, sizeof(float) * blockSize, hipHostMallocDefault);

    hipHostGetDevicePointer((void**)&din, inputBuffer,0);
    hipHostGetDevicePointer((void**)&dout, outputBuffer,0);
    hipHostGetDevicePointer((void**)&ddct, dctBuffer,0);
    hipHostGetDevicePointer((void**)&ddct_trans, dct_transBuffer,0);

    hipMemcpy(din, input,sizeof(float) * width * height, hipMemcpyHostToDevice);
    hipMemcpy(ddct,  dct8x8,sizeof(float) * blockSize, hipMemcpyHostToDevice);
    hipMemcpy(ddct_trans, dct8x8_trans,sizeof(float) * blockSize, hipMemcpyHostToDevice);

    return SDK_SUCCESS;
}


int
DCT::runKernels(void)
{
    hipEvent_t start, stop;
    float eventMs = 1.0f;

    hipEventCreate(&start);
    hipEventCreate(&stop);

    // Record the start event
    hipEventRecord(start, NULL);

    hipLaunchKernelGGL(DCTKernel,
                  dim3(width/blockWidth, height/blockWidth),
                  dim3(blockWidth,blockWidth),
                  0, 0,
                  outputBuffer ,inputBuffer ,dctBuffer, dct_transBuffer, width, blockWidth, inverse );

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);


    hipEventElapsedTime(&eventMs, start, stop);

    printf ("kernel_time (hipEventElapsedTime) =%6.3fms\n", eventMs);

    hipMemcpy(output, dout,sizeof(float) * width * height, hipMemcpyDeviceToHost);

    return SDK_SUCCESS;
}



unsigned int
DCT::getIdx(unsigned int blockIdx, unsigned int blockIdy, unsigned int localIdx,
            unsigned int localIdy, unsigned int blockWidth, unsigned int globalWidth)
{
    unsigned int globalIdx = blockIdx * blockWidth + localIdx;
    unsigned int globalIdy = blockIdy * blockWidth + localIdy;
    return (globalIdy * globalWidth  + globalIdx);
}


/*
 * Reference implementation of the Discrete Cosine Transform on the CPU
 */
void
DCT::DCTCPUReference( float * verificationOutput,
                      const float * input ,
                      const float * dct8x8 ,
                      const unsigned int    width,
                      const unsigned int    height,
                      const unsigned int   numBlocksX,
                      const unsigned int   numBlocksY,
                      const unsigned int    inverse)
{
    float * temp = (float *)malloc(width*height*sizeof(float));

    // for each block in the image
    for(unsigned int blockIdy=0; blockIdy < numBlocksY; ++blockIdy)
        for(unsigned int blockIdx=0; blockIdx < numBlocksX; ++blockIdx)
        {
            //  First calculate A^T * X for FDCT or A * X for iDCT
            for(unsigned int j=0; j < blockWidth ; ++j)
                for(unsigned int i = 0; i < blockWidth ; ++i)
                {
                    unsigned int index = getIdx(blockIdx, blockIdy, i, j, blockWidth, width);
                    float tmp = 0.0f;
                    for(unsigned int k=0; k < blockWidth; ++k)
                    {
                        // multiply with dct (j,k)
                        unsigned int index1 =  j*blockWidth +k;
                        //input(i,k)
						unsigned int index2 = getIdx(blockIdx, blockIdy, i, k, blockWidth, width);

                        if (inverse)
							tmp += dct8x8[index1]*input[index2];
						else
							tmp += dct8x8_trans[index1]*input[index2];
                    }
                    temp[index] = tmp;
                }


            // And now for FDCT, multiply the result of previous step with A i.e. calculate (A^T * X) * A for FDCT or
			// for iDCT multiply the result of previous step with A^T i.e. calculate (A * X) * A^T
            for(unsigned int j=0; j < blockWidth ; ++j)
                for(unsigned int i = 0; i < blockWidth ; ++i)
                {
                    unsigned int index = getIdx(blockIdx, blockIdy, i, j, blockWidth, width);
                    float tmp = 0.0f;
                    for(unsigned int k=0; k < blockWidth; ++k)
                    {
                        //input(j,k)
                        unsigned int index1 = getIdx(blockIdx, blockIdy, k,j, blockWidth, width);

                        // multiply with dct (k,i)
                        unsigned int index2 =  k*blockWidth + i;

						if (inverse)
							tmp += temp[index1]*dct8x8_trans[index2];
						else
							tmp += temp[index1]*dct8x8[index2];
                    }
                    verificationOutput[index] = tmp;
                }
        }

    free(temp);
}

int DCT::initialize()
{
    // Call base class Initialize to get default configuration
    CHECK_ERROR(sampleArgs->initialize(), SDK_SUCCESS,
                "HIP resource initialization failed");
    Option* width_option = new Option;
    CHECK_ALLOCATION(width_option, "Memory allocation error.\n");

    width_option->_sVersion = "x";
    width_option->_lVersion = "width";
    width_option->_description = "Width of the input matrix";
    width_option->_type = CA_ARG_INT;
    width_option->_value = &width;

    sampleArgs->AddOption(width_option);
    delete width_option;

    Option* height_option = new Option;
    CHECK_ALLOCATION(height_option, "Memory allocation error.\n");

    height_option->_sVersion = "y";
    height_option->_lVersion = "height";
    height_option->_description = "Height of the input matrix";
    height_option->_type = CA_ARG_INT;
    height_option->_value = &height;

    sampleArgs->AddOption(height_option);
    delete height_option;

    Option* num_iterations = new Option;
    CHECK_ALLOCATION(num_iterations, "Memory allocation error.\n");

    num_iterations->_sVersion = "i";
    num_iterations->_lVersion = "iterations";
    num_iterations->_description = "Number of iterations for kernel execution";
    num_iterations->_type = CA_ARG_INT;
    num_iterations->_value = &iterations;

    sampleArgs->AddOption(num_iterations);
    delete num_iterations;

	Option* isInverse = new Option;
    CHECK_ALLOCATION(isInverse, "Memory allocation error.\n");

    isInverse->_sVersion = "inv";
    isInverse->_lVersion = "inverse";
    isInverse->_description ="Run inverse DCT";
    isInverse->_type = CA_ARG_INT;
    isInverse->_value = &inverse;

    sampleArgs->AddOption(isInverse);
    delete isInverse;

    return SDK_SUCCESS;
}

int DCT::setup()
{
    // Make sure the width is a multiple of blockWidth 8 here
    if(width % blockWidth != 0)
    {
        width = (width/blockWidth + 1)*blockWidth;
    }

    // Make sure the height is a multiple of blockWidth 8 here
    if(height%blockWidth !=0)
    {
        height = (height/blockWidth + 1) * blockWidth;
    }
    CHECK_ERROR(setupDCT(), SDK_SUCCESS, " setupDCT failed");
    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);
    if(setupHIP() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }
    sampleTimer->stopTimer(timer);

    setupTime = (double)sampleTimer->readTimer(timer);
    return SDK_SUCCESS;
}


int DCT::run()
{
    // Warm up
    for(int i = 0; i < 2 && iterations != 1; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        if (runKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    std::cout << "Executing kernel for " << iterations
              << " iterations" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    for(int i = 0; i < iterations; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        if (runKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    sampleTimer->stopTimer(timer);
    totalKernelTime = (double)(sampleTimer->readTimer(timer)) / iterations;

    if(!sampleArgs->quiet)
    {
        printArray<float>("Output", output, width,1);
    }

    return SDK_SUCCESS;
}

int DCT::verifyResults()
{
    if(sampleArgs->verify)
    {
        verificationOutput = (float *) malloc(width*height*sizeof(float));
        CHECK_ALLOCATION(verificationOutput,
                         "Failed to allocate host memory. (verificationOutput)");

        /**
         * reference implementation
         */
        int refTimer = sampleTimer->createTimer();

        sampleTimer->resetTimer(refTimer);
        sampleTimer->startTimer(refTimer);
        DCTCPUReference(verificationOutput, input, dct8x8, width, height,
                        width/blockWidth, height/blockWidth, inverse);

        sampleTimer->stopTimer(refTimer);
        referenceKernelTime = sampleTimer->readTimer(refTimer);

        // compare the results and see if they match
		if(compare(output, verificationOutput, width*height))
        {
            std::cout<<"Passed!\n" << std::endl;
            return SDK_SUCCESS;
        }
        else
        {
            std::cout<<"Failed\n" << std::endl;
            return SDK_FAILURE;
        }
    }
    return SDK_SUCCESS;
}

void DCT::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[4] = {"Width", "Height", "Time(sec)", "[Transfer+Kernel]Time(sec)"};
        std::string stats[4];

        sampleTimer->totalTime = setupTime + totalKernelTime;

        stats[0]  = toString(width    , std::dec);
        stats[1]  = toString(height   , std::dec);
        stats[2]  = toString(sampleTimer->totalTime, std::dec);
        stats[3]  = toString(totalKernelTime, std::dec);

        printStatistics(strArray, stats, 4);
    }
}
int DCT::cleanup()
{

    hipHostFree(inputBuffer);
    hipHostFree(outputBuffer);
    hipHostFree(dctBuffer);
    hipHostFree(dct_transBuffer);
    FREE(input);
    FREE(output);
    FREE(verificationOutput);


    return SDK_SUCCESS;
}


int
main(int argc, char * argv[])
{
    DCT hipDCT;
    if(hipDCT.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(hipDCT.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(hipDCT.setup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(hipDCT.run() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(hipDCT.verifyResults() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(hipDCT.cleanup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    hipDCT.printStats();

    return SDK_SUCCESS;
}
