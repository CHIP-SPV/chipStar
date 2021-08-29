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
#include <cmath>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <malloc.h>

#include "../include/HIPUtil.hpp"

#define SAMPLE_VERSION "HIP-Examples-Applications-v1.0"

using namespace appsdk;
using namespace std;

/**
 * \RISKFREE 0.02f
 * \brief risk free interest rate.
 */
#define RISKFREE 0.02f

/**
 * \VOLATILITY 0.30f
 * \brief Volatility factor for Binomial Option Pricing.
 */
#define VOLATILITY 0.30f


class BinomialOption
{
        double setupTime;                     /**< Time taken to setup resources and building kernel */
        double kernelTime;                    /**< Time taken to run kernel and read result back */
        int numSamples;                       /**< No. of  samples*/
        unsigned int samplesPerVectorWidth;   /**< No. of samples per vector width */
        unsigned int numSteps;                /**< No. of time steps*/
        float* randArray;                     /**< Array of float random numbers */
        float* output;                        /**< Output result */
        float* refOutput;                     /**< Reference result */
        float4* randBuffer;                   /**< memory buffer for random numbers */
        float4* outBuffer;                    /**< memory buffer for output*/
        int iterations;                       /**< Number of iterations for kernel execution */
        SDKTimer    *sampleTimer;             /**< SDKTimer object */
        float *din, *dout;
    private:

        float random(float randMax, float randMin);

    public:
        HIPCommandArgs   *sampleArgs;
        /**
         * Constructor
         * Initialize member variables
         */
        BinomialOption()
          : setupTime(0),
            kernelTime(0),
            randArray(NULL),
            output(NULL),
            refOutput(NULL),
            iterations(1)
        {
            numSamples = 256;
            numSteps = 254;
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

        ~BinomialOption();

        /**
         * Allocate and initialize host memory array with random values
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setupBinomialOption();

        /**
         * HIP related initialisations.
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int setupHIP();

        /**
         * Set values for kernels' arguments, enqueue calls to the kernels
         * on to the command queue, wait till end of kernel execution.
         * Get kernel start and end time if timing is enabled
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int runKernels();

        /**
         * Reference CPU implementation of Binomial Option
         * for performance comparison
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int binomialOptionCPUReference();

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
         * Run HIP BinomialOption
         * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int run();

        /**
         * Override from SDKSample
         * Cleanup memory allocations
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


__global__ void binomial_options(
                                 int numSteps,
                                 const float4* randArray,
                                 float4* out)
{
    // load shared mem
    unsigned int tid = hipThreadIdx_x;
    unsigned int bid = hipBlockIdx_x;

    __shared__ float4 callA[254+1];//numSteps+1 elements
    __shared__ float4 callB[254+1];//numSteps+1 elements

    float4 inRand = randArray[bid];
    float4 s, x, optionYears, dt, vsdt, rdt, r, rInv, u, d, pu, pd, puByr, pdByr, profit;

    s.x = (1.0f - inRand.x) * 5.0f + inRand.x * 30.f;
    s.y = (1.0f - inRand.y) * 5.0f + inRand.y * 30.f;
    s.z = (1.0f - inRand.z) * 5.0f + inRand.z * 30.f;
    s.w = (1.0f - inRand.w) * 5.0f + inRand.w * 30.f;

    x.x = (1.0f - inRand.x) * 1.0f + inRand.x * 100.f;
    x.y = (1.0f - inRand.y) * 1.0f + inRand.y * 100.f;
    x.z = (1.0f - inRand.z) * 1.0f + inRand.z * 100.f;
    x.w = (1.0f - inRand.w) * 1.0f + inRand.w * 100.f;

    optionYears.x = (1.0f - inRand.x) * 0.25f + inRand.x * 10.f;
    optionYears.y = (1.0f - inRand.y) * 0.25f + inRand.y * 10.f;
    optionYears.z = (1.0f - inRand.z) * 0.25f + inRand.z * 10.f;
    optionYears.w = (1.0f - inRand.w) * 0.25f + inRand.w * 10.f;

    dt.x = optionYears.x * (1.0f / (float)numSteps);
    dt.y = optionYears.y * (1.0f / (float)numSteps);
    dt.z = optionYears.z * (1.0f / (float)numSteps);
    dt.w = optionYears.w * (1.0f / (float)numSteps);

    vsdt.x = VOLATILITY * sqrtf(dt.x);
    vsdt.y = VOLATILITY * sqrtf(dt.y);
    vsdt.z = VOLATILITY * sqrtf(dt.z);
    vsdt.w = VOLATILITY * sqrtf(dt.w);

    rdt.x = RISKFREE * dt.x;
    rdt.y = RISKFREE * dt.y;
    rdt.z = RISKFREE * dt.z;
    rdt.w = RISKFREE * dt.w;

    r.x = expf(rdt.x);
    r.y = expf(rdt.y);
    r.z = expf(rdt.z);
    r.w = expf(rdt.w);

    rInv.x = 1.0f / r.x;
    rInv.y = 1.0f / r.y;
    rInv.z = 1.0f / r.z;
    rInv.w = 1.0f / r.w;

    u.x  = expf(vsdt.x);
    u.y  = expf(vsdt.y);
    u.z  = expf(vsdt.z);
    u.w  = expf(vsdt.w);

    d.x = 1.0f / u.x;
    d.y = 1.0f / u.y;
    d.z = 1.0f / u.z;
    d.w = 1.0f / u.w;

    pu.x= (r.x - d.x)/(u.x - d.x);
    pu.y= (r.y - d.y)/(u.y - d.y);
    pu.z= (r.z - d.z)/(u.z - d.z);
    pu.w= (r.w - d.w)/(u.w - d.w);

    pd.x = 1.0f - pu.x;
    pd.y = 1.0f - pu.y;
    pd.z = 1.0f - pu.z;
    pd.w = 1.0f - pu.w;

    puByr.x = pu.x * rInv.x;
    puByr.y = pu.y * rInv.y;
    puByr.z = pu.z * rInv.z;
    puByr.w = pu.w * rInv.w;

    pdByr.x= pd.x * rInv.x;
    pdByr.y= pd.y * rInv.y;
    pdByr.z= pd.z * rInv.z;
    pdByr.w= pd.w * rInv.w;

    profit.x = s.x * expf(vsdt.x * (2.0f * tid - (float)numSteps)) - x.x;
    profit.y = s.y * expf(vsdt.y * (2.0f * tid - (float)numSteps)) - x.y;
    profit.z = s.z * expf(vsdt.z * (2.0f * tid - (float)numSteps)) - x.z;
    profit.w = s.w * expf(vsdt.w * (2.0f * tid - (float)numSteps)) - x.w;

    callA[tid].x = profit.x > 0 ? profit.x : 0.0f;
    callA[tid].y = profit.y > 0 ? profit.y : 0.0f;
    callA[tid].z = profit.z > 0 ? profit.z: 0.0f;
    callA[tid].w = profit.w > 0 ? profit.w: 0.0f;

    __syncthreads();

    for(int j = numSteps; j > 0; j -= 2)
    {
        if(tid < j)
        {
            callB[tid].x = puByr.x * callA[tid].x + pdByr.x * callA[tid + 1].x;
            callB[tid].y = puByr.y * callA[tid].y + pdByr.y * callA[tid + 1].y;
            callB[tid].z = puByr.z * callA[tid].z + pdByr.z * callA[tid + 1].z;
            callB[tid].w = puByr.w * callA[tid].w + pdByr.w * callA[tid + 1].w;
        }
        __syncthreads();

        if(tid < j - 1)
        {
            callA[tid].x = puByr.x * callB[tid].x + pdByr.x * callB[tid + 1].x;
            callA[tid].y = puByr.y * callB[tid].y + pdByr.y * callB[tid + 1].y;
            callA[tid].z = puByr.z * callB[tid].z + pdByr.z * callB[tid + 1].z;
            callA[tid].w = puByr.w * callB[tid].w + pdByr.w * callB[tid + 1].w;
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if(tid == 0) out[bid] = callA[0];

}

int
BinomialOption::setupBinomialOption()
{
    // Make numSamples multiple of 4
    numSamples = (numSamples / 4)? (numSamples / 4) * 4: 4;

    samplesPerVectorWidth = numSamples / 4;

    randArray = (float*)memalign(16, samplesPerVectorWidth * sizeof(float4));

    CHECK_ALLOCATION(randArray, "Failed to allocate host memory. (randArray)");

    for(int i = 0; i < numSamples; i++)
    {
        randArray[i] = (float)rand() / (float)RAND_MAX;
    }

    output = (float*)malloc(samplesPerVectorWidth * sizeof(float4));

    CHECK_ALLOCATION(output, "Failed to allocate host memory. (output)");
    memset(output, 0, samplesPerVectorWidth * sizeof(float4));

    return SDK_SUCCESS;
}

int
BinomialOption::setupHIP()
{

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    cout << " System minor " << devProp.minor << endl;
    cout << " System major " << devProp.major << endl;
    cout << " agent prop name " << devProp.name << endl;

    hipHostMalloc((void**)&randBuffer, samplesPerVectorWidth * sizeof(float4),hipHostMallocDefault);
    hipHostMalloc((void**)&outBuffer, samplesPerVectorWidth * sizeof(float4),hipHostMallocDefault);

    hipHostGetDevicePointer((void**)&din, randBuffer,0);
    hipHostGetDevicePointer((void**)&dout, outBuffer,0);

    hipMemcpy(din, randArray, samplesPerVectorWidth * sizeof(float4), hipMemcpyHostToDevice);

    return SDK_SUCCESS;
}

int
BinomialOption::runKernels()
{

    int gpu =0;
    hipSetDevice(gpu);

    hipEvent_t start, stop;

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;

    unsigned int localThreads = {numSteps + 1};

    // Record the start event
    hipEventRecord(start, NULL);

    hipLaunchKernelGGL(binomial_options,
                  dim3(samplesPerVectorWidth),
                  dim3(localThreads),
                  0, 0,
                  numSteps ,(float4*)randBuffer ,(float4*)outBuffer);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);

    printf ("kernel_time (hipEventElapsedTime) =%6.3fms\n", eventMs);

    hipMemcpy(output, dout, samplesPerVectorWidth * sizeof(float4), hipMemcpyDeviceToHost);

    return SDK_SUCCESS;
}

/*
 * Reduces the input array (in place)
 * length specifies the length of the array
 */
int
BinomialOption::binomialOptionCPUReference()
{
    refOutput = (float*)malloc(samplesPerVectorWidth * sizeof(float4));
    CHECK_ALLOCATION(refOutput, "Failed to allocate host memory. (refOutput)");

    float* stepsArray = (float*)malloc((numSteps + 1) * sizeof(float4));
    CHECK_ALLOCATION(stepsArray, "Failed to allocate host memory. (stepsArray)");

    // Iterate for all samples
    for(int bid = 0; bid < numSamples; ++bid)
    {
        float s[4];
        float x[4];
        float vsdt[4];
        float puByr[4];
        float pdByr[4];
        float optionYears[4];

        float inRand[4];

        for(int i = 0; i < 4; ++i)
        {
            inRand[i] = randArray[bid + i];
            s[i] = (1.0f - inRand[i]) * 5.0f + inRand[i] * 30.f;
            x[i] = (1.0f - inRand[i]) * 1.0f + inRand[i] * 100.f;
            optionYears[i] = (1.0f - inRand[i]) * 0.25f + inRand[i] * 10.f;
            float dt = optionYears[i] * (1.0f / (float)numSteps);
            vsdt[i] = VOLATILITY * sqrtf(dt);
            float rdt = RISKFREE * dt;
            float r = expf(rdt);
            float rInv = 1.0f / r;
            float u = expf(vsdt[i]);
            float d = 1.0f / u;
            float pu = (r - d)/(u - d);
            float pd = 1.0f - pu;
            puByr[i] = pu * rInv;
            pdByr[i] = pd * rInv;
        }

        for(int j = 0; j <= numSteps; j++)
        {
            for(int i = 0; i < 4; ++i)
            {
                float profit = s[i] * expf(vsdt[i] * (2.0f * (float)j - (float)numSteps)) - x[i];
                stepsArray[j * 4 + i] = profit > 0.0f ? profit : 0.0f;
            }
        }

        for(int j = numSteps; j > 0; --j)
        {
            for(int k = 0; k <= j - 1; ++k)
            {
                for(int i = 0; i < 4; ++i)
                {
                    int index_k = k * 4 + i;
                    int index_k_1 = (k + 1) * 4 + i;
                    stepsArray[index_k] = pdByr[i] * stepsArray[index_k_1] + puByr[i] *
                                          stepsArray[index_k];
                }
            }
        }

        //Copy the root to result
        refOutput[bid] = stepsArray[0];
    }

    free(stepsArray);

    return SDK_SUCCESS;
}

int BinomialOption::initialize()
{
    // Call base class Initialize to get default configuration
    CHECK_ERROR(sampleArgs->initialize(), SDK_SUCCESS,
                " Resource Intilization failed");

    Option* num_samples = new Option;
    CHECK_ALLOCATION(num_samples,"Error. Failed to allocate memory (num_samples)\n");

    num_samples->_sVersion = "x";
    num_samples->_lVersion = "samples";
    num_samples->_description = "Number of samples to be calculated";
    num_samples->_type = CA_ARG_INT;
    num_samples->_value = &numSamples;

    sampleArgs->AddOption(num_samples);

    delete num_samples;

    Option* num_iterations = new Option;
    CHECK_ALLOCATION(num_iterations, "Error. Failed to allocate memory (num_iterations)\n");

    num_iterations->_sVersion = "i";
    num_iterations->_lVersion = "iterations";
    num_iterations->_description = "Number of iterations for kernel execution";
    num_iterations->_type = CA_ARG_INT;
    num_iterations->_value = &iterations;

    sampleArgs->AddOption(num_iterations);

    delete num_iterations;

    return SDK_SUCCESS;
}

int BinomialOption::setup()
{
    if(setupBinomialOption())
    {
        return SDK_FAILURE;
    }

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);
    if(setupHIP())
    {
        return SDK_FAILURE;
    }
    sampleTimer->stopTimer(timer);
    // Compute setup time
    setupTime = (double)(sampleTimer->readTimer(timer));

    return SDK_SUCCESS;
}


int BinomialOption::run()
{
    // Warm up
    for(int i = 0; i < 2 && iterations != 1; i++)
    {
        if(runKernels())
        {
            return SDK_FAILURE;
        }
    }

    std::cout << "Executing kernel for " << iterations
              << " iterations" << std::endl;
    std::cout << "-------------------------------------------"
              << std::endl;

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    for(int i = 0; i < iterations; i++)
    {
        if(runKernels())
        {
            return SDK_FAILURE;
        }
    }
    sampleTimer->stopTimer(timer);

    // Compute average kernel time
    kernelTime = (double)(sampleTimer->readTimer(timer)) / iterations;

    if(!sampleArgs->quiet)
    {
	printArray<float>("input", randArray, numSamples, 1);
	printArray<float>("Output", output, numSamples, 1);
    }

    return SDK_SUCCESS;
}

int BinomialOption::verifyResults()
{
    if(sampleArgs->verify)
    {
        /**
         * reference implementation
         * it overwrites the input array with the output
         */
        int result = SDK_SUCCESS;
        result = binomialOptionCPUReference();
        CHECK_ERROR(result, SDK_SUCCESS, " verifyResults  failed");

        // compare the results and see if they match
        if(compare(output, refOutput, numSamples, 0.001f))
        {
            std::cout << "Passed!\n" << std::endl;
            return SDK_SUCCESS;
        }
        else
        {
            std::cout <<" Failed\n" << std::endl;

            std::cout <<"\n\n\nNo. Output Output(hex) Refoutput Refoutput(hex)\n";
            for(int i = 0; i < numSamples; ++i)
            {
                if(fabs(output[i] - refOutput[i])> 0.0001)
                {

                    printf(" [%d] %f %#x ", i, output[i], *(int*)&output[i]);
                    printf(" %f %#x, \n", refOutput[i], *(int*)&refOutput[i]);
                }
            }

            return SDK_FAILURE;
        }
    }
    return SDK_SUCCESS;
}

void BinomialOption::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[4] =
        {
            "Option Samples",
            "Time(sec)",
            "Transfer+kernel(sec)" ,
            "Options/sec"
        };

        sampleTimer->totalTime = setupTime + kernelTime;

        std::string stats[4];
        stats[0] = toString(numSamples, std::dec);
        stats[1] = toString(sampleTimer->totalTime, std::dec);
        stats[2] = toString(kernelTime, std::dec);
        stats[3] = toString(numSamples / sampleTimer->totalTime, std::dec);

        printStatistics(strArray, stats, 4);
    }
}

int
BinomialOption::cleanup()
{
    hipHostFree(randBuffer);
    hipHostFree(outBuffer);

    hipDeviceReset();

    return SDK_SUCCESS;
}

BinomialOption::~BinomialOption()
{

    FREE(randArray);

    FREE(output);

    FREE(refOutput);

}

int
main(int argc, char * argv[])
{

    BinomialOption hipBinomialOption;

    if(hipBinomialOption.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(hipBinomialOption.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(hipBinomialOption.setup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(hipBinomialOption.run() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(hipBinomialOption.verifyResults() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(hipBinomialOption.cleanup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    hipBinomialOption.printStats();

    return SDK_SUCCESS;
}
