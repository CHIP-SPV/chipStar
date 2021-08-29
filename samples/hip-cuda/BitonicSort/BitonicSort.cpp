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

/**
 * BitonicSort
 */

class BitonicSort
{
        unsigned int                   seed;    /**< Seed value for random number generation */
        double                    setupTime;    /**< Time for setting up HIP */
        double              totalKernelTime;    /**< Time for kernel execution */
        double             totalProgramTime;    /**< Time for program execution */
        double          referenceKernelTime;    /**< Time for reference implementation */
        unsigned int               sortFlag;    /**< Flag to indicate sorting order */
        std::string               sortOrder;    /**< Argument to indicate sorting order */
        unsigned int                 *input;    /**< Input array */
        unsigned int                 length;    /**< length of the array */
        unsigned int     *verificationInput;    /**< Input array for reference implementation */
        unsigned int*           inputBuffer;    /**< memory buffer */
        int                      iterations;    /**< Number of iterations to execute kernel */
        SDKTimer               *sampleTimer;    /**< SDKTimer object */

    public:
        HIPCommandArgs          *sampleArgs;    /**< Command argument class */
        /**
         * Constructor
         * Initialize member variables
         */
        BitonicSort()
        {
            seed = 123;
            sortFlag = 0;
            sortOrder ="desc";
            input = NULL;
            verificationInput = NULL;
            length = 32768;
            setupTime = 0;
            totalKernelTime = 0;
            iterations = 1;
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
         * @return SDK_SUCCESS on success and nonzero on failure
         */
        int setupBitonicSort();

        /**
         * HIP related initialisations.
         * Set up Memory buffers
         * @return SDK_SUCCESS on success and nonzero on failure
         */
        int setupHIP();

        /**
         * Set values for kernels' arguments, enqueue calls to the kernels
         * on to the command queue, wait till end of kernel execution.
         * Get kernel start and end time if timing is enabled
         * @return SDK_SUCCESS on success and nonzero on failure
         */
        int runKernels();

        /**
         * Helper to swap two values if first one is greater
         * @param a an unsigned int value
         * @param b an unsigned int value
         */
        void swapIfFirstIsGreater(unsigned int *a, unsigned int *b);

        /**
         * Reference CPU implementation of Bitonic Sort
         * for performance comparison
         * @param input the input array
         * @param length length of the array
         * @param sortIncreasing flag to indicate sorting order
         */
        void bitonicSortCPUReference(
            unsigned int * input,
            const unsigned int length,
            const bool sortIncreasing);

        /**
         * Override from SDKSample. Print sample stats.
         */
        void printStats();

        /**
         * Override from SDKSample. Initialize
         * command line parser, add custom options
         * @return SDK_SUCCESS on success and nonzero on failure
         */
        int initialize();

        /**
         * Override from SDKSample, adjust width and height
         * of execution domain, perform all sample setup
         * @return SDK_SUCCESS on success and nonzero on failure
         */
        int setup();

        /**
         * Override from SDKSample
         * Run HIP Bitonic Sort
         * @return SDK_SUCCESS on success and nonzero on failure
         */
        int run();

        /**
         * Override from SDKSample
         * Cleanup memory allocations
         * @return SDK_SUCCESS on success and nonzero on failure
         */
        int cleanup();

        /**
         * Override from SDKSample
         * Verify against reference implementation
         * @return SDK_SUCCESS on success and nonzero on failure
         */
        int verifyResults();
};



__global__ void bitonicSort(unsigned int * theArray,
                            const unsigned int stage,
                            const unsigned int passOfStage,
                            const unsigned int direction)
{
    unsigned int sortIncreasing = direction;
    unsigned int threadId = hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x;

    unsigned int pairDistance = 1 << (stage - passOfStage);
    unsigned int blockWidth   = 2 * pairDistance;

    unsigned int leftId = (threadId % pairDistance)
                   + (threadId / pairDistance) * blockWidth;

    unsigned int rightId = leftId + pairDistance;

    unsigned int leftElement = theArray[leftId];
    unsigned int rightElement = theArray[rightId];

    unsigned int sameDirectionBlockWidth = 1 << stage;

    if((threadId/sameDirectionBlockWidth) % 2 == 1)
        sortIncreasing = 1 - sortIncreasing;

    unsigned int greater;
    unsigned int lesser;
    if(leftElement > rightElement)
    {
        greater = leftElement;
        lesser  = rightElement;
    }
    else
    {
        greater = rightElement;
        lesser  = leftElement;
    }

    if(sortIncreasing)
    {
        theArray[leftId]  = lesser;
        theArray[rightId] = greater;
    }
    else
    {
        theArray[leftId]  = greater;
        theArray[rightId] = lesser;
    }
}



int BitonicSort::setupBitonicSort()
{

    int inputSizeBytes = length * sizeof(unsigned int);
    input = (unsigned int *) malloc(inputSizeBytes);
    fillRandom<unsigned int>(input, length, 1, 0, 255);

    if(sampleArgs->verify)
    {
        verificationInput = (unsigned int *) malloc(length * sizeof(unsigned int));
        CHECK_ALLOCATION(verificationInput,
                         "Failed to allocate host memory. (verificationInput)");
        memcpy(verificationInput, input, length * sizeof(unsigned int));
    }

    /*
     * Unless sampleArgs->quiet mode has been enabled, print the INPUT array.
     * No more than 256 values are printed because it clutters the screen
     * and it is not practical to manually compare a large set of numbers
     */
    if(!sampleArgs->quiet)
    {
        printArray<unsigned int>(
            "Unsorted Input",
            input,
            length,
            1);
    }

    return SDK_SUCCESS;
}


int
BitonicSort::setupHIP(void)
{
    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    cout << " System minor " << devProp.minor << endl;
    cout << " System major " << devProp.major << endl;
    cout << " agent prop name " << devProp.name << endl;

    return SDK_SUCCESS;
}


int
BitonicSort::runKernels(void)
{

    hipEvent_t start, stop;

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;

    unsigned int numStages = 0;
    unsigned int temp;

    unsigned int stage;
    unsigned int passOfStage;

    unsigned int globalThreads = {length / 2};
    unsigned int localThreads = 256;

    // allocate and init memory used by host

    hipHostMalloc((void**)&inputBuffer,length * sizeof(unsigned int), hipHostMallocDefault);
    unsigned int *din;
    hipHostGetDevicePointer((void**)&din, inputBuffer,0);
    hipMemcpy(din, input,length * sizeof(unsigned int), hipMemcpyHostToDevice);
    /*
     * This algorithm is run as NS stages. Each stage has NP passes.
     * so the total number of times the kernel call is enqueued is NS * NP.
     *
     * For every stage S, we have S + 1 passes.
     * eg: For stage S = 0, we have 1 pass.
     *     For stage S = 1, we have 2 passes.
     *
     * if length is 2^N, then the number of stages (numStages) is N.
     * Do keep in mind the fact that the algorithm only works for
     * arrays whose size is a power of 2.
     *
     * here, numStages is N.
     *
     * For an explanation of how the algorithm works, please go through
     * the documentation of this sample.
     */

    /*
     * 2^numStages should be equal to length.
     * i.e the number of times you halve length to get 1 should be numStages
     */
    for(temp = length; temp > 1; temp >>= 1)
    {
        ++numStages;
    }

    // Set appropriate arguments to the kernel

    // whether sort is to be in increasing order. TRUE implies increasing
    if(sortOrder.compare("asc")==0)
    {
        sortFlag = 1;
    }
    else if(sortOrder.compare("desc")==0)
    {
        sortFlag = 0;
    }
    else
    {
        std::cout << "Please input asc or desc,the default sort order is desc!" <<
                  std::endl;
        sortFlag = 0;
    }

    for(stage = 0; stage < numStages; ++stage)
    {

        for(passOfStage = 0; passOfStage < stage + 1; ++passOfStage)
        {
        hipEventRecord(start, NULL);

        hipLaunchKernelGGL(bitonicSort,
                        dim3(globalThreads/localThreads),
                        dim3(localThreads),
                        0, 0,
                        inputBuffer ,stage, passOfStage ,sortFlag);

        hipEventRecord(stop, NULL);
        hipEventSynchronize(stop);

        hipEventElapsedTime(&eventMs, start, stop);

        printf ("kernel_time (hipEventElapsedTime) =%6.3fms\n", eventMs);
        }
    }
        hipMemcpy(input, din,length * sizeof(unsigned int), hipMemcpyDeviceToHost);
    return SDK_SUCCESS;
}

void
BitonicSort::swapIfFirstIsGreater(unsigned int *a, unsigned int *b)
{
    if(*a > *b)
    {
        unsigned int temp = *a;
        *a = *b;
        *b = temp;
    }
}

/*
 * sorts the input array (in place) using the bitonic sort algorithm
 * sorts in increasing order if sortIncreasing is TRUE
 * else sorts in decreasing order
 * length specifies the length of the array
 */
void
BitonicSort::bitonicSortCPUReference(
    unsigned int * input,
    const unsigned int length,
    const bool sortIncreasing)
{
    const unsigned int halfLength = length/2;

    unsigned int i;
    for(i = 2; i <= length; i *= 2)
    {
        unsigned int j;
        for(j = i; j > 1; j /= 2)
        {
            bool increasing = sortIncreasing;
            const unsigned int half_j = j/2;

            unsigned int k;
            for(k = 0; k < length; k += j)
            {
                const unsigned int k_plus_half_j = k + half_j;
                unsigned int l;

                if(i < length)
                {
                    if((k == i) || (((k % i) == 0) && (k != halfLength)))
                    {
                        increasing = !increasing;
                    }
                }

                for(l = k; l < k_plus_half_j; ++l)
                {
                    if(increasing)
                    {
                        swapIfFirstIsGreater(&input[l], &input[l + half_j]);
                    }
                    else
                    {
                        swapIfFirstIsGreater(&input[l + half_j], &input[l]);
                    }
                }
            }
        }
    }
}

int BitonicSort::initialize()
{
    // Call base class Initialize to get default configuration
    CHECK_ERROR(sampleArgs->initialize(), SDK_SUCCESS,
                "HIP resource initilization failed");

    // Now add customized options
    Option* array_length = new Option;
    CHECK_ALLOCATION(array_length, "Memory allocation error.\n");

    array_length->_sVersion = "x";
    array_length->_lVersion = "length";
    array_length->_description = "Length of the array to be sorted";
    array_length->_type = CA_ARG_INT;
    array_length->_value = &length;
    sampleArgs->AddOption(array_length);

    delete array_length;

    Option* sort_order = new Option;
    CHECK_ALLOCATION(sort_order, "Memory allocation error.\n");

    sort_order->_sVersion = "s";
    sort_order->_lVersion = "sort";
    sort_order->_description = "Sort in descending/ascending order[desc/asc]";
    sort_order->_type = CA_ARG_STRING;
    sort_order->_value = &sortOrder;
    sampleArgs->AddOption(sort_order);

    delete sort_order;

    Option* num_iterations = new Option;
    CHECK_ALLOCATION(num_iterations, "Memory allocation error.\n");

    num_iterations->_sVersion = "i";
    num_iterations->_lVersion = "iterations";
    num_iterations->_description = "Number of iterations for kernel execution";
    num_iterations->_type = CA_ARG_INT;
    num_iterations->_value = &iterations;

    sampleArgs->AddOption(num_iterations);

    delete num_iterations;

    return SDK_SUCCESS;
}

int BitonicSort::setup()
{
    if(iterations < 1)
    {
        std::cout<<"Error, iterations cannot be 0 or negative. Exiting..\n";
        exit(0);
    }
    int i = length & (length - 1);
    if(i != 0)
    {
        std::cout<<"\nThe input lentgh must be a power of 2\n"<<std::endl;
        return SDK_FAILURE;
    }

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    if(setupHIP())
    {
        return SDK_FAILURE;
    }

    if(setupBitonicSort() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    sampleTimer->stopTimer(timer);

    setupTime = (double)sampleTimer->readTimer(timer);

    return SDK_SUCCESS;
}


int BitonicSort::run()
{
    // Warm up
    for(int i = 0; i < 2 && iterations != 1; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        if(runKernels())
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
        if(runKernels())
        {
            return SDK_FAILURE;
        }
    }
    sampleTimer->stopTimer(timer);
    totalKernelTime = (double)(sampleTimer->readTimer(timer));


    if(!sampleArgs->quiet)
    {


        printArray<unsigned int>("Output", input, length, 1);

    }

    return SDK_SUCCESS;
}

int BitonicSort::verifyResults()
{
    if(sampleArgs->verify)
    {
        /**
          * reference implementation
          * it overwrites the input array with the output
          */
        int refTimer = sampleTimer->createTimer();
        sampleTimer->resetTimer(refTimer);
        sampleTimer->startTimer(refTimer);
        bitonicSortCPUReference(verificationInput, length, sortFlag);
        sampleTimer->stopTimer(refTimer);
        referenceKernelTime = sampleTimer->readTimer(refTimer);

        //hipMemcpy(input, din,inputSizeBytes, hipMemcpyDeviceToHost);

        // compare the results and see if they match
        if(memcmp(input, verificationInput, length*sizeof(unsigned int)) == 0)
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

void BitonicSort::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[4] = {"Elements", "Setup Time (sec)", "Avg. Kernel Time (sec)", "Elements/sec"};
        std::string stats[4];

        sampleTimer->totalTime = ( totalKernelTime/ iterations );

        stats[0]  = toString(length, std::dec);
        stats[1]  = toString(setupTime, std::dec);
        stats[2]  = toString(sampleTimer->totalTime, std::dec);
        stats[3]  = toString(( length/sampleTimer->totalTime ), std::dec);

        printStatistics(strArray, stats, 4);
    }
}
int BitonicSort::cleanup()
{
    // Releases HIP resources (Context, Memory etc.)

    hipFree(inputBuffer);

    FREE(verificationInput);


    return SDK_SUCCESS;
}



int
main(int argc, char * argv[])
{
    BitonicSort hipBitonicSort;

    if(hipBitonicSort.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(hipBitonicSort.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }


    {
        if(hipBitonicSort.setup() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        if(hipBitonicSort.run() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        if(hipBitonicSort.verifyResults() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        if(hipBitonicSort.cleanup() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        hipBitonicSort.printStats();
    }
    return SDK_SUCCESS;
}
