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
#include <math.h>
#include "../include/HIPUtil.hpp"

using namespace appsdk;
using namespace std;

#define SIGNAL_LENGTH (1 << 10)
#define SAMPLE_VERSION "HIP-Examples-Application-v1.0"

/**
 * DwtHaar1D
 * Class implements One-dimensional Haar wavelet decomposition

 */

class DwtHaar1D
{

        unsigned int signalLength;           /**< Signal length (Must be power of 2)*/
        float *inData;                       /**< input data */
        float *dOutData;                     /**< output data */
        float *dPartialOutData;              /**< paritial decomposed signal */
        float *hOutData;                     /**< output data calculated on host */
        double setupTime;                    /**< time taken to setup resources and building kernel */
        double kernelTime;                   /**< time taken to run kernel and read result back */
        float* inDataBuf;                    /**< memory buffer for input data */
        float* dOutDataBuf;                  /**< memory buffer for output data */
        float* dPartialOutDataBuf;           /**< memory buffer for paritial decomposed signal */
        unsigned int maxLevelsOnDevice;      /**< Maximum levels to be computed on device */
        int iterations;                      /**< Number of iterations to be executed on kernel */
        unsigned int        globalThreads;   /**< global NDRange */
        unsigned int         localThreads;   /**< Local Work Group Size */
	    float *din, *dout, *dpart;
        SDKTimer    *sampleTimer;            /**< SDKTimer object */

    public:

        HIPCommandArgs   *sampleArgs;        /**< HIPCommand argument class */
        /**
         * Constructor
         * Initialize member variables
         * @param name name of sample (string)
         */
        DwtHaar1D()
            :
            signalLength(SIGNAL_LENGTH),
            setupTime(0),
            kernelTime(0),
            inData(NULL),
            dOutData(NULL),
            dPartialOutData(NULL),
            hOutData(NULL),
            iterations(1)
        {
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

        ~DwtHaar1D()
        {
        }

        /**
         * Allocate and initialize required host memory with appropriate values
         *  @return returns SDK_SUCCESS on success and SDK_FAILURE otherwise
         */
        int setupDwtHaar1D();

        /**
         * Calculates the value of WorkGroup Size based in global NDRange
         * and kernel properties
         *  @return returns SDK_SUCCESS on success and SDK_FAILURE otherwise
         */
        int setWorkGroupSize();

        /**
         * Override from SDKSample, Generate binary image of given kernel
         * and exit application
         *  @return returns SDK_SUCCESS on success and SDK_FAILURE otherwise
         */

        int setupHIP();

        /**
         * Set values for kernels' arguments, enqueue calls to the kernels
         * on to the command queue, wait till end of kernel execution.
         * Get kernel start and end time if timing is enabled
         *  @return returns SDK_SUCCESS on success and SDK_FAILURE otherwise
         */
        int runKernels();

        /**
         * Override from SDKSample. Print sample stats.
         */
        void printStats();

        /**
         * Override from SDKSample. Initialize
         * command line parser, add custom options
         *  @return returns SDK_SUCCESS on success and SDK_FAILURE otherwise
         */
        int initialize();

        /**
         * Override from SDKSample, adjust width and height
         * of execution domain, perform all sample setup
         * @return returns SDK_SUCCESS on success and SDK_FAILURE otherwise
         */
        int setup();

        /**
         * Override from SDKSample
         * Run HIP DwtHaar1D
         * @return returns SDK_SUCCESS on success and SDK_FAILURE otherwise
         */
        int run();

        /**
         * Override from SDKSample
         * Cleanup memory allocations
         *  @return returns SDK_SUCCESS on success and SDK_FAILURE otherwise
         */
        int cleanup();

        /**
         * Override from SDKSample
         * Verify against reference implementation
         * @return returns SDK_SUCCESS on success and SDK_FAILURE otherwise
         */
        int verifyResults();

    private:

        int groupSize;       /**< Work items in a group */
        int totalLevels;     /**< Total decomposition levels required for given signal length */
        int curSignalLength; /**< Length of signal for given iteration */
        int levelsDone;      /**< levels done */

        /**
         * @brief   Get number of decomposition levels to perform a full decomposition
         *          and also check if the input signal size is suitable
         * @return  returns the number of decomposition levels if they could be detrmined
         *          and the signal length is supported by the implementation,
         *          otherwise it returns SDK_FAILURE
         * @param   length  Length of input signal
         * @param   levels  Number of decoposition levels neessary to perform a full
         *                  decomposition
         *
         */
        int getLevels(unsigned int length, unsigned int* levels);

        /**
         * @brief   Runs the dwtHaar1D kernel to calculate
         *          the approximation coefficients on device
         * @return returns SDK_SUCCESS on success and SDK_FAILURE otherwise
         */
        int runDwtHaar1DKernel();

        /**
        * @brief   Reference implementation to calculates
        *          the approximation coefficients on host
        *          by normalized decomposition
        * @return returns SDK_SUCCESS on success and SDK_FAILURE otherwise
        */
        int calApproxFinalOnHost();

};


__global__ void dwtHaar1D(
                          float *inSignal,
                          float *coefsSignal,
                          float *AverageSignal,
                          unsigned int tLevels,
                          unsigned int signalLength,
                          unsigned int levelsDone,
		                  unsigned int mLevels)
{
    unsigned int localId = hipThreadIdx_x;
    unsigned int groupId =  hipBlockIdx_x;
    unsigned int localSize = hipBlockDim_x;

    __shared__ float sharedArray[32];
    /**
     * Read input signal data from global memory
     * to shared memory
     */
    float t0 = inSignal[groupId * localSize * 2 + localId];
    float t1 = inSignal[groupId * localSize * 2 + localSize + localId];

    // Divide with signal length for normalized decomposition
    if(0 == levelsDone)
    {
       float r = rsqrtf((float)signalLength);
       t0 *= r;
       t1 *= r;
    }
    sharedArray[localId] = t0;
    sharedArray[localSize + localId] = t1;

    __syncthreads();

    unsigned int levels = tLevels > mLevels ? mLevels: tLevels;
    unsigned int activeThreads = (1 << levels) / 2;
    unsigned int midOutPos = signalLength / 2;

    float rsqrt_two = rsqrtf(2.0f);
    for(unsigned int i = 0; i < levels; ++i)
    {

        float data0, data1;
        if(localId < activeThreads)
        {
            data0 = sharedArray[2 * localId];
            data1 = sharedArray[2 * localId + 1];
        }

        /* make sure all work items have read from sharedArray before modifying it */
        __syncthreads();

        if(localId < activeThreads)
        {
            sharedArray[localId] = (data0 + data1) * rsqrt_two;
            unsigned int globalPos = midOutPos + groupId * activeThreads + localId;
            coefsSignal[globalPos] = (data0 - data1) * rsqrt_two;

            midOutPos >>= 1;
        }
        activeThreads >>= 1;
        __syncthreads();
    }

    /**
     * Write 0th element for the next decomposition
     * steps which are performed on host
     */

     if(0 == localId)
        AverageSignal[groupId] = sharedArray[0];
}


int
DwtHaar1D::calApproxFinalOnHost()
{
    // Copy inData to hOutData
    float *tempOutData = (float*)malloc(signalLength * sizeof(float));
    CHECK_ALLOCATION(tempOutData, "Failed to allocate host memory. (tempOutData)");

    memcpy(tempOutData, inData, signalLength * sizeof(float));

    for(unsigned int i = 0; i < signalLength; ++i)
    {
        tempOutData[i] = tempOutData[i] / sqrt((float)signalLength);
    }

    unsigned int length = signalLength;
    while(length > 1u)
    {
        for(unsigned int i = 0; i < length / 2; ++i)
        {
            float data0 = tempOutData[2 * i];
            float data1 = tempOutData[2 * i + 1];

            hOutData[i] = (data0 + data1) / sqrt((float)2);
            hOutData[length / 2 + i] = (data0 - data1) / sqrt((float)2);
        }
        // Copy inData to hOutData
        memcpy(tempOutData, hOutData, signalLength * sizeof(float));

        length >>= 1;
    }

    FREE(tempOutData);
    return SDK_SUCCESS;
}

int
DwtHaar1D::getLevels(unsigned int length, unsigned int* levels)
{
    int returnVal = SDK_FAILURE;

    for(unsigned int i = 0; i < 24; ++i)
    {
        if(length == (1 << i))
        {
            *levels = i;
            returnVal = SDK_SUCCESS;
            break;
        }
    }

    return returnVal;
}

int DwtHaar1D::setupDwtHaar1D()
{
    // signal length must be power of 2
    signalLength = roundToPowerOf2<unsigned int>(signalLength);

    unsigned int levels = 0;
    int result = getLevels(signalLength, &levels);
    CHECK_ERROR(result,SDK_SUCCESS, "signalLength > 2 ^ 23 not supported");

    // Allocate and init memory used by host
    inData = (float*)malloc(signalLength * sizeof(float));
    CHECK_ALLOCATION(inData, "Failed to allocate host memory. (inData)");

    for(unsigned int i = 0; i < signalLength; i++)
    {
        inData[i] = (float)(rand() % 10);
    }

    dOutData = (float*) malloc(signalLength * sizeof(float));
    CHECK_ALLOCATION(dOutData, "Failed to allocate host memory. (dOutData)");

    memset(dOutData, 0, signalLength * sizeof(float));

    dPartialOutData = (float*) malloc(signalLength * sizeof(float));
    CHECK_ALLOCATION(dPartialOutData,
                     "Failed to allocate host memory.(dPartialOutData)");

    memset(dPartialOutData, 0, signalLength * sizeof(float));

    hOutData = (float*)malloc(signalLength * sizeof(float));
    CHECK_ALLOCATION(hOutData, "Failed to allocate host memory. (hOutData)");

    memset(hOutData, 0, signalLength * sizeof(float));

    if(!sampleArgs->quiet)
    {
        printArray<float>("Input Signal", inData, 256, 1);
    }

    return SDK_SUCCESS;
}


int
DwtHaar1D::setupHIP(void)
{
    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    cout << " System minor " << devProp.minor << endl;
    cout << " System major " << devProp.major << endl;
    cout << " agent prop name " << devProp.name << endl;

    hipHostMalloc((void**)&inDataBuf, sizeof(float) * signalLength,hipHostMallocDefault);
    hipHostMalloc((void**)&dOutDataBuf, signalLength * sizeof(float),hipHostMallocDefault);
    hipHostMalloc((void**)&dPartialOutDataBuf, signalLength * sizeof(float),hipHostMallocDefault);
    hipHostGetDevicePointer((void**)&din, inDataBuf,0);
    hipHostGetDevicePointer((void**)&dout, dOutDataBuf,0);
    hipHostGetDevicePointer((void**)&dpart, dPartialOutDataBuf,0);

    return SDK_SUCCESS;
}

int DwtHaar1D::setWorkGroupSize()
{
    globalThreads = curSignalLength >> 1;
    localThreads = groupSize;
    return SDK_SUCCESS;
}

int DwtHaar1D::runDwtHaar1DKernel()
{
    this->setWorkGroupSize();

    hipMemcpy(din, inData, sizeof(float) * curSignalLength, hipMemcpyHostToDevice);

    hipEvent_t start, stop;

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;

    // Record the start event
    hipEventRecord(start, NULL);

    hipLaunchKernelGGL(dwtHaar1D,
                  dim3(globalThreads/localThreads),
                  dim3(localThreads),
                  0, 0,
                  inDataBuf ,dOutDataBuf ,dPartialOutDataBuf, totalLevels,     curSignalLength,levelsDone, maxLevelsOnDevice);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);

    printf ("kernel_time (hipEventElapsedTime) =%6.3fms\n", eventMs);

    hipMemcpy(dOutData, dout, signalLength * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(dPartialOutData, dpart, signalLength * sizeof(float), hipMemcpyDeviceToHost);

    return SDK_SUCCESS;
}

int
DwtHaar1D::runKernels(void)
{

    // Calculate thread-histograms
    unsigned int levels = 0;
    unsigned int curLevels = 0;
    unsigned int actualLevels = 0;

    int result = getLevels(signalLength, &levels);
    CHECK_ERROR(result, SDK_SUCCESS, "getLevels() failed");

    actualLevels = levels;

    //max levels on device should be decided by kernelWorkGroupSize

    int tempVar = (int)(4);
    maxLevelsOnDevice = tempVar + 1;

    float* temp = (float*)malloc(signalLength * sizeof(float));
    memcpy(temp, inData, signalLength * sizeof(float));

    levelsDone = 0;
    int one = 1;
    while((unsigned int)levelsDone < actualLevels)
    {
        curLevels = (levels < maxLevelsOnDevice) ? levels : maxLevelsOnDevice;

        // Set the signal length for current iteration
        if(levelsDone == 0)
        {
            curSignalLength = signalLength;
        }
        else
        {
            curSignalLength = (one << levels);
        }

        // Set group size
        groupSize = (1 << curLevels) / 2;

        totalLevels = levels;
        runDwtHaar1DKernel();

        if(levels <= maxLevelsOnDevice)
        {
            dOutData[0] = dPartialOutData[0];
            memcpy(hOutData, dOutData, (one << curLevels) * sizeof(float));
            memcpy(dOutData + (one << curLevels), hOutData + (one << curLevels),
                   (signalLength  - (one << curLevels)) * sizeof(float));
            break;
        }
        else
        {
            levels -= maxLevelsOnDevice;
            memcpy(hOutData, dOutData, curSignalLength * sizeof(float));
            memcpy(inData, dPartialOutData, (one << levels) * sizeof(float));
            levelsDone += (int)maxLevelsOnDevice;
        }

    }


    memcpy(inData, temp, signalLength * sizeof(float));
    free(temp);

    return SDK_SUCCESS;
}

int
DwtHaar1D::initialize()
{
    // Call base class Initialize to get default configuration
    if(sampleArgs->initialize())
    {
        return SDK_FAILURE;
    }

    Option* length_option = new Option;
    CHECK_ALLOCATION(length_option,
                     "Error. Failed to allocate memory (length_option)\n");

    length_option->_sVersion = "x";
    length_option->_lVersion = "signalLength";
    length_option->_description = "Length of the signal";
    length_option->_type = CA_ARG_INT;
    length_option->_value = &signalLength;

    sampleArgs->AddOption(length_option);
    delete length_option;

    Option* iteration_option = new Option;
    CHECK_ALLOCATION(iteration_option,
                     "Error. Failed to allocate memory (iteration_option)\n");

    iteration_option->_sVersion = "i";
    iteration_option->_lVersion = "iterations";
    iteration_option->_description = "Number of iterations for kernel execution";
    iteration_option->_type = CA_ARG_INT;
    iteration_option->_value = &iterations;

    sampleArgs->AddOption(iteration_option);
    delete iteration_option;

    return SDK_SUCCESS;
}

int DwtHaar1D::setup()
{
    if(setupDwtHaar1D() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    if(setupHIP() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    sampleTimer->stopTimer(timer);
    // Compute setup time
    setupTime = (double)(sampleTimer->readTimer(timer));

    return SDK_SUCCESS;
}


int DwtHaar1D::run()
{
    // Warm up
    for(int i = 0; i < 2 && iterations != 1; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        if(runKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    std::cout << "Executing kernel for " <<
              iterations << " iterations" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    for(int i = 0; i < iterations; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        if(runKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    sampleTimer->stopTimer(timer);

    // Compute kernel time
    kernelTime = (double)(sampleTimer->readTimer(timer)) / iterations;

    if(!sampleArgs->quiet)
    {
        printArray<float>("dOutData", dOutData, 256, 1);
    }

    return SDK_SUCCESS;
}

int
DwtHaar1D::verifyResults()
{
    if(sampleArgs->verify)
    {
        // Rreference implementation on host device
        calApproxFinalOnHost();

        // Compare the results and see if they match
        bool result = true;
        for(unsigned int i = 0; i < signalLength; ++i)
        {
            if(fabs(dOutData[i] - hOutData[i]) > 0.1f)
            {
                result = false;
                break;
            }
        }

        if(result)
        {
            std::cout << "Passed!\n" << std::endl;
            return SDK_SUCCESS;
        }
        else
        {
            std::cout << "Failed\n" << std::endl;
            return SDK_FAILURE;
        }
    }

    return SDK_SUCCESS;
}

void DwtHaar1D::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[3] = {"SignalLength", "Time(sec)", "[Transfer+Kernel]Time(sec)"};
        sampleTimer->totalTime = setupTime + kernelTime;

        std::string stats[3];
        stats[0] = toString(signalLength, std::dec);
        stats[1] = toString(sampleTimer->totalTime, std::dec);
        stats[2] = toString(kernelTime, std::dec);

        printStatistics(strArray, stats, 3);
    }
}

int DwtHaar1D::cleanup()
{
    hipHostFree(inDataBuf);
    hipHostFree(dOutDataBuf);
    hipHostFree(dPartialOutDataBuf);

    FREE(inData);
    FREE(dOutData);
    FREE(dPartialOutData);
    FREE(hOutData);

    return SDK_SUCCESS;
}

int
main(int argc, char * argv[])
{
    // Create MonteCalroAsian object
    DwtHaar1D hipDwtHaar1D;

    // Initialization
    if(hipDwtHaar1D.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Parse command line options
    if(hipDwtHaar1D.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Setup
    if(hipDwtHaar1D.setup()!=SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Run
    if(hipDwtHaar1D.run()!=SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Verify
    if(hipDwtHaar1D.verifyResults()!=SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Cleanup resources created
    if(hipDwtHaar1D.cleanup()!=SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Print performance statistics
    hipDwtHaar1D.printStats();

    return SDK_SUCCESS;
}
