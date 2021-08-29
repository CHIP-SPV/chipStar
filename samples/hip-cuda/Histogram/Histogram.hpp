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

#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_


#include "hip/hip_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "../include/HIPUtil.hpp"

using namespace appsdk;
using namespace std;

#define SAMPLE_VERSION "HIP-Examples-Applications-v1.0"
#define WIDTH 1024
#define HEIGHT 1024
#define BIN_SIZE 256
#define GROUP_SIZE 16
#define GROUP_ITERATIONS (BIN_SIZE / 2)//This is done to avoid overflow in the kernel
#define SUB_HISTOGRAM_COUNT ((WIDTH * HEIGHT) /(GROUP_SIZE * GROUP_ITERATIONS))


/**
* Histogram
* Class implements 256 Histogram bin implementation

*/

class Histogram
{

        int binSize;             /**< Size of Histogram bin */
        int groupSize;           /**< Number of threads in group */
        int subHistgCnt;         /**< Sub histogram count */
        unsigned int *data;              /**< input data initialized with normalized(0 - binSize) random values */
        int width;               /**< width of the input */
        int height;              /**< height of the input */
        unsigned int *hostBin;           /**< Host result for histogram bin */
        unsigned int *midDeviceBin;      /**< Intermittent sub-histogram bins */
        unsigned int *deviceBin;         /**< Device result for histogram bin */

        double setupTime;        /**< time taken to setup OpenCL resources and building kernel */
        double kernelTime;       /**< time taken to run kernel and read result back */

        unsigned long totalLocalMemory;      /**< Max local memory allowed */
        unsigned long usedLocalMemory;       /**< Used local memory by kernel */

        unsigned int* dataBuf;                 /**< CL memory buffer for data */
        unsigned int* deviceBinBuf;         /**< CL memory buffer for intermittent device bin */

        int iterations;                     /**< Number of iterations for kernel execution */
        bool scalar;                        /**< scalar kernel */
        bool vector;                        /**< vector kernel */
        int vectorWidth;                    /**< vector width used by the kernel*/
        unsigned int globalThreads;
        unsigned int localThreads ;
        int groupIterations;

        SDKTimer    *sampleTimer;      /**< SDKTimer object */

    public:

        HIPCommandArgs   *sampleArgs;   /**< CLCommand argument class */
        /**
        * Constructor
        * Initialize member variables
        * @param name name of sample (string)
        */
        Histogram()
            :
            binSize(BIN_SIZE),
            groupSize(GROUP_SIZE),
            setupTime(0),
            kernelTime(0),
            subHistgCnt(SUB_HISTOGRAM_COUNT),
            groupIterations(GROUP_ITERATIONS),
            data(NULL),
            hostBin(NULL),
            midDeviceBin(NULL),
            deviceBin(NULL),
            iterations(1),
            scalar(false),
            vector(false),
            vectorWidth(0)
        {
            /* Set default values for width and height */
            width = WIDTH;
            height = HEIGHT;
            sampleArgs = new HIPCommandArgs() ;
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
        }


        ~Histogram()
        {
        }

        /**
        * Allocate and initialize required host memory with appropriate values
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setupHistogram();

        /**
        * HIP related initialisations.
        * Set up Context, Device list, Command Queue, Memory buffers
        * Build HIP kernel program executable
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
        * Run HIP Black-Scholes
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

    private:

        /**
        *  Calculate histogram bin on host
        */
        int calculateHostBin();

        /**
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        *
        template<typename T>
        int mapBuffer(cl_mem deviceBuffer, T* &hostPointer, size_t sizeInBytes,
                      cl_map_flags flags=CL_MAP_READ);
        */
        /**
        * clEnqueueUnmapMemObject
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        int unmapBuffer(cl_mem deviceBuffer, void* hostPointer);
        */
};
#endif
