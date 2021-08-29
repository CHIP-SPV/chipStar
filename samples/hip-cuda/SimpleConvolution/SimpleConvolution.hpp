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

#ifndef SIMPLECONVOLUTION_H_
#define SIMPLECONVOLUTION_H_

/**
 * Header Files
 */
#include "hip/hip_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "../include/HIPUtil.hpp"
#include "FilterCoeff.h"

#define mad(a,b,c) (a*b +c)
#define GROUP_SIZE 256
#define SAMPLE_VERSION "AMD-APP-SDK-vx.y.z.s"

using namespace appsdk;

/**
 * SimpleConvolution
 * Class implements HIP SimpleConvolution sample
 */

class SimpleConvolution
{
        unsigned int                  seed;               /**< Seed value for random number generation */
        double    setupTime;                              /**< Time for setting up HIP */
        double    totalNonSeparableKernelTime;            /**< Time for Non-Separable kernel execution */
	double    totalSeparableKernelTime;	          /**< Time for Separable kernel execution */

        int       width;                                  /**< Width of the Input array */
        int       height;                                 /**< Height of the Input array */
	int	  paddedWidth;	                          /**< Padded Width of the Input array */
	int	  paddedHeight;	                          /**< Padded Height of the Input array */
        unsigned int      *input;		          /**< Input array */
	unsigned int *paddedInput;	                  /**< Padded Input array */
	float     *tmpOutput;                             /**< Temporary Output array to store result of first pass kernel */
        int	*output;                                  /**< Non-Separable Output array */
	int	*outputSep;                               /**< Separable Output array */
        float     *mask;                                  /**< mask array */
        unsigned int      maskWidth;                      /**< mask dimensions */
        unsigned int      maskHeight;                     /**< mask dimensions */
	float     *rowFilter;	                          /**< Row-wise filter for pass1 */
	float     *colFilter;	                          /**< Column-wise filter for pass2 */
	unsigned int	filterSize;		          /**< FilterSize */
	int	filterRadius;		                  /**< FilterRadius */
        int  	*verificationOutput;                      /**< Output array for reference implementation */

        unsigned int*       inputBuffer;        /**< memory input buffer */
	float*       tmpOutputBuffer;    /**<memory temporary output buffer */
        int*       outputBuffer;       /**<  memory output buffer for Non-Separable kernel */
	int*      outputBufferSep;    /**<  memory output buffer for Separable Kernel */
        float*       maskBuffer;         /**<  memory mask buffer */
	float*       rowFilterBuffer;    /**< memory row filter buffer */
	float*      colFilterBuffer;    /**< memory col filter buffer */

        unsigned int       globalThreads;   /**< global NDRange */
        unsigned int       localThreads;    /**< Local Work Group Size */
		int			 localSize;			 /**< User-specified Local Work Group Size */
        int          iterations;         /**< Number of iterations to execute kernel */

        SDKTimer *sampleTimer;      /**< SDKTimer object */

    public:

        HIPCommandArgs   *sampleArgs;   /**< HIPCommand argument class */

        /**
         * Constructor
         * Initialize member variables
         */
        SimpleConvolution()
        {
            sampleArgs = new HIPCommandArgs();
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
            seed = 123;
            input = NULL;
            output = NULL;
			tmpOutput = NULL;
			outputSep = NULL;
            mask   = NULL;
            verificationOutput = NULL;
            width = 512;
            height = 512;
            setupTime = 0;
			totalNonSeparableKernelTime = 0;
            totalSeparableKernelTime = 0;
            iterations = 1;
			localSize = GROUP_SIZE;
        }

        /**
         * Allocate and initialize host memory array with random values
         * @return SDK_SUCCESS on success and SDK_FAILURE0 on failure
         */
        int setupSimpleConvolution();

        /**
         * Calculates the value of WorkGroup Size based in global NDRange
         * and kernel properties
         * @return 0 on success and nonzero on failure
         */
        int setWorkGroupSize();

        /**
         * HIP related initialisations.
         * Set up Context, Device list, Command Queue, Memory buffers
         * Build kernel program executable
         * @return SDK_SUCCESS on success and SDK_FAILURE0 on failure
         */
        int setupHIP();

        /**
		* Call both non-separable and separable HIP implementation of 
		* Convolution
		* @return SDK_SUCCESS on success and SDK_FAILURE0 on failure
		*/
        int runKernels();

		/**
         * Set values for Non-Separable kernels' arguments, enqueue calls to the kernels
         * on to the command queue, wait till end of kernel execution.
         * Get kernel start and end time if timing is enabled
         * @return SDK_SUCCESS on success and SDK_FAILURE0 on failure
         */
	int runNonSeparableKernels();

		/**
         * Set values for Separable kernels' arguments, enqueue calls to the kernels
         * on to the command queue, wait till end of kernel execution.
         * Get kernel start and end time if timing is enabled
         * @return SDK_SUCCESS on success and SDK_FAILURE0 on failure
         */
	int runSeparableKernels();

        /**
         * Reference CPU implementation of Simple Convolution
         * for performance comparison
         */
        void CPUReference();

        /**
         * Override from SDKSample. Print sample stats.
         */
        void printStats();

        /**
         * Override from SDKSample. Initialize
         * command line parser, add custom options
         * @return SDK_SUCCESS on success and SDK_FAILURE0 on failure
         */
        int initialize();

        /**
         * Override from SDKSample, adjust width and height
         * of execution domain, perform all sample setup
         * @return SDK_SUCCESS on success and SDK_FAILURE0 on failure
         */
        int setup();

        /**
         * Override from SDKSample
         * Run HIP SimpleConvolution kernel
         * @return SDK_SUCCESS on success and SDK_FAILURE0 on failure
         */
        int run();

        /**
         * Override from SDKSample
         * Cleanup memory allocations
         * @return SDK_SUCCESS on success and SDK_FAILURE0 on failure
         */
        int cleanup();

        /**
         * Override from SDKSample
         * Verify against reference implementation
         * @return SDK_SUCCESS on success and SDK_FAILURE0 on failure
         */
        int verifyResults();
};



#endif
