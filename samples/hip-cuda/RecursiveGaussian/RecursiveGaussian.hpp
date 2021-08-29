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

#ifndef RECURSIVE_GAUSSIAN_H_
#define RECURSIVE_GAUSSIAN_H_

#include "hip/hip_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "../include/HIPUtil.hpp"
#include "../include/SDKBitMap.hpp"

using namespace appsdk;
using namespace std;

#define INPUT_IMAGE "RecursiveGaussian_Input.bmp"
#define OUTPUT_IMAGE "RecursiveGaussian_Output.bmp"

#define SAMPLE_VERSION "HIP-Examples-Applications-v1.0"

#define GROUP_SIZE 256

/**
* Custom type for gaussian parameters
* precomputation
*/
typedef struct _GaussParms
{
    float nsigma;
    float alpha;
    float ema;
    float ema2;
    float b1;
    float b2;
    float a0;
    float a1;
    float a2;
    float a3;
    float coefp;
    float coefn;
} GaussParms, *pGaussParms;



/**
* Recursive Gaussian
* Class implements OpenRecursive Gaussian sample
*/

class RecursiveGaussian
{
        double setupTime;                /**< time taken to setup Openresources and building kernel */
        double kernelTime;               /**< time taken to run kernel and read result back */

        uchar4* inputImageData;          /**< Input bitmap data to device */
        uchar4* outputImageData;         /**< Output from device */

        uchar4* inputImageBuffer;        /**< memory buffer for input Image*/
        uchar4* tempImageBuffer;         /**< memory buffer for storing the transpose of the image*/
        uchar4* outputImageBuffer;       /**< memory buffer for Output Image*/
        uchar4*
        verificationInput;               /**< Input array for reference implementation */
        uchar4*
        verificationOutput;              /**< Output array for reference implementation */

        SDKBitMap inputBitmap;           /**< Bitmap class object */
        uchar4* pixelData;               /**< Pointer to image data */
        unsigned int pixelSize;          /**< Size of a pixel in BMP format> */
        GaussParms
        oclGP;                           /**< instance of struct to hold gaussian parameters */
        unsigned int width;              /**< Width of image */
        unsigned int height;             /**< Height of image */
        size_t blockSizeX;               /**< Work-group size in x-direction */
        size_t blockSizeY;               /**< Work-group size in y-direction */
        size_t blockSize;                /**< block size for transpose kernel */
        int iterations;                  /**< Number of iterations for kernel execution */
        //uchar4 *din, *dout, *dtemp;

        SDKTimer *sampleTimer;           /**< SDKTimer object */

    public:

        HIPCommandArgs   *sampleArgs;   /**< HIPCommand argument class */

        /**
        * Read bitmap image and allocate host memory
        * @param inputImageName name of the input file
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int readInputImage(std::string inputImageName);

        /**
        * Write output to an image file
        * @param outputImageName name of the output file
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int writeOutputImage(std::string outputImageName);

        /**
        * Preprocess gaussian parameters
        * @param fSigma sigma value
        * @param iOrder order
        * @param pGp pointer to gaussian parameter object
        */
        void computeGaussParms(float fSigma, int iOrder, GaussParms* pGP);

        /**
        * RecursiveGaussian on CPU (for verification)
        * @param input input image
        * @param output output image
        * @param width width of image
        * @param height height of image
        * @param a0..a3, b1, b2, coefp, coefn gaussian parameters
        */
        void recursiveGaussianCPU(uchar4* input, uchar4* output,
                                  const int width, const int height,
                                  const float a0, const float a1,
                                  const float a2, const float a3,
                                  const float b1, const float b2,
                                  const float coefp, const float coefn);

        /**
        * Transpose on CPU (for verification)
        * @param input input image
        * @param output output image
        * @param width width of input image
        * @param height height of input image
        */
        void transposeCPU(uchar4* input, uchar4* output,
                          const int width, const int height);

        /**
        * Constructor
        * Initialize member variables
        */
        RecursiveGaussian()
            : inputImageData(NULL),
              outputImageData(NULL),
              verificationOutput(NULL)
        {
            sampleArgs = new HIPCommandArgs();
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
            pixelSize = sizeof(uchar4);
            pixelData = NULL;
            blockSizeX = GROUP_SIZE;
            blockSizeY = 1;
            blockSize = 1;
            iterations = 1;
        }

        ~RecursiveGaussian()
        {
        }

        inline long long get_time()
        {
          struct timeval tv;
          gettimeofday(&tv, 0);
          return (tv.tv_sec * 1000000) + tv.tv_usec;
        }

        /**
        * Allocate image memory and Load bitmap file
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setupRecursiveGaussian();

        /**
        * Openrelated initialisations.
        * Set up Context, Device list, Command Queue, Memory buffers
        * Build kernel program executable
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
        void recursiveGaussianCPUReference();

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
        * Run OpenSobel Filter
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

#endif // RECURSIVE_GAUSSIAN_H_
