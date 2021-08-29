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

#include "RecursiveGaussian.hpp"
#include <cmath>


/*
 * Transpose Kernel 
 * input image is transposed by reading the data into a block
 * and writing it to output image
 */
__global__ void transpose_kernel( 
                                 uchar4 *output,
                                 uchar4  *input,
                                 unsigned int    width,
                                 unsigned int    height,
                                 unsigned int blockSize)
{

        __shared__  uchar4 block[16*16];
	unsigned int globalIdx = hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x;
	unsigned int globalIdy = hipBlockDim_y*hipBlockIdx_y + hipThreadIdx_y;

	unsigned int localIdx = hipThreadIdx_x;
	unsigned int localIdy = hipThreadIdx_y;
	
    /* copy from input to local memory */
	block[localIdy * blockSize + localIdx] = input[globalIdy*width + globalIdx];

    /* wait until the whole block is filled */
	__syncthreads();

    /* calculate the corresponding raster indices of source and target */
	unsigned int sourceIndex = localIdy * blockSize + localIdx;
	unsigned int targetIndex = globalIdy + globalIdx * height; 
	
	output[targetIndex] = block[sourceIndex];
}




/*  Recursive Gaussian filter
 *  parameters:	
 *      input - pointer to input data 
 *      output - pointer to output data 
 *      width  - image width
 *      iheight  - image height
 *      a0-a3, b1, b2, coefp, coefn - gaussian parameters
 */
__global__ void RecursiveGaussian_kernel(
                                       const uchar4* input, uchar4* output, 
				       const int width, const int height, 
				       const float a0, const float a1, 
				       const float a2, const float a3, 
				       const float b1, const float b2, 
				       const float coefp, const float coefn)
{
    // compute x : current column ( kernel executes on 1 column )
    unsigned int x = hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x;

    if (x >= width) 
	return;

    // start forward filter pass
    float4 xp = make_float4(0.0f,0.0f,0.0f,0.0f);  // previous input
    float4 yp = make_float4(0.0f,0.0f,0.0f,0.0f);  // previous output
    float4 yb = make_float4(0.0f,0.0f,0.0f,0.0f);  // previous output by 2

    for (int y = 0; y < height; y++) 
    {
	  int pos = x + y * width;
        float4 xc = make_float4(input[pos].x, input[pos].y, input[pos].z, input[pos].w);
        float4 yc ;
        yc.x = (a0 * xc.x) + (a1 * xp.x) - (b1 * yp.x) - (b2 * yb.x);
        yc.y = (a0 * xc.y) + (a1 * xp.y) - (b1 * yp.y) - (b2 * yb.y);
        yc.z = (a0 * xc.z) + (a1 * xp.z) - (b1 * yp.z) - (b2 * yb.z);
        yc.w = (a0 * xc.w) + (a1 * xp.w) - (b1 * yp.w) - (b2 * yb.w);

	output[pos] = make_uchar4(yc.x, yc.y, yc.z, yc.w);
        xp = xc; 
        yb = yp; 
        yp = yc; 

    }

    __syncthreads();


    // start reverse filter pass: ensures response is symmetrical
    float4 xn = make_float4(0.0f,0.0f,0.0f,0.0f);
    float4 xa = make_float4(0.0f,0.0f,0.0f,0.0f);
    float4 yn = make_float4(0.0f,0.0f,0.0f,0.0f);
    float4 ya = make_float4(0.0f,0.0f,0.0f,0.0f);


    for (int y = height - 1; y > -1; y--) 
    {
        int pos = x + y * width;
        float4 xc =  make_float4(input[pos].x, input[pos].y, input[pos].z, input[pos].w);
        float4 yc ;
        yc.x = (a2 * xn.x) + (a3 * xa.x) - (b1 * yn.x) - (b2 * ya.x);
        yc.y = (a2 * xn.y) + (a3 * xa.y) - (b1 * yn.y) - (b2 * ya.y);
        yc.z = (a2 * xn.z) + (a3 * xa.z) - (b1 * yn.z) - (b2 * ya.z);
        yc.w = (a2 * xn.w) + (a3 * xa.w) - (b1 * yn.w) - (b2 * ya.w);

        xa = xn; 
        xn = xc; 
        ya = yn; 
        yn = yc;
	float4 temp;
        temp.x = output[pos].x + yc.x;
        temp.y = output[pos].y + yc.y;
        temp.z = output[pos].z + yc.z;
        temp.w = output[pos].w + yc.w;

	output[pos] = make_uchar4(temp.x, temp.y, temp.z, temp.w);
    }
}




int
RecursiveGaussian::readInputImage(std::string inputImageName)
{
    // load input bitmap image
    inputBitmap.load(inputImageName.c_str());

    // error if image did not load
    if(!inputBitmap.isLoaded())
    {
        std::cout << "Failed to load input image!";
        return SDK_FAILURE;
    }

    // get width and height of input image
    height = inputBitmap.getHeight();
    width = inputBitmap.getWidth();
    printf("width = %d, height = %d", width, height);
    // Check width against blockSizeX
    if(width % GROUP_SIZE || height % GROUP_SIZE)
    {
        char err[2048];
        sprintf(err, "Width should be a multiple of %d \n", GROUP_SIZE);
        std::cout << err;
        return SDK_FAILURE;
    }

    // allocate memory for input & output image data
    inputImageData  = (uchar4*)malloc(width * height * sizeof(uchar4));
    CHECK_ALLOCATION(inputImageData, "Failed to allocate memory! (inputImageData)");
    verificationInput = (uchar4*)malloc(width * height * sizeof(uchar4));
    CHECK_ALLOCATION(verificationInput,
                     "Failed to allocate memory! (verificationInput)");

    // allocate memory for output image data
    outputImageData = (uchar4*)malloc(width * height * sizeof(uchar4));
    CHECK_ALLOCATION(outputImageData,
                     "Failed to allocate memory! (outputImageData)");

    // initialize the Image data to NULL
    memset(outputImageData, 0, width * height * sizeof(uchar4));

    // get the pointer to pixel data
    pixelData = inputBitmap.getPixels();
    if(pixelData == NULL)
    {
        std::cout << "Failed to read pixel Data!";
        return SDK_FAILURE;
    }

    // Copy pixel data into inputImageData
    memcpy(inputImageData, pixelData, width * height * sizeof(uchar4));
    memcpy(verificationInput, pixelData, width * height * sizeof(uchar4));

    // allocate memory for verification output
    verificationOutput = (uchar4*)malloc(width * height * sizeof(uchar4));
    CHECK_ALLOCATION(verificationOutput,
                     "Failed to allocate memory! (verificationOutput)");

    // initialize the data to NULL
    memset(verificationOutput, 0, width * height * sizeof(uchar4));

    return SDK_SUCCESS;

}


int
RecursiveGaussian::writeOutputImage(std::string outputImageName)
{
    // copy output image data back to original pixel data
    memcpy(pixelData, outputImageData, width * height * sizeof(uchar4));

    // write the output bmp file
    if(!inputBitmap.write(outputImageName.c_str()))
    {
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}


void
RecursiveGaussian::computeGaussParms(float fSigma, int iOrder, GaussParms* pGP)
{
    // pre-compute filter coefficients
    pGP->nsigma =
        fSigma; // note: fSigma is range-checked and clamped >= 0.1f upstream
    pGP->alpha = 1.695f / pGP->nsigma;
    pGP->ema = exp(-pGP->alpha);
    pGP->ema2 = exp(-2.0f * pGP->alpha);
    pGP->b1 = -2.0f * pGP->ema;
    pGP->b2 = pGP->ema2;
    pGP->a0 = 0.0f;
    pGP->a1 = 0.0f;
    pGP->a2 = 0.0f;
    pGP->a3 = 0.0f;
    pGP->coefp = 0.0f;
    pGP->coefn = 0.0f;

    switch (iOrder)
    {
    case 0:
    {
        const float k = (1.0f - pGP->ema)*(1.0f - pGP->ema)/(1.0f +
                        (2.0f * pGP->alpha * pGP->ema) - pGP->ema2);
        pGP->a0 = k;
        pGP->a1 = k * (pGP->alpha - 1.0f) * pGP->ema;
        pGP->a2 = k * (pGP->alpha + 1.0f) * pGP->ema;
        pGP->a3 = -k * pGP->ema2;
    }
    break;
    case 1:
    {
        pGP->a0 = (1.0f - pGP->ema) * (1.0f - pGP->ema);
        pGP->a1 = 0.0f;
        pGP->a2 = -pGP->a0;
        pGP->a3 = 0.0f;
    }
    break;
    case 2:
    {
        const float ea = exp(-pGP->alpha);
        const float k = -(pGP->ema2 - 1.0f)/(2.0f * pGP->alpha * pGP->ema);
        float kn = -2.0f * (-1.0f + (3.0f * ea) - (3.0f * ea * ea) + (ea * ea * ea));
        kn /= (((3.0f * ea) + 1.0f + (3.0f * ea * ea) + (ea * ea * ea)));
        pGP->a0 = kn;
        pGP->a1 = -kn * (1.0f + (k * pGP->alpha)) * pGP->ema;
        pGP->a2 = kn * (1.0f - (k * pGP->alpha)) * pGP->ema;
        pGP->a3 = -kn * pGP->ema2;
    }
    break;
    default:
        // note: iOrder is range-checked and clamped to 0-2 upstream
        return;
    }
    pGP->coefp = (pGP->a0 + pGP->a1)/(1.0f + pGP->b1 + pGP->b2);
    pGP->coefn = (pGP->a2 + pGP->a3)/(1.0f + pGP->b1 + pGP->b2);
}

int
RecursiveGaussian::setupHIP()
{

    hipMalloc((void**)&inputImageBuffer,width * height * pixelSize);
    hipMalloc((void**)&outputImageBuffer,width * height * pixelSize);
    hipMalloc((void**)&tempImageBuffer,width * height * pixelSize);

    blockSize = 16;

    return SDK_SUCCESS;
}

int
RecursiveGaussian::runKernels()
{
    // initialize Gaussian parameters
    float fSigma = 10.0f;               // filter sigma (blur factor)
    int iOrder = 0;                     // filter order

    // compute gaussian parameters
    computeGaussParms(fSigma, iOrder, &oclGP);

    hipMemcpy(inputImageBuffer, inputImageData,width * height * pixelSize, hipMemcpyHostToDevice);

    hipEvent_t start, stop;

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;

    hipEventRecord(start, NULL);

    hipLaunchKernelGGL(RecursiveGaussian_kernel,
                    dim3(width/blockSizeX, 1/blockSizeY),
                    dim3(blockSizeX,blockSizeY),
                    0, 0,
                    inputImageBuffer ,tempImageBuffer,width, height, oclGP.a0, oclGP.a1,oclGP.a2,oclGP.a3,oclGP.b1, oclGP.b2, oclGP.coefp,oclGP.coefn);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);
    printf ("kernel_time (hipEventElapsedTime) =%6.3fms\n", eventMs);

    hipEventRecord(start, NULL);

    hipLaunchKernelGGL(transpose_kernel,
                    dim3(width/blockSize,height/blockSize),
                    dim3(blockSize,blockSize),
                    0, 0,
                    inputImageBuffer ,tempImageBuffer,width, height, blockSize);
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);
    printf ("kernel_time (hipEventElapsedTime) =%6.3fms\n", eventMs);


    /* Set Arguments for Recursive Gaussian Kernel
    Image is now transposed
    new_width = height
    new_height = width */
    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(RecursiveGaussian_kernel,
                    dim3(height/blockSizeX, 1/blockSizeY),
                    dim3(blockSizeX,blockSizeY),
                    0, 0,
                    inputImageBuffer ,tempImageBuffer, height,width, oclGP.a0, oclGP.a1,oclGP.a2,oclGP.a3,oclGP.b1, oclGP.b2, oclGP.coefp,oclGP.coefn);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);
    printf ("kernel_time (hipEventElapsedTime) =%6.3fms\n", eventMs);

    hipEventRecord(start, NULL);
    // Enqueue final Transpose Kernel
    hipLaunchKernelGGL(transpose_kernel,
                    dim3(height/blockSize,width/blockSize),
                    dim3(blockSize,blockSize),
                    0, 0,
                    outputImageBuffer ,tempImageBuffer, height, width, blockSize);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);
    printf ("kernel_time (hipEventElapsedTime) =%6.3fms\n", eventMs);

    hipMemcpy(outputImageData, outputImageBuffer ,width * height * sizeof(uchar4), hipMemcpyDeviceToHost);

    return SDK_SUCCESS;
}



int
RecursiveGaussian::initialize()
{
    // Call base class Initialize to get default configuration
    if (sampleArgs->initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    Option* iteration_option = new Option;
    if(!iteration_option)
    {
        return SDK_FAILURE;
    }
    iteration_option->_sVersion = "i";
    iteration_option->_lVersion = "iterations";
    iteration_option->_description = "Number of iterations to execute kernel";
    iteration_option->_type = CA_ARG_INT;
    iteration_option->_value = &iterations;

    sampleArgs->AddOption(iteration_option);
    delete iteration_option;

    return SDK_SUCCESS;
}

int
RecursiveGaussian::setup()
{
    // Allocate host memory and read input image
    std::string filePath = getPath() + std::string(INPUT_IMAGE);
    std::cout << "Searching for input Image at following location : " <<
              filePath << std::endl;
    int status = readInputImage(filePath);
    CHECK_ERROR(status, SDK_SUCCESS, "Read Input Image Failed");

    // create and initialize timers
    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    if (setupHIP() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    sampleTimer->stopTimer(timer);
    // Compute setup time
    setupTime = (double)(sampleTimer->readTimer(timer));

    return SDK_SUCCESS;

}

int
RecursiveGaussian::run()
{
    int status = 0;
    for(int i = 0; i < 2 && iterations != 1; i++)
    {
        // Set kernel arguments and run kernel
        if (runKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    // create and initialize timers
    std::cout << "Executing kernel for " <<
              iterations << " iterations" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    for(int i = 0; i < iterations; i++)
    {
        // Set kernel arguments and run kernel
        if (runKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

    }

    sampleTimer->stopTimer(timer);
    // Compute kernel time
    kernelTime = (double)(sampleTimer->readTimer(timer)) / iterations;

    // write the output image to bitmap file
    std::string filePath = std::string(OUTPUT_IMAGE);
    status = writeOutputImage(filePath);
    CHECK_ERROR(status, SDK_SUCCESS, "Write Output Image Failed");

    return SDK_SUCCESS;
}

int
RecursiveGaussian::cleanup()
{
    hipFree(inputImageBuffer);
    hipFree(outputImageBuffer);
    hipFree(tempImageBuffer);

    FREE(inputImageData);
    FREE(outputImageData);
    FREE(verificationInput);
    FREE(verificationOutput);

    return SDK_SUCCESS;
}

void
RecursiveGaussian::recursiveGaussianCPU(uchar4* input, uchar4* output,
                                        const int width, const int height,
                                        const float a0, const float a1,
                                        const float a2, const float a3,
                                        const float b1, const float b2,
                                        const float coefp, const float coefn)
{

    // outer loop over all columns within image
    for (int X = 0; X < width; X++)
    {
        // start forward filter pass
        float xp[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // previous input
        float yp[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // previous output
        float yb[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // previous output by 2

        float xc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float yc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int Y = 0; Y < height; Y++)
        {
            // output position to write
            int pos = Y * width + X;

            // convert input element to float4
            xc[0] = input[pos].x;
            xc[1] = input[pos].y;
            xc[2] = input[pos].z;
            xc[3] = input[pos].w;

            yc[0] = (a0 * xc[0]) + (a1 * xp[0]) - (b1 * yp[0]) - (b2 * yb[0]);
            yc[1] = (a0 * xc[1]) + (a1 * xp[1]) - (b1 * yp[1]) - (b2 * yb[1]);
            yc[2] = (a0 * xc[2]) + (a1 * xp[2]) - (b1 * yp[2]) - (b2 * yb[2]);
            yc[3] = (a0 * xc[3]) + (a1 * xp[3]) - (b1 * yp[3]) - (b2 * yb[3]);

            // convert float4 element to output
            output[pos].x = (unsigned char)yc[0];
            output[pos].y = (unsigned char)yc[1];
            output[pos].z = (unsigned char)yc[2];
            output[pos].w = (unsigned char)yc[3];

            for (int i = 0; i < 4; i++)
            {
                xp[i] = xc[i];
                yb[i] = yp[i];
                yp[i] = yc[i];
            }
        }

        // start reverse filter pass: ensures response is symmetrical
        float xn[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float xa[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float yn[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float ya[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        float fTemp[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int Y = height - 1; Y > -1; Y--)
        {
            int pos = Y * width + X;

            // convert uchar4 to float4
            xc[0] = input[pos].x;
            xc[1] = input[pos].y;
            xc[2] = input[pos].z;
            xc[3] = input[pos].w;

            yc[0] = (a2 * xn[0]) + (a3 * xa[0]) - (b1 * yn[0]) - (b2 * ya[0]);
            yc[1] = (a2 * xn[1]) + (a3 * xa[1]) - (b1 * yn[1]) - (b2 * ya[1]);
            yc[2] = (a2 * xn[2]) + (a3 * xa[2]) - (b1 * yn[2]) - (b2 * ya[2]);
            yc[3] = (a2 * xn[3]) + (a3 * xa[3]) - (b1 * yn[3]) - (b2 * ya[3]);

            for (int i = 0; i< 4; i++)
            {
                xa[i] = xn[i];
                xn[i] = xc[i];
                ya[i] = yn[i];
                yn[i] = yc[i];
            }

            // convert uhcar4 to float4
            fTemp[0] = output[pos].x;
            fTemp[1] = output[pos].y;
            fTemp[2] = output[pos].z;
            fTemp[3] = output[pos].w;

            fTemp[0] += yc[0];
            fTemp[1] += yc[1];
            fTemp[2] += yc[2];
            fTemp[3] += yc[3];

            // convert float4 to uchar4
            output[pos].x = (unsigned char)fTemp[0];
            output[pos].y = (unsigned char)fTemp[1];
            output[pos].z = (unsigned char)fTemp[2];
            output[pos].w = (unsigned char)fTemp[3];
        }
    }

}

void
RecursiveGaussian::transposeCPU(uchar4* input,
                                uchar4* output,
                                const int width,
                                const int height)
{
    // transpose matrix
    for(int Y = 0; Y < height; Y++)
    {
        for(int X = 0; X < width; X++)
        {
            output[Y + X * height] = input[X + Y * width];
        }
    }
}

void
RecursiveGaussian::recursiveGaussianCPUReference()
{

    // Create a temp uchar4 array
    uchar4* temp = (uchar4*)malloc(width * height * sizeof(uchar4));
    if(temp == NULL)
    {
        printf("Failed to allocate host memory! (temp)");
        return;
    }

    // Call recursive Gaussian CPU
    recursiveGaussianCPU(verificationInput, temp, width, height,
                         oclGP.a0, oclGP.a1, oclGP.a2, oclGP.a3,
                         oclGP.b1, oclGP.b2, oclGP.coefp, oclGP.coefn);

    // Transpose the temp buffer
    transposeCPU(temp, verificationOutput, width, height);

    // again Call recursive Gaussian CPU
    recursiveGaussianCPU(verificationOutput, temp, height, width,
                         oclGP.a0, oclGP.a1, oclGP.a2, oclGP.a3,
                         oclGP.b1, oclGP.b2, oclGP.coefp, oclGP.coefn);

    // Do a final Transpose
    transposeCPU(temp, verificationOutput, height, width);

    if(temp)
    {
        free(temp);
    }

}

int
RecursiveGaussian::verifyResults()
{

    if(sampleArgs->verify)
    {
        recursiveGaussianCPUReference();

        float *outputDevice = new float[width * height * 4];
        CHECK_ALLOCATION(outputDevice,
                         "Failed to allocate host" "memory! (outputDevice)");

        float *outputReference = new float[width * height * 4];
        CHECK_ALLOCATION(outputReference,
                         "Failed to allocate host" "memory! (outputReference)");

        // copy uchar4 data to float array
        for(int i=0; i < (int)(width * height); i++)
        {
            outputDevice[4 * i + 0] = outputImageData[i].x;
            outputDevice[4 * i + 1] = outputImageData[i].y;
            outputDevice[4 * i + 2] = outputImageData[i].z;
            outputDevice[4 * i + 3] = outputImageData[i].w;

            outputReference[4 * i + 0] = verificationOutput[i].x;
            outputReference[4 * i + 1] = verificationOutput[i].y;
            outputReference[4 * i + 2] = verificationOutput[i].z;
            outputReference[4 * i + 3] = verificationOutput[i].w;
        }

        // compare the results and see if they match
        if(compare(outputReference,
                   outputDevice,
                   width * height,
                   (float)0.0001))
        {
            std::cout <<"Passed!\n" << std::endl;
            delete[] outputDevice;
            delete[] outputReference;
            return SDK_SUCCESS;
        }
        else
        {
            std::cout << "Failed\n" << std::endl;
            delete[] outputDevice;
            delete[] outputReference;
            return SDK_FAILURE;
        }
    }

    return SDK_SUCCESS;
}

void
RecursiveGaussian::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[4] =
        {
            "Width",
            "Height",
            "Time(sec)",
            "[Transfer+Kernel]Time(sec)"
        };
        std::string stats[4];

        sampleTimer->totalTime = setupTime + kernelTime;

        stats[0]  = toString(width, std::dec);
        stats[1]  = toString(height, std::dec);
        stats[2]  = toString(sampleTimer->totalTime, std::dec);
        stats[3]  = toString(kernelTime, std::dec);

        printStatistics(strArray, stats, 4);
    }
}


int
main(int argc, char * argv[])
{

    RecursiveGaussian hipRecursiveGaussian;

    if (hipRecursiveGaussian.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (hipRecursiveGaussian.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }


    if (hipRecursiveGaussian.setup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (hipRecursiveGaussian.run() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (hipRecursiveGaussian.verifyResults() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (hipRecursiveGaussian.cleanup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    hipRecursiveGaussian.printStats();

    return SDK_SUCCESS;
}
