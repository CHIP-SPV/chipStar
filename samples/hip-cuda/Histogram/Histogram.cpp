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

#include "Histogram.hpp"

#include <math.h>

#define LINEAR_MEM_ACCESS


/**
 * @brief   Calculates block-histogram bin whose bin size is 256
 * @param   data  input data pointer
 * @param   sharedArray shared array for thread-histogram bins
 * @param   binResult block-histogram array
 */

__global__ void compute_histogram(const size_t num_elements,
                                  const unsigned *data,
                                  unsigned *histogram) {

  size_t thread = hipThreadIdx_x;
  size_t block = hipBlockDim_x;

  __shared__ unsigned local[BIN_SIZE];

  for (size_t i = thread; i < BIN_SIZE; i += block)
    local[i] = 0;

  __syncthreads();

  for (size_t idx = (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;
       idx < num_elements; idx += hipGridDim_x * hipBlockDim_x) {

    size_t bucket = data[idx] % BIN_SIZE;
    atomicAdd(&local[bucket], 1);
  }

  __syncthreads();

  for (size_t i = thread; i < BIN_SIZE; i += block)
    atomicAdd(&histogram[i], local[i]);
}



int
Histogram::calculateHostBin()
{
    for(int i = 0; i < height; ++i)
    {
        for(int j = 0; j < width; ++j)
        {
            hostBin[data[i * width + j]]++;
        }
    }

    return SDK_SUCCESS;
}


int
Histogram::setupHistogram()
{
    int i = 0;

    data = (unsigned int *)malloc(sizeof(unsigned int) * width * height);

    for(i = 0; i < width * height; i++)
    {
        data[i] = rand() % (unsigned int)(binSize);
    }

    hostBin = (unsigned int*)malloc(binSize * sizeof(unsigned int));
    CHECK_ALLOCATION(hostBin, "Failed to allocate host memory. (hostBin)");
    memset(hostBin, 0, binSize * sizeof(unsigned int));

    deviceBin = (unsigned int*)malloc(binSize * sizeof(unsigned int));
    CHECK_ALLOCATION(deviceBin, "Failed to allocate host memory. (deviceBin)");
    memset(deviceBin, 0, binSize * sizeof(unsigned int));

    return SDK_SUCCESS;
}

int
Histogram::setupHIP(void)
{
    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    cout << " System minor " << devProp.minor << endl;
    cout << " System major " << devProp.major << endl;
    cout << " agent prop name " << devProp.name << endl;

    return SDK_SUCCESS;
}



int
Histogram::runKernels(void)
{

    hipEvent_t start, stop;

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;

    globalThreads = (width * height) / (GROUP_ITERATIONS);

    hipMalloc((void**)&dataBuf,sizeof(unsigned int) * width * height);
    hipMalloc((void**)&deviceBinBuf,sizeof(unsigned int) * binSize);
    hipMemcpy(dataBuf, data, sizeof(unsigned int) * width * height, hipMemcpyHostToDevice);
    hipMemset(deviceBinBuf, 0, sizeof(unsigned int) * binSize);

    hipEventRecord(start, NULL);

    hipLaunchKernelGGL(compute_histogram,
                    dim3(globalThreads/GROUP_SIZE),
                    dim3(GROUP_SIZE),
                    groupSize * binSize * sizeof(unsigned int), 0,
                    (width * height), dataBuf, deviceBinBuf);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);
    printf ("kernel_time (hipEventElapsedTime) =%6.3f ms\n", eventMs);

    hipMemcpy(deviceBin, deviceBinBuf, sizeof(unsigned int) * binSize, hipMemcpyDeviceToHost);

    return SDK_SUCCESS;
}

int
Histogram::initialize()
{
    // Call base class Initialize to get default configuration
    if(sampleArgs->initialize())
    {
        return SDK_FAILURE;
    }

    Option* width_option = new Option;
    CHECK_ALLOCATION(width_option, "Memory allocation error.\n");

    width_option->_sVersion = "x";
    width_option->_lVersion = "width";
    width_option->_description = "Width of the input";
    width_option->_type = CA_ARG_INT;
    width_option->_value = &width;

    sampleArgs->AddOption(width_option);
    delete width_option;

    Option* height_option = new Option;
    CHECK_ALLOCATION(height_option, "Memory allocation error.\n");

    height_option->_sVersion = "y";
    height_option->_lVersion = "height";
    height_option->_description = "Height of the input";
    height_option->_type = CA_ARG_INT;
    height_option->_value = &height;

    sampleArgs->AddOption(height_option);
    delete height_option;

    Option* iteration_option = new Option;
    CHECK_ALLOCATION(iteration_option, "Memory allocation error.\n");

    iteration_option->_sVersion = "i";
    iteration_option->_lVersion = "iterations";
    iteration_option->_description = "Number of iterations to execute kernel";
    iteration_option->_type = CA_ARG_INT;
    iteration_option->_value = &iterations;

    sampleArgs->AddOption(iteration_option);
    delete iteration_option;

    Option* scalar_option = new Option;
    CHECK_ALLOCATION(scalar_option, "Memory allocation error.\n");

    scalar_option->_sVersion = "";
    scalar_option->_lVersion = "scalar";
    scalar_option->_description =
        "Run scalar version of the kernel (--scalar and --vector options are mutually exclusive)";
    scalar_option->_type = CA_NO_ARGUMENT;
    scalar_option->_value = &scalar;

    sampleArgs->AddOption(scalar_option);
    delete scalar_option;

    Option* vector_option = new Option;
    CHECK_ALLOCATION(vector_option, "Memory allocation error.\n");

    vector_option->_sVersion = "";
    vector_option->_lVersion = "vector";
    vector_option->_description =
        "Run vector version of the kernel (--scalar and --vector options are mutually exclusive)";
    vector_option->_type = CA_NO_ARGUMENT;
    vector_option->_value = &vector;

    sampleArgs->AddOption(vector_option);
    delete vector_option;



    return SDK_SUCCESS;
}

int
Histogram::setup()
{
    if(iterations < 1)
    {
        std::cout<<"Error, iterations cannot be 0 or negative. Exiting..\n";
        exit(0);
    }
    int status = 0;

    /* width must be multiples of binSize and
     * height must be multiples of groupSize
     */
//    width = (width / binSize ? width / binSize: 1) * binSize;
//    height = (height / groupSize ? height / groupSize: 1) * groupSize;

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    status = setupHIP();
    if(status != SDK_SUCCESS)
    {
        return status;
    }

    status = setupHistogram();
    CHECK_ERROR(status, SDK_SUCCESS, "Sample Resource Setup Failed");

    sampleTimer->stopTimer(timer);

    // Compute setup time
    setupTime = (double)(sampleTimer->readTimer(timer));

    return SDK_SUCCESS;
}


int
Histogram::run()
{

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
    // Compute average kernel time
    kernelTime = (double)(sampleTimer->readTimer(timer));

    if(!sampleArgs->quiet)
    {
        printArray<unsigned int>("deviceBin", deviceBin, binSize, 1);
    }

    return SDK_SUCCESS;
}

int
Histogram::verifyResults()
{
    if(sampleArgs->verify)
    {
        /**
         * Reference implementation on host device
         * calculates the histogram bin on host
         */
        calculateHostBin();
        printArray<unsigned int>("hostBin", hostBin, binSize, 1);
        // compare the results and see if they match
        bool result = true;
        for(int i = 0; i < binSize; ++i)
        {
            if(hostBin[i] != deviceBin[i])
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

void Histogram::printStats()
{
    if(sampleArgs->timing)
    {
        // calculate total time
        double avgKernelTime = kernelTime/iterations;

        std::string strArray[5] =
        {
            "Width",
            "Height",
            "Setup Time(sec)",
            "Avg. Kernel Time (sec)",
            "Elements/sec"
        };
        std::string stats[5];

        stats[0] = toString(width, std::dec);
        stats[1] = toString(height, std::dec);
        stats[2] = toString(setupTime, std::dec);
        stats[3] = toString(avgKernelTime, std::dec);
        stats[4] = toString(((width*height)/avgKernelTime), std::dec);

        printStatistics(strArray, stats, 5);
    }
}

int Histogram::cleanup()
{

    // Releases HIP resources (Memory)


    hipFree(dataBuf);
    hipFree(deviceBinBuf);

    // Release program resources (input memory etc.)
    FREE(hostBin);
    FREE(deviceBin);

    return SDK_SUCCESS;
}

int
main(int argc, char * argv[])
{
    int status = 0;
    // Create MonteCalroAsian object
    Histogram hipHistogram;

    // Initialization
    if(hipHistogram.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Parse command line options
    if(hipHistogram.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Setup
    status = hipHistogram.setup();
    if(status != SDK_SUCCESS)
    {
        return (status == SDK_EXPECTED_FAILURE)? SDK_SUCCESS : SDK_FAILURE;
    }

    // Run
    if(hipHistogram.run() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Verify
    if(hipHistogram.verifyResults() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Cleanup resources created
    if(hipHistogram.cleanup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Print performance statistics
    hipHistogram.printStats();

    return SDK_SUCCESS;
}
