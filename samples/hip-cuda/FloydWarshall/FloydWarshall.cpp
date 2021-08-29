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

#define SAMPLE_VERSION "HIP-Examples-Application-v1.0"

using namespace appsdk;
using namespace std;

/**
 * FloydWarshall
 * Class implements OpenFloydWarshall Pathfinding sample
 */

/*
 * MACROS
 * By default, MAXDISTANCE between two nodes
 */
#define MAXDISTANCE    (200)

class FloydWarshall
{
        unsigned int                       seed;  /**< Seed value for random number generation */
        double                        setupTime;  /**< Time for setting up Open*/
        double                  totalKernelTime;  /**< Time for kernel execution */
        double                 totalProgramTime;  /**< Time for program execution */
        double              referenceKernelTime;  /**< Time for reference implementation */
        unsigned int                   numNodes;  /**< Number of nodes in the graph */
        unsigned int        *pathDistanceMatrix;  /**< path distance array */
        unsigned int                *pathMatrix;  /**< path arry */
        unsigned int
                *verificationPathDistanceMatrix;  /**< path distance array for reference implementation */
        unsigned int    *verificationPathMatrix;  /**< path array for reference implementation */
        unsigned int*        pathDistanceBuffer;  /**< path distance memory buffer */
        unsigned int*                pathBuffer;  /**< path memory buffer */
        int                          iterations;  /**< Number of iterations to execute kernel */
        unsigned int                  blockSize;  /**< use local memory of size blockSize x blockSize */

        SDKTimer    *sampleTimer;      /**< SDKTimer object */

    public:

        HIPCommandArgs   *sampleArgs;   /**< HIPCommand argument class */
        /**
         * Constructor
         * Initialize member variables
         * @param name name of sample (string)
         */
        FloydWarshall()
        {
            seed = 123;
            numNodes = 256;
            pathDistanceMatrix = NULL;
            pathMatrix = NULL;
            verificationPathDistanceMatrix = NULL;
            verificationPathMatrix         = NULL;
            setupTime = 0;
            totalKernelTime = 0;
            iterations = 1;
            blockSize = 16;
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
        int setupFloydWarshall();


        int setupHIP();


        int runKernels();

        /**
         * Returns the lesser of the two unsigned integers a and b
         */
        unsigned int minimum(unsigned int a, unsigned int b);

        /**
         * Reference CPU implementation of FloydWarshall PathFinding
         * for performance comparison
         * @param pathDistanceMatrix Distance between nodes of a graph
         * @param intermediate node between two nodes of a graph
         * @param number of nodes in the graph
         */
        void floydWarshallCPUReference(unsigned int * pathDistanceMatrix,
                                       unsigned int * pathMatrix, unsigned int numNodes);

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
         * Run OpenFloydWarshall Path finding
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



/*!
 * returns the lesser of the two integers a and b
 */
inline __device__
unsigned int uintMin(unsigned int a, unsigned int b)
{
    return (b < a) ? b : a;
}

/*!
 * The floyd Warshall algorithm is a multipass algorithm
 * that calculates the shortest path between each pair of
 * nodes represented by pathDistanceBuffer.
 *
 * In each pass a node k is introduced and the pathDistanceBuffer
 * which has the shortest distance between each pair of nodes
 * considering the (k-1) nodes (that are introduced in the previous
 * passes) is updated such that
 *
 * ShortestPath(x,y,k) = min(ShortestPath(x,y,k-1), ShortestPath(x,k,k-1) + ShortestPath(k,y,k-1))
 * where x and y are the pair of nodes between which the shortest distance
 * is being calculated.
 *
 * pathBuffer stores the intermediate nodes through which the shortest
 * path goes for each pair of nodes.
 *
 * numNodes is the number of nodes in the graph.
 *
 * for more detailed explaination of the algorithm kindly refer to the document
 * provided with the sample
 */

__global__ void floydWarshallPass(
		                  unsigned int * pathDistanceBuffer,
                                  unsigned int * pathBuffer        ,
                                  const unsigned int numNodes      ,
                                  const unsigned int pass)
{
    int xValue = hipThreadIdx_x+hipBlockIdx_x*hipBlockDim_x;
    int yValue = hipThreadIdx_y+hipBlockIdx_y*hipBlockDim_y;

    int k = pass;
    int oldWeight = pathDistanceBuffer[yValue * numNodes + xValue];
    int tempWeight = (pathDistanceBuffer[yValue * numNodes + k] + pathDistanceBuffer[k * numNodes + xValue]);

    if (tempWeight < oldWeight)
    {
        pathDistanceBuffer[yValue * numNodes + xValue] = tempWeight;
        pathBuffer[yValue * numNodes + xValue] = k;
    }
}


int FloydWarshall::setupFloydWarshall()
{
    unsigned int matrixSizeBytes;

    // allocate and init memory used by host
    matrixSizeBytes = numNodes * numNodes * sizeof(unsigned int);
    pathDistanceMatrix = (unsigned int *) malloc(matrixSizeBytes);
    CHECK_ALLOCATION(pathDistanceMatrix,
                     "Failed to allocate host memory. (pathDistanceMatrix)");

    pathMatrix = (unsigned int *) malloc(matrixSizeBytes);
    CHECK_ALLOCATION(pathMatrix, "Failed to allocate host memory. (pathMatrix)");

    // random initialisation of input

    /*
     * pathMatrix is the intermediate node from which the path passes
     * pathMatrix(i,j) = k means the shortest path from i to j
     * passes through an intermediate node k
     * Initialized such that pathMatrix(i,j) = i
     */

    fillRandom<unsigned int>(pathDistanceMatrix, numNodes, numNodes, 0, MAXDISTANCE);
    for(int i = 0; i < numNodes; ++i)
    {
        unsigned int iXWidth = i * numNodes;
        pathDistanceMatrix[iXWidth + i] = 0;
    }

    /*
     * pathMatrix is the intermediate node from which the path passes
     * pathMatrix(i,j) = k means the shortest path from i to j
     * passes through an intermediate node k
     * Initialized such that pathMatrix(i,j) = i
     */
    for(int i = 0; i < numNodes; ++i)
    {
        for(int j = 0; j < i; ++j)
        {
            pathMatrix[i * numNodes + j] = i;
            pathMatrix[j * numNodes + i] = j;
        }
        pathMatrix[i * numNodes + i] = i;
    }

    /*
     * Unless sampleArgs->quiet mode has been enabled, print the INPUT array.
     */
    if(!sampleArgs->quiet)
    {
        printArray<unsigned int>(
            "Path Distance",
            pathDistanceMatrix,
            numNodes,
            1);

        printArray<unsigned int>(
            "Path ",
            pathMatrix,
            numNodes,
            1);
    }

    if(sampleArgs->verify)
    {
        verificationPathDistanceMatrix = (unsigned int *) malloc(numNodes * numNodes *
                                         sizeof(int));
        CHECK_ALLOCATION(verificationPathDistanceMatrix,
                         "Failed to allocate host memory. (verificationPathDistanceMatrix)");

        verificationPathMatrix = (unsigned int *) malloc(numNodes * numNodes * sizeof(
                                     int));
        CHECK_ALLOCATION(verificationPathMatrix,
                         "Failed to allocate host memory. (verificationPathMatrix)");

        memcpy(verificationPathDistanceMatrix, pathDistanceMatrix,
               numNodes * numNodes * sizeof(int));
        memcpy(verificationPathMatrix, pathMatrix, numNodes*numNodes*sizeof(int));
    }

    return SDK_SUCCESS;
}

int
FloydWarshall::setupHIP(void)
{
    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    cout << " System minor " << devProp.minor << endl;
    cout << " System major " << devProp.major << endl;
    cout << " agent prop name " << devProp.name << endl;

    return SDK_SUCCESS;
}

int
FloydWarshall::runKernels(void)
{

    unsigned int numPasses = numNodes;
    unsigned int globalThreads[2] = {numNodes, numNodes};
    unsigned int localThreads[2] = {blockSize, blockSize};

    totalKernelTime = 0;


    if((unsigned int)(localThreads[0] * localThreads[0]) >256)
    {

        blockSize = 4;

        localThreads[0] = blockSize;
        localThreads[1] = blockSize;
    }

    /*
    * The floyd Warshall algorithm is a multipass algorithm
    * that calculates the shortest path between each pair of
    * nodes represented by pathDistanceBuffer.
    *
    * In each pass a node k is introduced and the pathDistanceBuffer
    * which has the shortest distance between each pair of nodes
    * considering the (k-1) nodes (that are introduced in the previous
    * passes) is updated such that
    *
    * ShortestPath(x,y,k) = min(ShortestPath(x,y,k-1), ShortestPath(x,k,k-1) + ShortestPath(k,y,k-1))
    * where x and y are the pair of nodes between which the shortest distance
    * is being calculated.
    *
    * pathBuffer stores the intermediate nodes through which the shortest
    * path goes for each pair of nodes.
    */

    hipHostMalloc((void**)&pathDistanceBuffer, sizeof(unsigned int) * numNodes * numNodes,hipHostMallocDefault);
    hipHostMalloc((void**)&pathBuffer, sizeof(unsigned int) * numNodes * numNodes,hipHostMallocDefault);

    float *din, *di;

    hipHostGetDevicePointer((void**)&din, pathDistanceBuffer,0);
    hipHostGetDevicePointer((void**)&di, pathBuffer,0);

    hipMemcpy(din, pathDistanceMatrix, sizeof(unsigned int) * numNodes * numNodes,    hipMemcpyHostToDevice);

    hipEvent_t start, stop;

    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;

    for(unsigned int i = 0; i < numPasses; i += 1)
    {
    // Record the start event
    hipEventRecord(start, NULL);

    hipLaunchKernelGGL(floydWarshallPass,
                  dim3(globalThreads[0]/localThreads[0],globalThreads[1]/localThreads[1]),
                  dim3(localThreads[0],localThreads[1]),
                  0, 0,
                  pathDistanceBuffer,pathBuffer,numNodes ,i);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);

    printf ("kernel_time (hipEventElapsedTime) =%6.3fms\n", eventMs);
    }


    hipMemcpy(pathDistanceMatrix, din,numNodes * numNodes * sizeof(unsigned int), hipMemcpyDeviceToHost);
    hipMemcpy(pathMatrix, di,numNodes * numNodes * sizeof(unsigned int), hipMemcpyDeviceToHost);

    return SDK_SUCCESS;
}

/*
 * Returns the lesser of the two unsigned integers a and b
 */
unsigned int
FloydWarshall::minimum(unsigned int a, unsigned int b)
{
    return (b < a) ? b : a;
}

/*
 * Calculates the shortest path between each pair of nodes in a graph
 * pathDistanceMatrix gives the shortest distance between each node
 * in the graph.
 * pathMatrix gives the path intermediate node through which the shortest
 * distance in calculated
 * numNodes is the number of nodes in the graph
 */
void
FloydWarshall::floydWarshallCPUReference(unsigned int * pathDistanceMatrix,
        unsigned int * pathMatrix,
        const unsigned int numNodes)
{
    unsigned int distanceYtoX, distanceYtoK, distanceKtoX, indirectDistance;

    /*
     * pathDistanceMatrix is the adjacency matrix(square) with
     * the dimension equal to the number of nodes in the graph.
     */
    unsigned int width = numNodes;
    unsigned int yXwidth;

    /*
     * for each intermediate node k in the graph find the shortest distance between
     * the nodes i and j and update as
     *
     * ShortestPath(i,j,k) = min(ShortestPath(i,j,k-1), ShortestPath(i,k,k-1) + ShortestPath(k,j,k-1))
     */
    for(unsigned int k = 0; k < numNodes; ++k)
    {
        for(unsigned int y = 0; y < numNodes; ++y)
        {
            yXwidth =  y*numNodes;
            for(unsigned int x = 0; x < numNodes; ++x)
            {
                distanceYtoX = pathDistanceMatrix[yXwidth + x];
                distanceYtoK = pathDistanceMatrix[yXwidth + k];
                distanceKtoX = pathDistanceMatrix[k * width + x];

                indirectDistance = distanceYtoK + distanceKtoX;

                if(indirectDistance < distanceYtoX)
                {
                    pathDistanceMatrix[yXwidth + x] = indirectDistance;
                    pathMatrix[yXwidth + x]         = k;
                }
            }
        }
    }
}

int FloydWarshall::initialize()
{
    // Call base class Initialize to get default configuration
    if(sampleArgs->initialize())
    {
        return SDK_FAILURE;
    }

    Option* num_nodes = new Option;
    CHECK_ALLOCATION(num_nodes, "Memory allocation error.\n");

    num_nodes->_sVersion = "x";
    num_nodes->_lVersion = "nodes";
    num_nodes->_description = "number of nodes";
    num_nodes->_type = CA_ARG_INT;
    num_nodes->_value = &numNodes;
    sampleArgs->AddOption(num_nodes);
    delete num_nodes;

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

int FloydWarshall::setup()
{
    // numNodes should be multiples of blockSize
    if(numNodes % blockSize != 0)
    {
        numNodes = (numNodes / blockSize + 1) * blockSize;
    }

    if(setupFloydWarshall() != SDK_SUCCESS)
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

    setupTime = (double)sampleTimer->readTimer(timer);

    return SDK_SUCCESS;
}


int FloydWarshall::run()
{
    for(int i = 0; i < 2 && iterations != 1; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        if(runKernels() != SDK_SUCCESS)
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
        if(runKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    sampleTimer->stopTimer(timer);
    totalKernelTime = (double)(sampleTimer->readTimer(timer)) / iterations;

    if(!sampleArgs->quiet)
    {
        printArray<unsigned int>("Output Path Distance Matrix", pathDistanceMatrix, numNodes,
                            1);
        printArray<unsigned int>("Output Path Matrix", pathMatrix, numNodes, 1);
    }

    return SDK_SUCCESS;
}

int FloydWarshall::verifyResults()
{
    if(sampleArgs->verify)
    {
        /*
         * reference implementation
         * it overwrites the input array with the output
         */
        int refTimer = sampleTimer->createTimer();
        sampleTimer->resetTimer(refTimer);
        sampleTimer->startTimer(refTimer);
        floydWarshallCPUReference(verificationPathDistanceMatrix,
                                  verificationPathMatrix, numNodes);
        sampleTimer->stopTimer(refTimer);
        referenceKernelTime = sampleTimer->readTimer(refTimer);

        if(sampleArgs -> timing)
        {
            std::cout << "CPU time " << referenceKernelTime << std::endl;
        }

        // compare the results and see if they match
        if(memcmp(pathDistanceMatrix, verificationPathDistanceMatrix,
                  numNodes*numNodes*sizeof(unsigned int)) == 0)
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

void FloydWarshall::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[3] = {"Nodes", "Time(sec)", "[Transfer+Kernel]Time(sec)"};
        std::string stats[3];

        sampleTimer->totalTime = setupTime + totalKernelTime;

        stats[0] = toString(numNodes, std::dec);
        stats[1] = toString(sampleTimer->totalTime, std::dec);
        stats[2] = toString(totalKernelTime, std::dec);

        printStatistics(strArray, stats, 3);
    }
}

int FloydWarshall::cleanup()
{

    hipFree(pathDistanceBuffer);
    hipFree(pathBuffer);

    // release program resources (input memory etc.)
    FREE(pathDistanceMatrix);
    FREE(pathMatrix);
    FREE(verificationPathDistanceMatrix);
    FREE(verificationPathMatrix);

    return SDK_SUCCESS;
}

int
main(int argc, char * argv[])
{
    FloydWarshall hipFloydWarshall;

    // Initialize
    if(hipFloydWarshall.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(hipFloydWarshall.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Setup
    if(hipFloydWarshall.setup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Run
    if(hipFloydWarshall.run() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // VerifyResults
    if(hipFloydWarshall.verifyResults() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Cleanup
    if(hipFloydWarshall.cleanup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }
    hipFloydWarshall.printStats();

    return SDK_SUCCESS;
}
