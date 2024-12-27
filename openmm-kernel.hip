#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>

// Type definitions
typedef float real;
typedef float4 real4;
typedef float3 real3;
typedef int2 int2;

struct BoundingBox {
    real4 min;
    real4 max;
    
    __device__ real3 toReal3() const {
        return {max.x - min.x, max.y - min.y, max.z - min.z};
    }
};

// Constants
#define GROUP_SIZE 256
#define TILE_SIZE 32
#define BUFFER_SIZE 256
#define MAX_EXCLUSIONS 16
#define NUM_TILES_IN_BATCH 4
#define PADDED_CUTOFF 9.0f
#define PADDED_CUTOFF_SQUARED (PADDED_CUTOFF * PADDED_CUTOFF)
#define MAX_BITS_FOR_PAIRS 8
#define BLOCK_INDEX_MASK 0x3FFFFFFF
#define NUM_BLOCKS 4096
#define NUM_ATOMS 65536
#define PADDED_NUM_ATOMS 65536

// HIP specific types
typedef unsigned int warpflags;
typedef unsigned int tileflags;

// Helper functions 
inline __device__ real3 trimTo3(const real4& v) {
    return {v.x, v.y, v.z};
}

template<int N>
inline __device__ void mfma4x4(real4 pos1, real4 pos2, real c, tileflags& interacts) {
    // Implementation depends on hardware specifics
}

inline __device__ void collectInteractions(tileflags& interacts, real d, int j) {
    if (d < 0)
        interacts |= 1u << j;
}

// Warp-level population count and ballot operations
inline __device__ unsigned int warpPopc(unsigned int val) {
    return __popc(val);
}

inline __device__ unsigned int BALLOT(int predicate) {
    return __ballot(predicate);
}

#define BALLOT(predicate) __ballot(predicate)
#define SHFL(val, lane) __shfl(val, lane)


extern "C" __global__ __launch_bounds__(GROUP_SIZE) void findBlocksWithInteractions(real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        unsigned int* __restrict__ interactionCount, int* __restrict__ interactingTiles, unsigned int* __restrict__ interactingAtoms,
        int2* __restrict__ singlePairs, const real4* __restrict__ posq, unsigned int maxTiles, unsigned int maxSinglePairs,
        unsigned int startBlockIndex, unsigned int numBlocks, const unsigned int* __restrict__ sortedBlocks, const real4* __restrict__ sortedBlockCenter,
        const BoundingBox* __restrict__ sortedBlockBoundingBox,
#ifdef USE_LARGE_BLOCKS
        const real4* __restrict__ largeBlockCenter, const BoundingBox* __restrict__ largeBlockBoundingBox,
#endif
        const unsigned int* __restrict__ exclusionIndices, const unsigned int* __restrict__ exclusionRowIndices,
        real4* __restrict__ oldPositions, const int* __restrict__ rebuildNeighborList) {

    if (rebuildNeighborList[0] == 0)
        return; // The neighbor list doesn't need to be rebuilt.

    constexpr int tilesPerWarp = warpSize/TILE_SIZE;
    constexpr int warpsPerBlock = GROUP_SIZE/warpSize;
    const int indexInWarp = threadIdx.x%warpSize;
    const int indexInTile = threadIdx.x%TILE_SIZE;
    const int tileInWarp = tilesPerWarp == 1 ? 0 : (threadIdx.x/TILE_SIZE)%tilesPerWarp;
    const int warpInBlock = warpsPerBlock == 1 ? 0 : threadIdx.x/warpSize;
    const int warpIndex = blockIdx.x*warpsPerBlock + (warpsPerBlock == 1 ? 0 : warpInBlock);
    const warpflags warpMask = (static_cast<warpflags>(1)<<indexInWarp)-1;

    __shared__ int workgroupBuffer[BUFFER_SIZE*warpsPerBlock];
    __shared__ real4 workgroupPosBuffer[TILE_SIZE*warpsPerBlock];
    __shared__ int workgroupExclusions[MAX_EXCLUSIONS*warpsPerBlock];
    __shared__ int workgroupBlock2Buffer[(warpSize+1)*warpsPerBlock];

    int* buffer = workgroupBuffer+BUFFER_SIZE*warpInBlock;
    real4* posBuffer = workgroupPosBuffer+TILE_SIZE*warpInBlock;
    int* exclusionsForX = workgroupExclusions+MAX_EXCLUSIONS*warpInBlock;
    int* block2Buffer = workgroupBlock2Buffer+(warpSize+1)*warpInBlock;

    // Loop over blocks.

    int block1 = startBlockIndex+warpIndex/NUM_TILES_IN_BATCH;
    if (block1 < startBlockIndex+numBlocks) {
        // Load data for this block.  Note that all threads in a warp are processing the same block.

        int x = sortedBlocks[block1] & BLOCK_INDEX_MASK;
        real4 blockCenterX = sortedBlockCenter[block1];
        real3 blockSizeX = sortedBlockBoundingBox[block1].toReal3();
        int neighborsInBuffer = 0;
        real4 pos1 = posq[x*TILE_SIZE+indexInTile];
#ifdef USE_PERIODIC
        const bool singlePeriodicCopy = (0.5f*periodicBoxSize.x-blockSizeX.x >= PADDED_CUTOFF &&
                                         0.5f*periodicBoxSize.y-blockSizeX.y >= PADDED_CUTOFF &&
                                         0.5f*periodicBoxSize.z-blockSizeX.z >= PADDED_CUTOFF);
        if (singlePeriodicCopy) {
            // The box is small enough that we can just translate all the atoms into a single periodic
            // box, then skip having to apply periodic boundary conditions later.

            APPLY_PERIODIC_TO_POS_WITH_CENTER(pos1, blockCenterX)
        }
#endif
        pos1.w = 0.5f * (pos1.x * pos1.x + pos1.y * pos1.y + pos1.z * pos1.z);
        if (tileInWarp == 0) {
            posBuffer[indexInTile] = pos1;
        }

        // Load exclusion data for block x.

        const int exclusionStart = exclusionRowIndices[x];
        const int exclusionEnd = exclusionRowIndices[x+1];
        const int numExclusions = exclusionEnd-exclusionStart;
        #pragma unroll 1
        for (int j = indexInWarp; j < numExclusions; j += warpSize)
            exclusionsForX[j] = exclusionIndices[exclusionStart+j];

        // Loop over atom blocks to search for neighbors.  The threads in a warp compare block1 against warpSize
        // other blocks in parallel.
        // For small systems multiple warps (NUM_TILES_IN_BATCH = 4, 2...) process one block1 reducing the overall
        // duration of the kernel because first blocks block1 have to process more block2 blocks so most of compute
        // units are idle at the end of the kernel (the kernel works on the upper triangle of
        // the NUM_BLOCKS x NUM_BLOCKS matrix).

#ifdef USE_LARGE_BLOCKS
        warpflags largeBlockFlags = 0;
        int loadedLargeBlocks = 0;
#endif
        int block2Count = 0;
        // Load blocks from addresses aligned by warpSize for faster loading from sortedBlockCenter and sortedBlockBoundingBox.
        for (int block2Base = ((block1+1)/warpSize + warpIndex%NUM_TILES_IN_BATCH)*warpSize; block2Base < NUM_BLOCKS; block2Base += warpSize*NUM_TILES_IN_BATCH) {
            // Last iteration cannot be skipped (on CDNA where tilesPerWarp == 2)
            const bool lastIteration = block2Base + warpSize*NUM_TILES_IN_BATCH >= NUM_BLOCKS;
#ifdef USE_LARGE_BLOCKS
            if (loadedLargeBlocks == 0) {
                // Check the next set of large blocks.

                int largeBlockIndex = block2Base + warpSize*NUM_TILES_IN_BATCH*indexInWarp;
                bool includeLargeBlock = false;
                if (largeBlockIndex < NUM_BLOCKS) {
                    real4 largeCenter = largeBlockCenter[largeBlockIndex];
                    real3 largeSize = largeBlockBoundingBox[largeBlockIndex].toReal3();
                    real4 blockDelta = blockCenterX-largeCenter;
#ifdef USE_PERIODIC
                    APPLY_PERIODIC_TO_DELTA(blockDelta)
#endif
                    blockDelta.x = max(0.0f, fabs(blockDelta.x)-blockSizeX.x-largeSize.x);
                    blockDelta.y = max(0.0f, fabs(blockDelta.y)-blockSizeX.y-largeSize.y);
                    blockDelta.z = max(0.0f, fabs(blockDelta.z)-blockSizeX.z-largeSize.z);
                    includeLargeBlock = (blockDelta.x*blockDelta.x+blockDelta.y*blockDelta.y+blockDelta.z*blockDelta.z < PADDED_CUTOFF_SQUARED);
#ifdef TRICLINIC
                    // The calculation to find the nearest periodic copy is only guaranteed to work if the nearest copy is less than half a box width away.
                    // If there's any possibility we might have missed it, do a detailed check.

                    if (periodicBoxSize.z/2-blockSizeX.z-largeSize.z < PADDED_CUTOFF || periodicBoxSize.y/2-blockSizeX.y-largeSize.y < PADDED_CUTOFF)
                        includeLargeBlock = true;
#endif
                }
                largeBlockFlags = __ballot(includeLargeBlock);
                loadedLargeBlocks = warpSize;
            }
            loadedLargeBlocks--;
            if ((largeBlockFlags&1) == 0 && !lastIteration) {
                // None of the next warpSize blocks interact with block 1.

                largeBlockFlags >>= 1;
                continue;
            }
            largeBlockFlags >>= 1;
#endif
            int block2 = block2Base+indexInWarp;
            bool includeBlock2 = (block1 < block2 && block2 < NUM_BLOCKS);
            block2 = includeBlock2 ? block2 : block1;
            bool forceInclude = false;
            real4 blockCenterY = sortedBlockCenter[block2];
            real4 blockDelta = blockCenterX-blockCenterY;
#ifdef USE_PERIODIC
            APPLY_PERIODIC_TO_DELTA(blockDelta)
#endif
            includeBlock2 &= (blockDelta.x*blockDelta.x+blockDelta.y*blockDelta.y+blockDelta.z*blockDelta.z < (PADDED_CUTOFF+blockCenterX.w+blockCenterY.w)*(PADDED_CUTOFF+blockCenterX.w+blockCenterY.w));
#ifndef TRICLINIC
            if (!lastIteration && __ballot(includeBlock2) == 0)
                continue;
#endif
            real3 blockSizeY = sortedBlockBoundingBox[block2].toReal3();
            blockDelta.x = max(0.0f, fabs(blockDelta.x)-blockSizeX.x-blockSizeY.x);
            blockDelta.y = max(0.0f, fabs(blockDelta.y)-blockSizeX.y-blockSizeY.y);
            blockDelta.z = max(0.0f, fabs(blockDelta.z)-blockSizeX.z-blockSizeY.z);
            // includeBlock2 &= (blockDelta.x*blockDelta.x+blockDelta.y*blockDelta.y+blockDelta.z*blockDelta.z < PADDED_CUTOFF_SQUARED);
#ifdef TRICLINIC
            // The calculation to find the nearest periodic copy is only guaranteed to work if the nearest copy is less than half a box width away.
            // If there's any possibility we might have missed it, do a detailed check.

            if (periodicBoxSize.z/2-blockSizeX.z-blockSizeY.z < PADDED_CUTOFF || periodicBoxSize.y/2-blockSizeX.y-blockSizeY.y < PADDED_CUTOFF)
                includeBlock2 = forceInclude = true;
#endif

            // Collect any blocks we identified as potentially containing neighbors.

            // warpflags includeBlockFlags = __ballot(includeBlock2);
            // if (includeBlock2) {
            //     int index = block2Count + warpPopc(includeBlockFlags&warpMask);
            //     block2Buffer[index] = (block2 << 1) | (forceInclude ? 1 : 0);
            // }
            // block2Count += warpPopc(includeBlockFlags);

            // Loop over the collected candidates (each warp processes 2 blocks on CDNA or 1 block on RDNA).
            // Process even number of blocks on CDNA so both half-warps have work to do (except for
            // the last iteration of the for-block2Base loop when the left-over must be processed).

            const int block2ToProcess = lastIteration ? block2Count : block2Count/tilesPerWarp*tilesPerWarp;
            for (int block2Index = 0; block2Index < block2ToProcess; block2Index += tilesPerWarp) {
                // bool includeBlock2 = block2Index + tileInWarp < block2Count;
                // const int b = block2Buffer[min(block2Index + tileInWarp, block2Count - 1)];
                // const bool forceInclude = b & 1;
                // const int block2 = b >> 1;
                // int y = sortedBlocks[block2] & BLOCK_INDEX_MASK;

                // #pragma unroll 1
                // for (int k = indexInTile; k < numExclusions; k += TILE_SIZE)
                //     includeBlock2 &= (exclusionsForX[k] != y);
                // includeBlock2 = BALLOT(!includeBlock2) == 0;

                // Check each atom in block Y for interactions.

                // int atom2 = y*TILE_SIZE+indexInTile;
                // real4 pos2 = posq[atom2];
#ifdef USE_PERIODIC
                if (singlePeriodicCopy) {
                    APPLY_PERIODIC_TO_POS_WITH_CENTER(pos2, blockCenterX)
                }
#endif
                // pos2.w = 0.5f * (pos2.x * pos2.x + pos2.y * pos2.y + pos2.z * pos2.z);
                // real4 blockCenterY = sortedBlockCenter[block2];
                // real3 atomDelta = trimTo3(pos1)-trimTo3(blockCenterY);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(atomDelta)
#endif
                // tileflags atomFlags = BALLOT(forceInclude || atomDelta.x*atomDelta.x+atomDelta.y*atomDelta.y+atomDelta.z*atomDelta.z < (PADDED_CUTOFF+blockCenterY.w)*(PADDED_CUTOFF+blockCenterY.w));
                // tileflags interacts = 0;
                // The condition `posj.w + pos2.w - posj.x*pos2.x - posj.y*pos2.y - posj.z*pos2.z < 0.5f * PADDED_CUTOFF_SQUARED` is expressed as
                // `posj.x*pos2.x - posj.y*pos2.y - posj.z*pos2.z - posj.w - 0.5f * PADDED_CUTOFF_SQUARED - pos2.w` and computed using fma
                // (it saves 1 instruction).
                // Sign bit is used directly instead of `halfDist2 < 0.5f * PADDED_CUTOFF_SQUARED ? 1<<j : 0`.
#ifdef USE_PERIODIC
                if (!singlePeriodicCopy) {
                    while (atomFlags) {
                        int j = __ffs(atomFlags)-1;
                        atomFlags = atomFlags ^ (static_cast<tileflags>(1) << j);
                        real3 delta = trimTo3(pos2)-trimTo3(posBuffer[j]);
                        APPLY_PERIODIC_TO_DELTA(delta)
                        real d = delta.x*delta.x+delta.y*delta.y+delta.z*delta.z - PADDED_CUTOFF_SQUARED;
                        collectInteractions(interacts, d, j);
                    }
                }
                else {
#endif
                    // const real lim = 0.5f * PADDED_CUTOFF_SQUARED - pos2.w;
#if defined(USE_MFMA)
                    const vfloat c = { -lim, -lim, -lim, -lim };
                    mfma4x4<0>(pos1, pos2, c, interacts);
                    mfma4x4<1>(pos1, pos2, c, interacts);
                    mfma4x4<2>(pos1, pos2, c, interacts);
                    mfma4x4<3>(pos1, pos2, c, interacts);
                    mfma4x4<4>(pos1, pos2, c, interacts);
                    mfma4x4<5>(pos1, pos2, c, interacts);
                    mfma4x4<6>(pos1, pos2, c, interacts);
                    mfma4x4<7>(pos1, pos2, c, interacts);
#else
                    // while (atomFlags) {
                    //     int j = __ffs(atomFlags)-1;
                    //     atomFlags = atomFlags ^ (static_cast<tileflags>(1) << j);
                    //     real4 posj = posBuffer[j];
                    //     real d = fma(-posj.x, pos2.x, fma(-posj.y, pos2.y, fma(-posj.z, pos2.z, posj.w - lim)));
                    //     collectInteractions(interacts, d, j);
                    // }
#endif
#ifdef USE_PERIODIC
                }
#endif
                // if (atom2 >= NUM_ATOMS || !includeBlock2) {
                //     interacts = 0;
                // }

#if MAX_BITS_FOR_PAIRS > 0
                // const unsigned int interactCount = __popc(interacts);

                // // Record interactions that should be computed as single pairs rather than in blocks.
                // const bool storeAsSinglePair = interactCount > 0 && interactCount <= MAX_BITS_FOR_PAIRS;
                // if (__ballot(storeAsSinglePair)) {
                //     unsigned int sum = 0;
                //     unsigned int prevSum = 0;
                //     for (int i = 1; i <= MAX_BITS_FOR_PAIRS; i++) {
                //         warpflags b = __ballot(interactCount == i);
                //         sum += warpPopc(b) * i;
                //         prevSum += warpPopc(b&warpMask) * i;
                //     }
                //     unsigned int pairStartIndex = 0;
                //     if (indexInWarp == 0)
                //         pairStartIndex = atomicAdd(&interactionCount[1], sum);
                //     pairStartIndex = __shfl(pairStartIndex, 0);
                //     unsigned int pairIndex = pairStartIndex + prevSum;
                //     if (storeAsSinglePair && pairIndex+interactCount <= maxSinglePairs) {
                //         while (interacts != 0) {
                //             int j = __ffs(interacts)-1;
                //             singlePairs[pairIndex] = make_int2(atom2, x*TILE_SIZE+j);
                //             interacts = interacts ^ (static_cast<tileflags>(1) << j);
                //             pairIndex++;
                //         }
                //     }
                // }
#else
                const unsigned int interactCount = interacts;
#endif

                // Add any interacting atoms to the buffer.

                // warpflags includeAtomFlags = __ballot(interactCount > MAX_BITS_FOR_PAIRS);
                // if (interactCount > MAX_BITS_FOR_PAIRS) {
                //     int index = neighborsInBuffer+warpPopc(includeAtomFlags&warpMask);
                //     buffer[index] = atom2;
                // }
                // neighborsInBuffer += warpPopc(includeAtomFlags);
                // if (neighborsInBuffer > BUFFER_SIZE-warpSize) {
                //     // Store the new tiles to memory.

                //     unsigned int tilesToStore = neighborsInBuffer/warpSize*tilesPerWarp;
                //     unsigned int tileStartIndex = 0;
                //     if (indexInWarp == 0)
                //         tileStartIndex = atomicAdd(&interactionCount[0], tilesToStore);
                //     unsigned int newTileStartIndex = __shfl(tileStartIndex, 0);
                //     if (newTileStartIndex+tilesToStore <= maxTiles) {
                //         if (indexInWarp < tilesToStore)
                //             interactingTiles[newTileStartIndex+indexInWarp] = x;
                //         for (int j = 0; j < tilesToStore/tilesPerWarp; j++)
                //             interactingAtoms[newTileStartIndex*TILE_SIZE+j*warpSize+indexInWarp] = buffer[j*warpSize+indexInWarp];
                //     }
                //     if (indexInWarp+TILE_SIZE*tilesToStore < BUFFER_SIZE)
                //         buffer[indexInWarp] = buffer[indexInWarp+TILE_SIZE*tilesToStore];
                //     neighborsInBuffer -= TILE_SIZE*tilesToStore;
                // }
            }

            // Move not processed blocks to the head of block2Buffer.
            // if (indexInWarp < block2Count - block2ToProcess)
            //     block2Buffer[indexInWarp] = block2Buffer[block2ToProcess + indexInWarp];
            // block2Count = block2Count - block2ToProcess;
        }

        // If we have a partially filled buffer,  store it to memory.

        // if (neighborsInBuffer > 0) {
        //     unsigned int tilesToStore = (neighborsInBuffer+TILE_SIZE-1)/TILE_SIZE;
        //     unsigned int tileStartIndex = 0;
        //     if (indexInWarp == 0)
        //         tileStartIndex = atomicAdd(&interactionCount[0], tilesToStore);
        //     unsigned int newTileStartIndex = __shfl(tileStartIndex, 0);
        //     if (newTileStartIndex+tilesToStore <= maxTiles) {
        //         if (indexInWarp < tilesToStore)
        //             interactingTiles[newTileStartIndex+indexInWarp] = x;
        //         for (int j = 0; j <= tilesToStore/tilesPerWarp; j++) {
        //             if (j*warpSize+indexInWarp < tilesToStore*TILE_SIZE)
        //                 interactingAtoms[newTileStartIndex*TILE_SIZE+j*warpSize+indexInWarp] = (j*warpSize+indexInWarp < neighborsInBuffer ? buffer[j*warpSize+indexInWarp] : PADDED_NUM_ATOMS);
        //         }
        //     }
        // }
    }

//     // Record the positions the neighbor list is based on.

//     int i = threadIdx.x+blockIdx.x*GROUP_SIZE;
//     if (i < NUM_ATOMS) {
//         oldPositions[i] = posq[i];
//     }
}




int main() {
    // Initialize device pointers and allocate memory
    real4 *posq, *oldPositions;
    unsigned int *interactionCount, *interactingAtoms, *sortedBlocks;
    int *interactingTiles, *rebuildNeighborList;
    int2 *singlePairs;
    real4 *sortedBlockCenter;
    BoundingBox *sortedBlockBoundingBox;
    unsigned int *exclusionIndices, *exclusionRowIndices;

    // Set up grid and block dimensions
    dim3 gridDim((NUM_ATOMS + GROUP_SIZE - 1)/GROUP_SIZE);
    dim3 blockDim(GROUP_SIZE);

    // Example periodic box parameters
    real4 periodicBoxSize = {100.0f, 100.0f, 100.0f, 0.0f};
    real4 invPeriodicBoxSize = {1.0f/100.0f, 1.0f/100.0f, 1.0f/100.0f, 0.0f};
    real4 periodicBoxVecX = {100.0f, 0.0f, 0.0f, 0.0f};
    real4 periodicBoxVecY = {0.0f, 100.0f, 0.0f, 0.0f}; 
    real4 periodicBoxVecZ = {0.0f, 0.0f, 100.0f, 0.0f};

    // Launch kernel
    hipLaunchKernelGGL(findBlocksWithInteractions, gridDim, blockDim, 0, 0,
        periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ,
        interactionCount, interactingTiles, interactingAtoms,
        singlePairs, posq, NUM_BLOCKS, NUM_ATOMS,
        0, NUM_BLOCKS, sortedBlocks, sortedBlockCenter,
        sortedBlockBoundingBox,
        exclusionIndices, exclusionRowIndices,
        oldPositions, rebuildNeighborList);
    return 0;
}