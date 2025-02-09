#define CL_TARGET_OPENCL_VERSION 200
#include <opencl-c.h>

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

#define ALIGNMENT 16
#define ALIGN_SIZE(size) (((size) + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1))
#define DEVICE_HEAP_SIZE (1024 * 1024) // 1MB heap

__global uchar* __chipspv_device_heap;


void __chip_init_device_heap(uchar* device_heap) {
    __chipspv_device_heap = (__global uchar*)device_heap;
}

// Structure for the header of each block in the heap
typedef struct {
    int size;   // Size of the block
    int used;   // Flag indicating if the block is used (1) or free (0)
} block_header_t;

void lock(__global volatile atomic_int* mutex) {
    int attempts = 0;
    int backoff = 1;
    do {
        if (atomic_exchange_explicit(mutex, 1, memory_order_acquire, memory_scope_device) == 0) {
            return; // Lock acquired
        }
        for (int i = 0; i < backoff; i++) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        attempts++;
        backoff = min(backoff * 2, 1024);
    } while (attempts < 100);
}

void unlock(__global volatile atomic_int* mutex) {
    atomic_store_explicit(mutex, 0, memory_order_release, memory_scope_device);
}

// Add these debug macros
// #define DEBUG_PRINT(fmt, ...) printf("[DEBUG] " fmt "\n", ##__VA_ARGS__)
// #define ERROR_PRINT(fmt, ...) printf("[ERROR] " fmt "\n", ##__VA_ARGS__)
#define DEBUG_PRINT(fmt, ...) 
#define ERROR_PRINT(fmt, ...) 



void* __chip_malloc(unsigned int size) {
    __global void* result = NULL;

    // Ensure only the first thread in the 3D workgroup performs malloc
    if (get_local_id(0) == 0 && get_local_id(1) == 0 && get_local_id(2) == 0) {
        // Ensure the size is aligned
        size = ALIGN_SIZE(size);

        // Pointers to the mutex and initialization flag
        __global volatile atomic_int* mutex = (__global volatile atomic_int*)&__chipspv_device_heap[0];
        __global int* initialized = (__global int*)&__chipspv_device_heap[sizeof(atomic_int)];

        // Pointer to the start of the heap
        __global uchar* heap = (__global uchar*)__chipspv_device_heap + sizeof(atomic_int) + sizeof(int);
        int real_heap_size = DEVICE_HEAP_SIZE - sizeof(atomic_int) - sizeof(int);

        lock(mutex);

        // Initialize the heap if not already done
        if (*initialized == 0) {
            __global block_header_t* first_header = (__global block_header_t*)heap;
            first_header->size = real_heap_size - sizeof(block_header_t);
            first_header->used = 0;
            *initialized = 1;
            DEBUG_PRINT("Heap initialized with size %d", first_header->size);
        }

        // Start of malloc algorithm
        __global uchar* heap_end = heap + real_heap_size;
        __global uchar* ptr = heap;

        DEBUG_PRINT("Attempting to allocate %zu bytes", size);

        while (ptr + sizeof(block_header_t) <= heap_end) {
            __global block_header_t* header = (__global block_header_t*)ptr;

            // Add error checking
            if (header == NULL) {
                ERROR_PRINT("Invalid header pointer");
                break;
            }

            DEBUG_PRINT("Checking block at %p, size: %d, used: %d", (void*)ptr, header->size, header->used);

            if (header->used == 0 && header->size >= size) {
                // Found a suitable block
                int remaining_size = header->size - size - sizeof(block_header_t);
                if (remaining_size > ALIGNMENT) {
                    // Split the block
                    __global uchar* next_block_ptr = ptr + sizeof(block_header_t) + size;
                    __global block_header_t* next_header = (__global block_header_t*)next_block_ptr;
                    next_header->size = remaining_size;
                    next_header->used = 0;

                    header->size = size;
                    DEBUG_PRINT("Split block. New block at %p with size %d", next_block_ptr, remaining_size);
                }
                header->used = 1;
                result = ptr + sizeof(block_header_t);
                DEBUG_PRINT("Allocated block at %p with size %d", result, size);
                break;
            }
            // Move to the next block
            ptr = ptr + sizeof(block_header_t) + header->size;
        }

        if (result == NULL) {
            // No suitable block found
            ERROR_PRINT("device_malloc: Out of memory");
        }

        unlock(mutex);
    }

    // Broadcast the result to all threads in the workgroup
    result = (__global void*)work_group_broadcast((uintptr_t)result, 0);
    barrier(CLK_LOCAL_MEM_FENCE);
    return result;
}

void __chip_free(void* ptr) {
    if (ptr == NULL) return;

    // Ensure only the first thread in the 3D workgroup performs free
    if (get_local_id(0) == 0 && get_local_id(1) == 0 && get_local_id(2) == 0) {
        uchar* device_heap = (__global uchar*)__chipspv_device_heap;
        __global volatile atomic_int* mutex = (__global volatile atomic_int*)&device_heap[0];
        lock(mutex);

        __global block_header_t* header = (__global block_header_t*)(((__global uchar*)ptr) - sizeof(block_header_t));
        
        if (header->used) {
            header->used = 0;
            DEBUG_PRINT("Freed block at %p with size %d", ptr, header->size);

            // Attempt to coalesce with next block if it's free
            __global block_header_t* next_header = (__global block_header_t*)((__global uchar*)ptr + header->size);
            if (((__global uchar*)next_header < device_heap + DEVICE_HEAP_SIZE) && !next_header->used) {
                header->size += sizeof(block_header_t) + next_header->size;
                DEBUG_PRINT("Coalesced with next block, new size: %d", header->size);
            }

            // Attempt to coalesce with previous block if it's free
            __global block_header_t* prev_header = (__global block_header_t*)device_heap;
            while (((__global uchar*)prev_header + sizeof(block_header_t) + prev_header->size) < (__global uchar*)header) {
                if (!prev_header->used && ((__global uchar*)prev_header + sizeof(block_header_t) + prev_header->size == (__global uchar*)header)) {
                    prev_header->size += sizeof(block_header_t) + header->size;
                    DEBUG_PRINT("Coalesced with previous block, new size: %d", prev_header->size);
                    break;
                }
                prev_header = (__global block_header_t*)((__global uchar*)prev_header + sizeof(block_header_t) + prev_header->size);
            }
        } else {
            ERROR_PRINT("Attempted to free an already free block at %p", ptr);
        }
        unlock(mutex);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}
