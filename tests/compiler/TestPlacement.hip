#include <hip/hip_runtime.h>
#include <iostream>
#include <cstring>

// Test struct with constructor and destructor
struct TestStruct {
    int value;
    float data;
    
    __device__ TestStruct(int v, float d) : value(v), data(d) {}
    __device__ ~TestStruct() {}
    
    __device__ void setValue(int v) { value = v; }
    __device__ void setData(float d) { data = d; }
};

// Test class with more complex initialization
class TestClass {
public:
    int* ptr;
    int size;
    
    __device__ TestClass(int s) : size(s) {
        ptr = new int[size];
        for (int i = 0; i < size; i++) {
            ptr[i] = i * i;
        }
    }
    
    __device__ ~TestClass() {
        delete[] ptr;
    }
    
    __device__ int getSum() const {
        int sum = 0;
        for (int i = 0; i < size; i++) {
            sum += ptr[i];
        }
        return sum;
    }
};

__global__ void testPlacementNewSingle(char* buffer, int* result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        // Test single object placement new
        TestStruct* obj = new (buffer) TestStruct(42, 3.14f);
        
        // Verify object was constructed correctly
        result[0] = (obj->value == 42) ? 1 : 0;
        result[1] = (obj->data == 3.14f) ? 1 : 0;
        
        // Modify object
        obj->setValue(100);
        obj->setData(2.71f);
        
        result[2] = (obj->value == 100) ? 1 : 0;
        result[3] = (obj->data == 2.71f) ? 1 : 0;
        
        // Explicitly call destructor
        obj->~TestStruct();
    }
}

__global__ void testPlacementNewArray(char* buffer, int* result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        // Test array placement new
        TestStruct* arr = new (buffer) TestStruct[3] {
            TestStruct(1, 1.1f),
            TestStruct(2, 2.2f),
            TestStruct(3, 3.3f)
        };
        
        // Verify array elements
        result[0] = (arr[0].value == 1 && arr[0].data == 1.1f) ? 1 : 0;
        result[1] = (arr[1].value == 2 && arr[1].data == 2.2f) ? 1 : 0;
        result[2] = (arr[2].value == 3 && arr[2].data == 3.3f) ? 1 : 0;
        
        // Modify array elements
        for (int i = 0; i < 3; i++) {
            arr[i].setValue(i * 10);
            arr[i].setData(i * 1.5f);
        }
        
        result[3] = (arr[0].value == 0 && arr[0].data == 0.0f) ? 1 : 0;
        result[4] = (arr[1].value == 10 && arr[1].data == 1.5f) ? 1 : 0;
        result[5] = (arr[2].value == 20 && arr[2].data == 3.0f) ? 1 : 0;
        
        // Explicitly call destructors
        for (int i = 2; i >= 0; i--) {
            arr[i].~TestStruct();
        }
    }
}

__global__ void testPlacementNewComplex(char* buffer, int* result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        // Test placement new with complex class
        TestClass* obj = new (buffer) TestClass(5);
        
        // Verify object functionality
        int sum = obj->getSum();
        result[0] = (sum == 0 + 1 + 4 + 9 + 16) ? 1 : 0;  // 0^2 + 1^2 + 2^2 + 3^2 + 4^2 = 30
        result[1] = (obj->size == 5) ? 1 : 0;
        
        // Explicitly call destructor
        obj->~TestClass();
    }
}

int main() {
    const int bufferSize = 1024;
    char* d_buffer;
    int* d_result;
    int h_result[10];
    
    // Allocate device memory
    hipMalloc(&d_buffer, bufferSize);
    hipMalloc(&d_result, 10 * sizeof(int));
    
    // Initialize result array
    hipMemset(d_result, 0, 10 * sizeof(int));
    
    // Test 1: Single object placement new
    testPlacementNewSingle<<<1, 1>>>(d_buffer, d_result);
    hipDeviceSynchronize();
    
    hipMemcpy(h_result, d_result, 4 * sizeof(int), hipMemcpyDeviceToHost);
    
    bool test1Passed = true;
    for (int i = 0; i < 4; i++) {
        if (h_result[i] != 1) {
            std::cout << "FAIL: Single object placement new test " << i << " failed" << std::endl;
            test1Passed = false;
        }
    }
    if (test1Passed) {
        std::cout << "PASS: Single object placement new test passed" << std::endl;
    }
    
    // Reset buffer and result
    hipMemset(d_buffer, 0, bufferSize);
    hipMemset(d_result, 0, 10 * sizeof(int));
    
    // Test 2: Array placement new
    testPlacementNewArray<<<1, 1>>>(d_buffer, d_result);
    hipDeviceSynchronize();
    
    hipMemcpy(h_result, d_result, 6 * sizeof(int), hipMemcpyDeviceToHost);
    
    bool test2Passed = true;
    for (int i = 0; i < 6; i++) {
        if (h_result[i] != 1) {
            std::cout << "FAIL: Array placement new test " << i << " failed" << std::endl;
            test2Passed = false;
        }
    }
    if (test2Passed) {
        std::cout << "PASS: Array placement new test passed" << std::endl;
    }
    
    // Reset buffer and result
    hipMemset(d_buffer, 0, bufferSize);
    hipMemset(d_result, 0, 10 * sizeof(int));
    
    // Test 3: Complex object placement new
    testPlacementNewComplex<<<1, 1>>>(d_buffer, d_result);
    hipDeviceSynchronize();
    
    hipMemcpy(h_result, d_result, 2 * sizeof(int), hipMemcpyDeviceToHost);
    
    bool test3Passed = true;
    for (int i = 0; i < 2; i++) {
        if (h_result[i] != 1) {
            std::cout << "FAIL: Complex object placement new test " << i << " failed" << std::endl;
            test3Passed = false;
        }
    }
    if (test3Passed) {
        std::cout << "PASS: Complex object placement new test passed" << std::endl;
    }
    
    // Cleanup
    hipFree(d_buffer);
    hipFree(d_result);
    
    // Return 0 if all tests passed
    return (test1Passed && test2Passed && test3Passed) ? 0 : 1;
} 