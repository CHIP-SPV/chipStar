#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>

#define CHECK_ERROR(err)                                                      \
  if (err != CL_SUCCESS) {                                                     \
    fprintf(stderr, "Error: %d at line %d\n", err, __LINE__);                  \
    abort();                                                                   \
  }


const char *kernelSource = 
"__kernel void simple_kernel(__global char *ptr1, __global char *ptr2, int n) {\n"
"  int id = get_global_id(0);\n"
"  if (id < n) {\n"
"    ptr2[id] = ptr1[id];\n"
"  }\n"
"}\n";

void CL_CALLBACK callbackFunction(cl_event event, cl_int event_command_exec_status, void *user_data) {
  printf("callback complete\n");
}

int main() {
  cl_int err;
  cl_platform_id platform = NULL;
  cl_device_id device = NULL;
  cl_context context = NULL;
  cl_command_queue command_queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_event user_event1 = NULL;
  cl_event user_event2 = NULL;
  cl_event kernel_event = NULL;
  cl_event barrier_event1 = NULL;
  cl_event barrier_event2 = NULL;

  // Get platform
  cl_uint num_platforms;
  err = clGetPlatformIDs(0, NULL, &num_platforms);
  CHECK_ERROR(err);

  cl_platform_id *platforms = (cl_platform_id *)malloc(num_platforms * sizeof(cl_platform_id));
  err = clGetPlatformIDs(num_platforms, platforms, NULL);
  CHECK_ERROR(err);

  for (cl_uint i = 0; i < num_platforms; ++i) {
    size_t name_size;
    err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &name_size);
    CHECK_ERROR(err);

    char *platform_name = (char *)malloc(name_size);
    err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, name_size, platform_name, NULL);
    CHECK_ERROR(err);

    if (strstr(platform_name, "Intel(R) OpenCL Graphics") != NULL) {
      platform = platforms[i];
      free(platform_name);
      break;
    }
    free(platform_name);
  }
  free(platforms);

  if (platform == NULL) {
    fprintf(stderr, "Error: Intel(R) OpenCL Graphics platform not found\n");
    return -1;
  }

  // Get device
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  CHECK_ERROR(err);

  // Create context
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR(err);

  // Create command queue
  command_queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
  CHECK_ERROR(err);

  cl_command_queue command_queue2 = clCreateCommandQueueWithProperties(context, device, NULL, &err);
  CHECK_ERROR(err);

  // Allocate device memory
  void *ptr1 = clSVMAlloc(context, CL_MEM_READ_WRITE, 10, 1);
  CHECK_ERROR(ptr1 == NULL ? CL_OUT_OF_RESOURCES : CL_SUCCESS);
  void *ptr2 = clSVMAlloc(context, CL_MEM_READ_WRITE, 10, 1);
  CHECK_ERROR(ptr2 == NULL ? CL_OUT_OF_RESOURCES : CL_SUCCESS);

  // Create and compile program
  program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
  CHECK_ERROR(err);
  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char *log = (char *)malloc(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    fprintf(stderr, "Build log:\n%s\n", log);
    free(log);
    CHECK_ERROR(err);
  }

  // Create kernel
  kernel = clCreateKernel(program, "simple_kernel", &err);
  CHECK_ERROR(err);

  // Set kernel args
  err = clSetKernelArgSVMPointer(kernel, 0, ptr1);
  CHECK_ERROR(err);
  err = clSetKernelArgSVMPointer(kernel, 1, ptr2);
  CHECK_ERROR(err);
  size_t n = 10;
  err = clSetKernelArg(kernel, 2, sizeof(size_t), &n);
  CHECK_ERROR(err);

  // Enqueue kernel
  size_t global_size[1] = {10};
  size_t local_size[1] = {1};
  err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global_size, local_size, 0, NULL, &kernel_event);
  CHECK_ERROR(err);

  // Create user events
  user_event1 = clCreateUserEvent(context, &err);
  CHECK_ERROR(err);
  user_event2 = clCreateUserEvent(context, &err);
  CHECK_ERROR(err);

  // Enqueue barriers
  err = clEnqueueBarrierWithWaitList(command_queue, 1, &user_event1, &barrier_event1);
  CHECK_ERROR(err);
  err = clEnqueueBarrierWithWaitList(command_queue, 1, &user_event2, &barrier_event2);
  CHECK_ERROR(err);

  // Set event callback
  err = clSetEventCallback(barrier_event1, CL_COMPLETE, callbackFunction, NULL);
  CHECK_ERROR(err);

  // Flush queue
  err = clFlush(command_queue);
  CHECK_ERROR(err);

  // Set user event status
  err = clSetUserEventStatus(user_event1, CL_COMPLETE);
  CHECK_ERROR(err);
  err = clSetUserEventStatus(user_event2, CL_COMPLETE);
  CHECK_ERROR(err);

  // Finish execution
  err = clFinish(command_queue);
  CHECK_ERROR(err);

  // Free device memory
  clSVMFree(context, ptr2);
  clSVMFree(context, ptr1);

  // Release objects
  clReleaseEvent(barrier_event2);
  clReleaseEvent(barrier_event1);
  clReleaseEvent(kernel_event);
  clReleaseEvent(user_event2);
  clReleaseEvent(user_event1);
  clReleaseCommandQueue(command_queue2);
  clReleaseCommandQueue(command_queue);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseContext(context);
  clReleaseDevice(device);

  return 0;
}
