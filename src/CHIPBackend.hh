/**
 * @file CHIPBackend.hh
 * @author Paulius Velesko (pvelesko@gmail.com)
 * @brief CHIPBackend class definition. CHIP backends are to inherit from this
 * base class and override desired virtual functions. Overrides for this class
 * are expected to be minimal with primary overrides being done on lower-level
 * classes such as CHIPContext consturctors, etc.
 * @version 0.1
 * @date 2021-08-19
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef CHIP_BACKEND_H
#define CHIP_BACKEND_H

#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <mutex>
#include <string>
#include <vector>
#include <stack>

#include "spirv.hh"
#include "hip/hip_runtime_api.h"
#include "hip/spirv_hip.hh"

#include "CHIPDriver.hh"
#include "logging.hh"
#include "macros.hh"
#include "CHIPException.hh"

class CHIPEventMonitor;

class CHIPCallbackData {
 public:
  CHIPQueue* chip_queue;
  CHIPEvent* gpu_ready;
  CHIPEvent* cpu_callback_complete;
  CHIPEvent* gpu_ack;

  hipError_t status;
  void* callback_args;
  hipStreamCallback_t callback_f;

  CHIPCallbackData(hipStreamCallback_t callback_f_, void* callback_args_,
                   CHIPQueue* chip_queue_);

  /**
   * @brief
   *
   */
  virtual void setup();

  void execute(hipError_t resultFromDependency) {
    callback_f(chip_queue, resultFromDependency, callback_args);
  }
};

void* monitor_wrapper(void* event_monitor_);
class CHIPEventMonitor {
  std::mutex mtx;
  typedef void* (*THREADFUNCPTR)(void*);

 protected:
  // The thread ID for monitor thread
  pthread_t thread;

  CHIPQueue* chip_queue;

 public:
  /**
   * @brief Pop the callback stack and execute
   */
  virtual void monitor();

  CHIPEventMonitor() {
    logDebug("CHIPEventMonitor::CHIPEventMonitor()");
    auto res = pthread_create(&thread, 0, (THREADFUNCPTR)&monitor_wrapper,
                              (void*)this);
    logDebug("Thread Created with ID : {}", thread);
  }

  /**
   * @brief wait until event completes
   *
   */
  void wait();
};

inline void* monitor_wrapper(void* event_monitor_) {
  CHIPEventMonitor* event_monitor = (CHIPEventMonitor*)event_monitor_;
  event_monitor->monitor();
  return nullptr;
}

class CHIPTexture {
 protected:
  // delete default constructor since texture needs both image and sampler
  CHIPTexture() = delete;

  CHIPTexture(intptr_t image_, intptr_t sampler_)
      : image(image_), sampler(sampler_) {}

 public:
  intptr_t image;
  intptr_t sampler;
  hipTextureObject_t tex_obj;
  hipTextureObject_t get() { return tex_obj; }
};

template <class T>
std::string resultToString(T err);

enum class CHIPMemoryType : unsigned { Host = 0, Device = 1, Shared = 2 };
enum class CHIPEventType : unsigned {
  Default = hipEventDefault,
  BlockingSync = hipEventBlockingSync,
  DisableTiming = hipEventDisableTiming,
  Interprocess = hipEventInterprocess
};

struct allocation_info {
  void* base_ptr;
  size_t size;
};

/**
 * @brief Class for keeping track of device allocations.
 *
 */
class CHIPAllocationTracker {
 private:
  std::mutex mtx;
  std::unordered_map<void*, void*> host_to_dev;
  std::unordered_map<void*, void*> dev_to_host;
  std::string name;
  std::set<void*> ptr_set;

  std::unordered_map<void*, allocation_info> dev_to_allocation_info;

 public:
  size_t global_mem_size, total_mem_used, max_mem_used;
  /**
   * @brief Construct a new CHIPAllocationTracker object
   *
   * @param global_mem_size_ Total available global memory on the device
   * @param name_ name for this allocation tracker for logging. Normally device
   * name
   */
  CHIPAllocationTracker(size_t global_mem_size_, std::string name_);

  /**
   * @brief Destroy the CHIPAllocationTracker object
   *
   */
  ~CHIPAllocationTracker();

  /**
   * @brief Get the Name object
   *
   * @return std::string
   */
  std::string getName();

  /**
   * @brief Get allocation_info based on host pointer
   *
   * @return allocation_info contains the base pointer and allocation size;
   */
  allocation_info* getByHostPtr(const void*);
  /**
   * @brief Get allocation_info based on device pointer
   *
   * @return allocation_info contains the base pointer and allocation size;
   */
  allocation_info* getByDevPtr(const void*);

  /**
   * @brief Reserve memory for an allocation.
   * This method is run prior to allocations to keep track of how much memory is
   * available on the device
   *
   * @param bytes
   * @return true Reservation successful
   * @return false Not enough available memory for reservation of this size.
   */
  bool reserveMem(size_t bytes);

  /**
   * @brief Release some of the reserved memory. Called by free()
   *
   * @param bytes
   * @return true
   * @return false
   */
  bool releaseMemReservation(size_t bytes);

  /**
   * @brief Record the pointer received from CHIPContext::allocate_()
   *
   * @param dev_ptr
   */
  void recordAllocation(void* dev_ptr, size_t size_);
};

class CHIPDeviceVar {
 private:
  std::string host_var_name;
  void* dev_ptr;
  size_t size;

 public:
  CHIPDeviceVar(std::string host_var_name_, void* dev_ptr_, size_t size);
  ~CHIPDeviceVar();

  void* getDevAddr();
  std::string getName();
  size_t getSize();
};

// fw declares
class CHIPExecItem;
class CHIPQueue;
class CHIPContext;
class CHIPDevice;

class CHIPEvent {
 protected:
  std::mutex mtx;
  event_status_e event_status;
  /**
   * @brief event bahavior modifier -  valid values are hipEventDefault,
   * hipEventBlockingSync, hipEventDisableTiming, hipEventInterprocess
   *
   */
  CHIPEventType flags;
  /**
   * @brief Events are always created with a context
   *
   */
  CHIPContext* chip_context;

  /**
   * @brief hidden default constructor for CHIPEvent. Only derived class
   * constructor should be called.
   *
   */
  CHIPEvent() = default;

 public:
  /**
   * @brief CHIPEvent constructor. Must always be created with some context.
   *
   */
  CHIPEvent(CHIPContext* ctx_, CHIPEventType flags_ = CHIPEventType::Default);
  /**
   * @brief Destroy the CHIPEvent object
   *
   */
  virtual ~CHIPEvent() = default;
  /**
   * @brief Enqueue this event in a given CHIPQueue
   *
   * @param chip_queue_ CHIPQueue in which to enque this event
   * @return true
   * @return false
   */
  virtual void recordStream(CHIPQueue* chip_queue_) = 0;
  /**
   * @brief Wait for this event to complete
   *
   * @return true
   * @return false
   */
  virtual bool wait() = 0;
  /**
   * @brief Query the event to see if it completed
   *
   * @return true
   * @return false
   */
  virtual bool isFinished() = 0;
  /**
   * @brief Calculate absolute difference between completion timestamps of this
   * event and other
   *
   * @param other
   * @return float
   */
  virtual float getElapsedTime(CHIPEvent* other) = 0;

  /**
   * @brief Toggle this event from the host.
   *
   */
  virtual void hostSignal() = 0;

  virtual void barrier(CHIPQueue* chip_queue_) = 0;
};

/**
 * @brief Module abstraction. Contains global variables and kernels. Can be
 * extracted from FatBinary or loaded at runtime.
 * OpenCL - ClProgram
 * Level Zero - zeModule
 * ROCclr - amd::Program
 * CUDA - CUmodule
 */
class CHIPModule {
 protected:
  uint8_t* funcIL;
  size_t ilSize;
  std::mutex mtx;
  // Global variables
  std::vector<CHIPDeviceVar*> chip_vars;
  // Kernels
  std::vector<CHIPKernel*> chip_kernels;
  /// Binary representation extracted from FatBinary
  std::string src;
  // Kernel JIT compilation can be lazy
  std::once_flag compiled;

  int32_t* binary_data;
  OpenCLFunctionInfoMap func_infos;

  /**
   * @brief hidden default constuctor. Only derived type constructor should be
   * called.
   *
   */
  CHIPModule() = default;

 public:
  /**
   * @brief Destroy the CHIPModule object
   *
   */
  ~CHIPModule();
  /**
   * @brief Construct a new CHIPModule object.
   * This constructor should be implemented by the derived class (specific
   * backend implementation). Call to this constructor should result in a
   * populated chip_kernels vector.
   *
   * @param module_str string prepresenting the binary extracted from FatBinary
   */
  CHIPModule(std::string* module_str);
  /**
   * @brief Construct a new CHIPModule object using move semantics
   *
   * @param module_str string from which to move resources
   */
  CHIPModule(std::string&& module_str);

  /**
   * @brief Add a CHIPKernel to this module.
   * During initialization when the FatBinary is consumed, a CHIPModule is
   * constructed for every device. SPIR-V kernels reside in this module. This
   * method is called called via the constructor during this initialization
   * phase. Modules can also be loaded from a file during runtime, however.
   *
   * @param kernel CHIPKernel to be added to this module.
   */
  void addKernel(CHIPKernel* kernel);

  /**
   * @brief Wrapper around compile() called via std::call_once
   *
   * @param chip_dev device for which to compile the kernels
   */
  void compileOnce(CHIPDevice* chip_dev);
  /**
   * @brief Kernel JIT compilation can be lazy. This is configured via Cmake
   * LAZY_JIT option. If LAZY_JIT is set to true then this module won't be
   * compiled until the first call to one of its kernels. If LAZY_JIT is set to
   * false(default) then this method should be called in the constructor;
   *
   * This method should populate this modules chip_kernels vector. These
   * kernels would have a name extracted from the kernel but no associated host
   * function pointers.
   *
   */
  virtual void compile(CHIPDevice* chip_dev) = 0;
  /**
   * @brief Get the Global Var object
   * A module, along with device kernels, can also contain global variables.
   *
   * @param name global variable name
   * @return CHIPDeviceVar*
   */
  virtual CHIPDeviceVar* getGlobalVar(const char* var_name_);

  /**
   * @brief parse this module for variable matching a given name, create
   * a CHIPDeviceVar for it and add it to this modules device var list
   *
   * @param var_name_ name of the variable to register
   * @return true a variable matching the given name was found and registered
   * @return false no variable was found matching this name
   */
  virtual bool registerVar(const char* var_name_) = 0;

  /**
   * @brief Get the Kernel object
   *
   * @param name name of the corresponding host function
   * @return CHIPKernel*
   */
  CHIPKernel* getKernel(std::string name);

  /**
   * @brief Get the Kernels object
   *
   * @return std::vector<CHIPKernel*>&
   */
  std::vector<CHIPKernel*>& getKernels();

  /**
   * @brief Get the Kernel object
   *
   * @param host_f_ptr host-side function pointer
   * @return CHIPKernel*
   */
  CHIPKernel* getKernel(const void* host_f_ptr);

  /**
   * @brief consume SPIRV and fill in OCLFuncINFO
   *
   */
  void consumeSPIRV();
};

/**
 * @brief Contains information about the function on the host and device
 */
class CHIPKernel {
 protected:
  /**
   * @brief hidden default constructor. Only derived type constructor should be
   * called.
   *
   */
  CHIPKernel(std::string host_f_name_, OCLFuncInfo* func_info_);
  /// Name of the function
  std::string host_f_name;
  /// Pointer to the host function
  const void* host_f_ptr;
  /// Pointer to the device function
  const void* dev_f_ptr;

  OCLFuncInfo* func_info;

 public:
  ~CHIPKernel();

  /**
   * @brief Get the Name object
   *
   * @return std::string
   */
  std::string getName();

  /**
   * @brief Get the Func Info object
   *
   * @return OCLFuncInfo&
   */
  OCLFuncInfo* getFuncInfo();
  /**
   * @brief Get the associated host pointer to a host function
   *
   * @return const void*
   */
  const void* getHostPtr();
  /**
   * @brief Get the associated funciton pointer on the device
   *
   * @return const void*
   */
  const void* getDevPtr();

  /**
   * @brief Get the Name object
   *
   * @return std::string
   */
  void setName(std::string host_f_name_);
  /**
   * @brief Get the associated host pointer to a host function
   *
   * @return const void*
   */
  void setHostPtr(const void* host_f_ptr_);
  /**
   * @brief Get the associated funciton pointer on the device
   *
   * @return const void*
   */
  void setDevPtr(const void* hev_f_ptr_);
};

/**
 * @brief Contains kernel arguments and a queue on which to execute.
 * Prior to kernel launch, the arguments are setup via
 * CHIPBackend::configureCall(). Because of this, we get the kernel last so the
 * kernel so the launch() takes a kernel argument as opposed to queue receiving
 * a CHIPExecItem containing the kernel and arguments
 *
 */
class CHIPExecItem {
 protected:
  size_t shared_mem;
  // Structures for old HIP launch API.
  std::vector<uint8_t> arg_data;
  std::vector<std::tuple<size_t, size_t>> offset_sizes;

  dim3 grid_dim;
  dim3 block_dim;

  CHIPQueue* stream;
  CHIPKernel* chip_kernel;
  CHIPQueue* chip_queue;

  // Structures for new HIP launch API.
  void** ArgsPointer = nullptr;

 public:
  /**
   * @brief Deleted default constructor
   * Doesn't make sense for CHIPExecItem to exist without arguments
   *
   */
  CHIPExecItem() = delete;
  /**
   * @brief Construct a new CHIPExecItem object
   *
   * @param grid_dim_
   * @param block_dim_
   * @param shared_mem_
   * @param chip_queue_
   */
  CHIPExecItem(dim3 grid_dim_, dim3 block_dim_, size_t shared_mem_,
               hipStream_t chip_queue_);

  /**
   * @brief Destroy the CHIPExecItem object
   *
   */
  ~CHIPExecItem();

  /**
   * @brief Get the Kernel object
   *
   * @return CHIPKernel* Kernel to be executed
   */
  CHIPKernel* getKernel();
  /**
   * @brief Get the Queue object
   *
   * @return CHIPQueue*
   */
  CHIPQueue* getQueue();

  std::vector<uint8_t> getArgData();

  /**
   * @brief Get the Grid object
   *
   * @return dim3
   */
  dim3 getGrid();

  /**
   * @brief Get the Block object
   *
   * @return dim3
   */
  dim3 getBlock();

  /**
   * @brief Get the SharedMem
   *
   * @return size_t
   */
  size_t getSharedMem();

  /**
   * @brief Setup a single argument.
   * gets called by hipSetupArgument calls to which are emitted by hip-clang.
   *
   * @param arg
   * @param size
   * @param offset
   */
  void setArg(const void* arg, size_t size, size_t offset);

  /**
   * @brief Set the Arg Pointer object for launching kernels via new HIP API
   *
   * @param args args pointer
   */
  void setArgPointer(void** args) { ArgsPointer = args; }

  /**
   * @brief Sets up the kernel arguments via backend API calls.
   * Called after all the arugments are setup either via hipSetupArg() (old HIP
   * kernel launch API)
   * Or after hipLaunchKernel (new HIP kernel launch API)
   *
   */
  void setupAllArgs();

  /**
   * @brief Launch a kernel associated with a host function pointer.
   * Looks up the CHIPKernel associated with this pointer and calls launch()
   *
   * @param hostPtr pointer to the host function
   * @return hipError_t possible values: hipSuccess, hipErrorLaunchFailure
   */
  hipError_t launchByHostPtr(const void* hostPtr);
};

/**
 * @brief Compute device class
 */
class CHIPDevice {
 protected:
  std::string device_name;
  std::mutex mtx;
  CHIPContext* ctx;
  std::vector<CHIPQueue*> chip_queues;
  int active_queue_id = 0;
  std::once_flag propsPopulated;

  hipDeviceAttribute_t attrs;
  hipDeviceProp_t hip_device_props;

  size_t TotalUsedMem;
  size_t MaxUsedMem;

 public:
  /// chip_modules in parsed representation
  std::vector<CHIPModule*> chip_modules;

  /// Map host pointer-to-module to pointer-to-CHIPModule
  std::unordered_map<std::string*, CHIPModule*> module_str_to_chip_map;

  int idx;

  CHIPAllocationTracker* allocation_tracker = nullptr;

  /**
   * @brief Construct a new CHIPDevice object
   *
   */
  CHIPDevice(CHIPContext* ctx_);

  /**
   * @brief Construct a new CHIPDevice object
   *
   */
  CHIPDevice();

  /**
   * @brief Destroy the CHIPDevice object
   *
   */
  ~CHIPDevice();

  /**
   * @brief Get the Kernels object
   *
   * @return std::vector<CHIPKernel*>&
   */
  std::vector<CHIPKernel*> getKernels();

  /**
   * @brief Get the Modules object
   *
   * @return std::vector<CHIPModule*>&
   */
  std::vector<CHIPModule*>& getModules();

  /**
   * @brief Use a backend to populate device properties such as memory
   * available, frequencies, etc.
   */
  void populateDeviceProperties();

  /**
   * @brief Use a backend to populate device properties such as memory
   * available, frequencies, etc.
   */
  virtual void populateDeviceProperties_() = 0;

  /**
   * @brief Query the device for properties
   *
   * @param prop
   */
  void copyDeviceProperties(hipDeviceProp_t* prop);

  /**
   * @brief Use the host function pointer to retrieve the kernel
   *
   * @param hostPtr
   * @return CHIPKernel* CHIPKernel associated with this host pointer
   */
  CHIPKernel* findKernelByHostPtr(const void* hostPtr);

  /**
   * @brief Get the context object
   *
   * @return CHIPContext* pointer to the CHIPContext object this CHIPDevice
   * was created with
   */
  CHIPContext* getContext();

  /**
   * @brief Construct an additional queue for this device
   *
   * @param flags
   * @param priority
   * @return CHIPQueue* pointer to the newly created queue (can also be found
   * in chip_queues vector)
   */
  virtual CHIPQueue* addQueue(
      unsigned int flags,
      int priority) = 0;  // TODO how do I instantiate a CHIPQueue derived type
                          // in a generic way?

  /**
   * @brief Add a queue to this device
   *
   * @param chip_queue_  CHIPQueue to be added
   */
  void addQueue(CHIPQueue* chip_queue_);
  /**
   * @brief Get the Queues object
   *
   * @return std::vector<CHIPQueue*>
   */
  std::vector<CHIPQueue*> getQueues();
  /**
   * @brief HIP API allows for setting the active device, not the active queue
   * so active device's active queue is always it's 0th/default/primary queue
   *
   * @return CHIPQueue*
   */
  CHIPQueue* getActiveQueue();
  /**
   * @brief Remove a queue from this device's queue vector
   *
   * @param q
   * @return true
   * @return false
   */
  bool removeQueue(CHIPQueue* q);

  /**
   * @brief Get the integer ID of this device as it appears in the Backend's
   * chip_devices list
   *
   * @return int
   */
  int getDeviceId();
  /**
   * @brief Get the device name
   *
   * @return std::string
   */
  std::string getName();

  /**
   * @brief Destroy all allocations and reset all state on the current device in
   the current process.
   *
   */
  virtual void reset() = 0;

  /**
   * @brief Query for a specific device attribute. Implementation copied from
   * HIPAMD.
   *
   * @param attr attribute to query
   * @return int attribute value. In case invalid query returns -1;
   */
  int getAttr(hipDeviceAttribute_t attr);

  /**
   * @brief Get the total global memory available for this device.
   *
   * @return size_t
   */
  size_t getGlobalMemSize();

  /**
   * @brief Set the Cache Config object
   *
   * @param cfg configuration
   */
  virtual void setCacheConfig(hipFuncCache_t cfg);

  /**
   * @brief Get the cache configuration for this device
   *
   * @return hipFuncCache_t
   */
  virtual hipFuncCache_t getCacheConfig();

  /**
   * @brief Configure shared memory for this device
   *
   * @param config
   */
  virtual void setSharedMemConfig(hipSharedMemConfig config);

  /**
   * @brief Get the shared memory configuration for this device
   *
   * @return hipSharedMemConfig
   */
  virtual hipSharedMemConfig getSharedMemConfig();

  /**
   * @brief Setup the cache configuration for the device to use when executing
   * this function
   *
   * @param func
   * @param config
   */
  virtual void setFuncCacheConfig(const void* func, hipFuncCache_t config);

  /**
   * @brief Check if the current device has same PCI bus ID as the one given by
   * input
   *
   * @param pciDomainID
   * @param pciBusID
   * @param pciDeviceID
   * @return true
   * @return false
   */
  bool hasPCIBusId(int pciDomainID, int pciBusID, int pciDeviceID);

  /**
   * @brief Get peer-accesability between this and another device
   *
   * @param peerDevice
   * @return int
   */
  int getPeerAccess(CHIPDevice* peerDevice);

  /**
   * @brief Set access between this and another device
   *
   * @param peer
   * @param flags
   * @param canAccessPeer
   * @return hipError_t
   */
  hipError_t setPeerAccess(CHIPDevice* peer, int flags, bool canAccessPeer);

  /**
   * @brief Get the total used global memory
   *
   * @return size_t
   */
  size_t getUsedGlobalMem();

  /**
   * @brief Get the global variable that came from a FatBinary module
   *
   * @param var_name host pointer to the variable
   * @return CHIPDeviceVar*
   */
  virtual CHIPDeviceVar* getDynGlobalVar(const char* var_name_) {
    UNIMPLEMENTED(nullptr);
  }

  /**
   * @brief Get the global variable that came from a FatBinary module
   *
   * @param var_name name of the global variable
   * @return CHIPDeviceVar*
   */
  virtual CHIPDeviceVar* getStatGlobalVar(const char* var_name_);

  /**
   * @brief Get the global variable
   *
   * @param var_name name of the global variable
   * @return CHIPDeviceVar* if not found returns nullptr
   */
  CHIPDeviceVar* getGlobalVar(const char* var_name_);

  /**
   * @brief Take the module source, compile the kernels and associate the host
   * function pointer with a kernel whose name matches host function name
   *
   * @param module_str Binary representation of the SPIR-V module
   * @param host_f_ptr host function pointer
   * @param host_f_name host function name
   */
  void registerFunctionAsKernel(std::string* module_str, const void* host_f_ptr,
                                const char* host_f_name);

  virtual CHIPModule* addModule(std::string* module_str) = 0;

  virtual CHIPTexture* createTexture(
      const hipResourceDesc* pResDesc, const hipTextureDesc* pTexDesc,
      const struct hipResourceViewDesc* pResViewDesc) = 0;

  virtual void destroyTexture(CHIPTexture* textureObject) = 0;
};

/**
 * @brief Context class
 * Contexts contain execution queues and are created on top of a single or
 * multiple devices. Provides for creation of additional queues, events, and
 * interaction with devices.
 */
class CHIPContext {
 protected:
  std::vector<CHIPDevice*> chip_devices;
  std::vector<CHIPQueue*> chip_queues;
  std::mutex mtx;
  std::vector<void*> allocated_ptrs;

  unsigned int flags;

 public:
  /**
   * @brief Construct a new CHIPContext object
   *
   */
  CHIPContext();
  /**
   * @brief Destroy the CHIPContext object
   *
   */
  ~CHIPContext();

  /**
   * @brief Add a device to this context
   *
   * @param dev pointer to CHIPDevice object
   * @return true if device was added successfully
   * @return false upon failure
   */
  void addDevice(CHIPDevice* dev);
  /**
   * @brief Add a queue to this context
   *
   * @param q CHIPQueue to be added
   */
  void addQueue(CHIPQueue* q);

  /**
   * @brief Get this context's CHIPDevices
   *
   * @return std::vector<CHIPDevice*>&
   */
  std::vector<CHIPDevice*>& getDevices();

  /**
   * @brief Get the this contexts CHIPQueues
   *
   * @return std::vector<CHIPQueue*>&
   */
  std::vector<CHIPQueue*>& getQueues();

  /**
   * @brief Find a queue. If a null pointer is passed, return the Active Queue
   * (active devices's primary queue). If this queue is not found in this
   * context then return nullptr
   *
   * @param stream CHIPQueue to find
   * @return hipStream_t
   */
  hipStream_t findQueue(hipStream_t stream);

  /**
   * @brief Allocate data.
   * Calls reserveMem() to keep track memory used on the device.
   * Calls CHIPContext::allocate_(size_t size, size_t alignment,
   * CHIPMemoryType mem_type) with allignment = 0 and allocation type = Shared
   *
   *
   * @param size size of the allocation
   * @return void* pointer to allocated memory
   */
  void* allocate(size_t size);

  /**
   * @brief Allocate data.
   * Calls reserveMem() to keep track memory used on the device.
   * Calls CHIPContext::allocate_(size_t size, size_t alignment,
   * CHIPMemoryType mem_type) with allignment = 0
   *
   * @param size size of the allocation
   * @param mem_type type of the allocation: Host, Device, Shared
   * @return void* pointer to allocated memory
   */
  void* allocate(size_t size, CHIPMemoryType mem_type);

  /**
   * @brief Allocate data.
   * Calls reserveMem() to keep track memory used on the device.
   * Calls CHIPContext::allocate_(size_t size, size_t alignment,
   * CHIPMemoryType mem_type)
   *
   * @param size size of the allocation
   * @param alignment allocation alignment in bytes
   * @param mem_type type of the allocation: Host, Device, Shared
   * @return void* pointer to allocated memory
   */
  void* allocate(size_t size, size_t alignment, CHIPMemoryType mem_type);

  /**
   * @brief Allocate data. Pure virtual function - to be overriden by each
   * backend. This member function is the one that's called by all the
   * publically visible CHIPContext::allocate() variants
   *
   * @param size size of the allocation.
   * @param alignment allocation alignment in bytes
   * @param mem_type type of the allocation: Host, Device, Shared
   * @return void*
   */
  virtual void* allocate_(size_t size, size_t alignment,
                          CHIPMemoryType mem_type) = 0;

  /**
   * @brief Free memory
   *
   * @param ptr pointer to the memory location to be deallocated. Internally
   * calls CHIPContext::free_()
   * @return true Success
   * @return false Failure
   */
  hipError_t free(void* ptr);

  /**
   * @brief Free memory
   * To be overriden by the backend
   *
   * @param ptr
   * @return true
   * @return false
   */
  virtual void free_(void* ptr) = 0;

  /**
   * @brief Copy memory
   *
   * @param dst destination
   * @param src source
   * @param size size of the copy
   * @param stream queue to which this copy should be submitted to
   * @return hipError_t
   */
  virtual hipError_t memCopy(void* dst, const void* src, size_t size,
                             hipStream_t stream) {
    UNIMPLEMENTED(hipSuccess);
  };

  /**
   * @brief Finish all the queues in this context
   *
   */
  void finishAll();

  /**
   * @brief For a given device pointer, return the base address of the
   * allocation to which it belongs to along with the allocation size
   *
   * @param pbase device base pointer to which dptr belongs to
   * @param psize size of the allocation with which pbase was created
   * @param dptr device pointer
   * @return hipError_t
   */
  virtual hipError_t findPointerInfo(hipDeviceptr_t* pbase, size_t* psize,
                                     hipDeviceptr_t dptr);

  /**
   * @brief Get the flags set on this context
   *
   * @return unsigned int context flags
   */
  unsigned int getFlags();

  /**
   * @brief Set the flags for this context
   *
   * @param flags flags to set on this context
   */
  void setFlags(unsigned int flags);

  /**
   * @brief Reset this context.
   *
   */
  void reset();

  /**
   * @brief Retain this context.
   * TODO: What does it mean to retain a context?
   *
   * @return CHIPContext*
   */
  CHIPContext* retain();

  /**
   * @brief Create a Event object
   *
   * @param flags
   * @return CHIPEvent*
   */
  virtual CHIPEvent* createEvent(unsigned flags) = 0;
};

/**
 * @brief Primary object to interact with the backend
 */
class CHIPBackend {
 protected:
  /**
   * @brief chip_modules stored in binary representation.
   * During compilation each translation unit is parsed for functions that are
   * marked for execution on the device. These functions are then compiled to
   * device code and stored in binary representation.
   *  */
  std::vector<std::string*> modules_str;
  std::mutex mtx;

  CHIPContext* active_ctx;
  CHIPDevice* active_dev;
  CHIPQueue* active_q;

 public:
  std::mutex callback_stack_mtx;
  std::stack<CHIPCallbackData*> callback_stack;
  /**
   * @brief Keep track of pointers allocated on the device. Used to get info
   * about allocaitons based on device poitner in case that findPointerInfo() is
   * not overriden
   *
   */
  // Adds -std=c++17 requirement
  inline static thread_local hipError_t tls_last_error = hipSuccess;
  inline static thread_local CHIPContext* tls_active_ctx;

  std::stack<CHIPExecItem*> chip_execstack;
  std::vector<CHIPContext*> chip_contexts;
  std::vector<CHIPQueue*> chip_queues;
  std::vector<CHIPDevice*> chip_devices;

  // TODO
  // key for caching compiled modules. To get a cached compiled module on a
  // particular device you must make sure that you have a module which matches
  // the host funciton pointer and also that this module was compiled for the
  // same device model.
  // typedef  std::pair<const void*, std::string> ptr_dev;
  // /**
  //  * @brief
  //  *
  //  */
  // std::unordered_map<ptr_dev, CHIPModule*> host_f_ptr_to_chipmodule_map;

  /**
   * @brief Construct a new CHIPBackend object
   *
   */
  CHIPBackend();
  /**
   * @brief Destroy the CHIPBackend objectk
   *
   */
  ~CHIPBackend();

  /**
   * @brief Initialize this backend with given environment flags
   *
   * @param platform_str
   * @param device_type_str
   * @param device_ids_str
   */
  void initialize(std::string platform_str, std::string device_type_str,
                  std::string device_ids_str);

  /**
   * @brief Initialize this backend with given environment flags
   *
   * @param platform_str
   * @param device_type_str
   * @param device_ids_str
   */
  virtual void initialize_(std::string platform_str,
                           std::string device_type_str,
                           std::string device_ids_str) = 0;

  /**
   * @brief
   *
   */
  virtual void uninitialize() = 0;

  /**
   * @brief Get the Queues object
   *
   * @return std::vector<CHIPQueue*>&
   */
  std::vector<CHIPQueue*>& getQueues();
  /**
   * @brief Get the Active Queue object
   *
   * @return CHIPQueue*
   */
  CHIPQueue* getActiveQueue();
  /**
   * @brief Get the Active Context object. Returns the context of the active
   * queue.
   *
   * @return CHIPContext*
   */
  CHIPContext* getActiveContext();
  /**
   * @brief Get the Active Device object. Returns the device of the active
   * queue.
   *
   * @return CHIPDevice*
   */
  CHIPDevice* getActiveDevice();
  /**
   * @brief Set the active device. Sets the active queue to this device's
   * first/default/primary queue.
   *
   * @param chip_dev
   */
  void setActiveDevice(CHIPDevice* chip_dev);

  std::vector<CHIPDevice*>& getDevices();
  /**
   * @brief Get the Num Devices object
   *
   * @return size_t
   */
  size_t getNumDevices();
  /**
   * @brief Get the vector of registered modules (in string/binary format)
   *
   * @return std::vector<std::string*>&
   */
  std::vector<std::string*>& getModulesStr();
  /**
   * @brief Add a context to this backend.
   *
   * @param ctx_in
   */
  void addContext(CHIPContext* ctx_in);
  /**
   * @brief Add a context to this backend.
   *
   * @param q_in
   */
  void addQueue(CHIPQueue* q_in);
  /**
   * @brief  Add a device to this backend.
   *
   * @param dev_in
   */
  void addDevice(CHIPDevice* dev_in);
  /**
   * @brief
   *
   * @param mod_str
   */
  void registerModuleStr(std::string* mod_str);
  /**
   * @brief
   *
   * @param mod_str
   */
  void unregisterModuleStr(std::string* mod_str);
  /**
   * @brief Configure an upcoming kernel call
   *
   * @param grid
   * @param block
   * @param shared
   * @param q
   * @return hipError_t
   */
  hipError_t configureCall(dim3 grid, dim3 block, size_t shared, hipStream_t q);
  /**
   * @brief Set the Arg object
   *
   * @param arg
   * @param size
   * @param offset
   * @return hipError_t
   */
  hipError_t setArg(const void* arg, size_t size, size_t offset);

  /**
   * @brief Register this function as a kernel for all devices initialized
   * in this backend
   *
   * @param module_str
   * @param host_f_ptr
   * @param host_f_name
   * @return true
   * @return false
   */
  virtual bool registerFunctionAsKernel(std::string* module_str,
                                        const void* host_f_ptr,
                                        const char* host_f_name);

  /**
   * @brief Return a device which meets or exceeds the requirements
   *
   * @param props
   * @return CHIPDevice*
   */
  CHIPDevice* findDeviceMatchingProps(const hipDeviceProp_t* props);

  /**
   * @brief Find a given queue in this backend.
   *
   * @param q queue to find
   * @return CHIPQueue* return queue or nullptr if not found
   */
  CHIPQueue* findQueue(CHIPQueue* q);

  /**
   * @brief Add a CHIPModule to every initialized device
   *
   * @param chip_module pointer to CHIPModule object
   * @return hipError_t
   */
  // CHIPModule* addModule(std::string* module_src);
  /**
   * @brief Remove this module from every device
   *
   * @param chip_module pointer to the module which is to be removed
   * @return hipError_t
   */
  // hipError_t removeModule(CHIPModule* chip_module);

  /************Factories***************/

  virtual CHIPTexture* createCHIPTexture(intptr_t image_,
                                         intptr_t sampler_) = 0;
  virtual CHIPQueue* createCHIPQueue(CHIPDevice* chip_dev) = 0;
  // virtual CHIPDevice* createCHIPDevice(CHIPContext* ctx_) = 0;
  // virtual CHIPContext* createCHIPContext() = 0;
  virtual CHIPEvent* createCHIPEvent(CHIPContext* chip_ctx_,
                                     CHIPEventType event_type_) = 0;

  /**
   * @brief Create a Callback Obj object
   * Each backend must implement this function which calls a derived
   * CHIPCallbackData constructor
   * @return CHIPCallbackData* pointer to newly allocated CHIPCallbackData
   * object.
   */
  virtual CHIPCallbackData* createCallbackData(hipStreamCallback_t callback,
                                               void* userData,
                                               CHIPQueue* chip_queue_) = 0;

  virtual CHIPEventMonitor* createEventMonitor() = 0;

  /**
 * @brief Get the Callback object

 * @param callback_data pointer to callback object
 * @return true callback object available
 * @return false callback object not available
 */
  bool getCallback(CHIPCallbackData** callback_data) {
    bool res = false;
    {
      std::lock_guard<std::mutex> Lock(callback_stack_mtx);
      if (this->callback_stack.size()) {
        *callback_data = callback_stack.top();
        callback_stack.pop();

        res = true;
      }
    }

    return res;
  }
};

/**
 * @brief Queue class for submitting kernels to for execution
 */
class CHIPQueue {
 protected:
  std::mutex mtx;
  int priority;
  unsigned int flags;
  /// Device on which this queue will execute
  CHIPDevice* chip_device;
  /// Context to which device belongs to
  CHIPContext* chip_context;

  CHIPEventMonitor* event_monitor = nullptr;

 public:
  /** Keep track of what was the last event submitted to this queue. Required
   * for enforcing proper queue syncronization as per HIP/CUDA API. */
  CHIPEvent* LastEvent;

  /**
   * @brief Construct a new CHIPQueue object
   *
   * @param chip_dev
   */
  CHIPQueue(CHIPDevice* chip_dev);
  /**
   * @brief Construct a new CHIPQueue object
   *
   * @param chip_dev
   * @param flags
   */
  CHIPQueue(CHIPDevice* chip_dev, unsigned int flags);
  /**
   * @brief Construct a new CHIPQueue object
   *
   * @param chip_dev
   * @param flags
   * @param priority
   */
  CHIPQueue(CHIPDevice* chip_dev, unsigned int flags, int priority);
  /**
   * @brief Destroy the CHIPQueue object
   *
   */
  ~CHIPQueue();

  /**
   * @brief Blocking memory copy
   *
   * @param dst Destination
   * @param src Source
   * @param size Transfer size
   * @return hipError_t
   */

  virtual hipError_t memCopy(
      void* dst, const void* src,
      size_t size) = 0;  // Implement using Async with wait?
  /**
   * @brief Non-blocking memory copy
   *
   * @param dst Destination
   * @param src Source
   * @param size Transfer size
   * @return hipError_t
   */
  virtual hipError_t memCopyAsync(void* dst, const void* src, size_t size) = 0;

  /**
   * @brief Blocking memset
   *
   * @param dst
   * @param size
   * @param pattern
   * @param pattern_size
   */
  virtual void memFill(void* dst, size_t size, const void* pattern,
                       size_t pattern_size);

  /**
   * @brief Non-blocking mem set
   *
   * @param dst
   * @param size
   * @param pattern
   * @param pattern_size
   */
  virtual void memFillAsync(void* dst, size_t size, const void* pattern,
                            size_t pattern_size) = 0;

  // The memory copy 2D support
  virtual void memCopy2D(void* dst, size_t dpitch, const void* src,
                         size_t spitch, size_t width, size_t height);

  virtual void memCopy2DAsync(void* dst, size_t dpitch, const void* src,
                              size_t spitch, size_t width, size_t height) = 0;

  // The memory copy 3D support
  virtual void memCopy3D(void* dst, size_t dpitch, size_t dspitch,
                         const void* src, size_t spitch, size_t sspitch,
                         size_t width, size_t height, size_t depth);

  virtual void memCopy3DAsync(void* dst, size_t dpitch, size_t dspitch,
                              const void* src, size_t spitch, size_t sspitch,
                              size_t width, size_t height, size_t depth) = 0;

  // Memory copy to texture object, i.e. image
  virtual void memCopyToTexture(CHIPTexture* texObj, void* src) = 0;
  /**
   * @brief Submit a CHIPExecItem to this queue for execution. CHIPExecItem
   * needs to be complete - contain the kernel and arguments
   *
   * @param exec_item
   * @return hipError_t
   */
  virtual hipError_t launch(CHIPExecItem* exec_item) = 0;

  /**
   * @brief Get the Device obj
   *
   * @return CHIPDevice*
   */

  CHIPDevice* getDevice();
  /**
   * @brief Wait for this queue to finish.
   *
   */

  virtual void finish() = 0;
  /**
   * @brief Check if the queue is still actively executing
   *
   * @return true
   * @return false
   */

  bool query();  // TODO Depends on Events
  /**
   * @brief Get the Priority Range object defining the bounds for
   * hipStreamCreateWithPriority
   *
   * @param lower_or_upper 0 to get lower bound, 1 to get upper bound
   * @return int bound
   */

  int getPriorityRange(int lower_or_upper);  // TODO CHIP
  /**
   * @brief Insert an event into this queue
   *
   * @param e
   * @return true
   * @return false
   */

  virtual void enqueueBarrier(CHIPEvent* eventToSignal,
                              std::vector<CHIPEvent*>* eventsToWaitFor) = 0;

  virtual void enqueueSignal(CHIPEvent* eventToSignal) = 0;
  /**
   * @brief Get the Flags object with which this queue was created.
   *
   * @return unsigned int
   */

  unsigned int getFlags();  // TODO CHIP
  /**
   * @brief Get the Priority object with which this queue was created.
   *
   * @return int
   */

  int getPriority();  // TODO CHIP
  /**
   * @brief Add a callback funciton to be called on the host after the specified
   * stream is done
   *
   * @param callback function pointer for a ballback function
   * @param userData
   * @return true
   * @return false
   */

  bool addCallback(hipStreamCallback_t callback, void* userData);
  /**
   * @brief Insert a memory prefetch
   *
   * @param ptr
   * @param count
   * @return true
   * @return false
   */

  bool memPrefetch(const void* ptr, size_t count);

  /**
   * @brief Launch a kernel on this queue given a host pointer and arguments
   *
   * @param hostFunction
   * @param numBlocks
   * @param dimBlocks
   * @param args
   * @param sharedMemBytes
   * @return true
   * @return false
   */
  bool launchHostFunc(const void* hostFunction, dim3 numBlocks, dim3 dimBlocks,
                      void** args, size_t sharedMemBytes);

  /**
   * @brief
   *
   * @param grid
   * @param block
   * @param sharedMemBytes
   * @param args
   * @param kernel
   * @return hipError_t
   */
  hipError_t launchWithKernelParams(dim3 grid, dim3 block,
                                    unsigned int sharedMemBytes, void** args,
                                    CHIPKernel* kernel);

  /**
   * @brief
   *
   * @param grid
   * @param block
   * @param sharedMemBytes
   * @param extra
   * @param kernel
   * @return hipError_t
   */
  hipError_t launchWithExtraParams(dim3 grid, dim3 block,
                                   unsigned int sharedMemBytes, void** extra,
                                   CHIPKernel* kernel);

  virtual void getBackendHandles(unsigned long* nativeInfo, int* size) = 0;

  CHIPContext* getContext() { return chip_context; }
};

#endif
