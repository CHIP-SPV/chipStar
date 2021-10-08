/**
 * @file HIPxxBackend.hh
 * @author Paulius Velesko (pvelesko@gmail.com)
 * @brief HIPxxBackend class definition. HIPxx backends are to inherit from this
 * base class and override desired virtual functions. Overrides for this class
 * are expected to be minimal with primary overrides being done on lower-level
 * classes such as HIPxxContext consturctors, etc.
 * @version 0.1
 * @date 2021-08-19
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef HIPXX_BACKEND_H
#define HIPXX_BACKEND_H

#include <algorithm>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <stack>

#include "spirv.hh"
#include "include/hip/hip.hh"

#include "HIPxxDriver.hh"
#include "logging.hh"
#include "macros.hh"

enum class HIPxxMemoryType : unsigned { Host = 0, Device = 1, Shared = 2 };
enum class HIPxxEventType : unsigned {
  Default = hipEventDefault,
  BlockingSync = hipEventBlockingSync,
  DisableTiming = hipEventDisableTiming,
  Interprocess = hipEventInterprocess
};

class HIPxxDeviceVar {
 private:
  std::string host_var_name;
  void* dev_ptr;
  size_t size;

 public:
  HIPxxDeviceVar(std::string host_var_name_, void* dev_ptr_, size_t size);
  ~HIPxxDeviceVar();

  void* getDevAddr();
  std::string getName();
  size_t getSize();
};

// fw declares
class HIPxxExecItem;
class HIPxxQueue;
class HIPxxContext;
class HIPxxDevice;

class HIPxxEvent {
 protected:
  std::mutex mutex;
  event_status_e status;
  /**
   * @brief event bahavior modifier -  valid values are hipEventDefault,
   * hipEventBlockingSync, hipEventDisableTiming, hipEventInterprocess
   *
   */
  HIPxxEventType flags;
  /**
   * @brief Events are always created with a context
   *
   */
  HIPxxContext* hipxx_context;

  /**
   * @brief hidden default constructor for HIPxxEvent. Only derived class
   * constructor should be called.
   *
   */
  HIPxxEvent() = default;

 public:
  /**
   * @brief HIPxxEvent constructor. Must always be created with some context.
   *
   */
  HIPxxEvent(HIPxxContext* ctx_,
             HIPxxEventType flags_ = HIPxxEventType::Default);
  /**
   * @brief Destroy the HIPxxEvent object
   *
   */
  ~HIPxxEvent();
  /**
   * @brief Enqueue this event in a given HIPxxQueue
   *
   * @param hipxx_queue_ HIPxxQueue in which to enque this event
   * @return true
   * @return false
   */
  virtual bool recordStream(HIPxxQueue* hipxx_queue_);
  /**
   * @brief Wait for this event to complete
   *
   * @return true
   * @return false
   */
  virtual bool wait();
  /**
   * @brief Query the event to see if it completed
   *
   * @return true
   * @return false
   */
  virtual bool isFinished();
  /**
   * @brief Calculate absolute difference between completion timestamps of this
   * event and other
   *
   * @param other
   * @return float
   */
  virtual float getElapsedTime(HIPxxEvent* other);
};

/**
 * @brief Module abstraction. Contains global variables and kernels. Can be
 * extracted from FatBinary or loaded at runtime.
 * OpenCL - ClProgram
 * Level Zero - zeModule
 * ROCclr - amd::Program
 * CUDA - CUmodule
 */
class HIPxxModule {
 protected:
  std::mutex mtx;
  // Global variables
  std::vector<HIPxxDeviceVar*> hipxx_vars;
  // Kernels
  std::vector<HIPxxKernel*> hipxx_kernels;
  /// Binary representation extracted from FatBinary
  std::string src;
  // Kernel JIT compilation can be lazy
  std::once_flag compiled;

  /**
   * @brief hidden default constuctor. Only derived type constructor should be
   * called.
   *
   */
  HIPxxModule() = default;

 public:
  /**
   * @brief Destroy the HIPxxModule object
   *
   */
  ~HIPxxModule();
  /**
   * @brief Construct a new HIPxxModule object.
   * This constructor should be implemented by the derived class (specific
   * backend implementation). Call to this constructor should result in a
   * populated hipxx_kernels vector.
   *
   * @param module_str string prepresenting the binary extracted from FatBinary
   */
  HIPxxModule(std::string* module_str);
  /**
   * @brief Construct a new HIPxxModule object using move semantics
   *
   * @param module_str string from which to move resources
   */
  HIPxxModule(std::string&& module_str);

  /**
   * @brief Add a HIPxxKernel to this module.
   * During initialization when the FatBinary is consumed, a HIPxxModule is
   * constructed for every device. SPIR-V kernels reside in this module. This
   * method is called called via the constructor during this initialization
   * phase. Modules can also be loaded from a file during runtime, however.
   *
   * @param kernel HIPxxKernel to be added to this module.
   */
  void addKernel(HIPxxKernel* kernel);

  /**
   * @brief Wrapper around compile() called via std::call_once
   *
   * @param hipxx_dev device for which to compile the kernels
   */
  void compileOnce(HIPxxDevice* hipxx_dev);
  /**
   * @brief Kernel JIT compilation can be lazy. This is configured via Cmake
   * LAZY_JIT option. If LAZY_JIT is set to true then this module won't be
   * compiled until the first call to one of its kernels. If LAZY_JIT is set to
   * false(default) then this method should be called in the constructor;
   *
   * This method should populate this modules hipxx_kernels vector. These
   * kernels would have a name extracted from the kernel but no associated host
   * function pointers.
   *
   */
  virtual void compile(HIPxxDevice* hipxx_dev);
  /**
   * @brief Get the Global Var object
   * A module, along with device kernels, can also contain global variables.
   *
   * @param name global variable name
   * @return HIPxxDeviceVar*
   */
  HIPxxDeviceVar* getGlobalVar(std::string name);

  /**
   * @brief Get the Kernel object
   *
   * @param name name of the corresponding host function
   * @return HIPxxKernel*
   */
  HIPxxKernel* getKernel(std::string name);

  /**
   * @brief Get the Kernels object
   *
   * @return std::vector<HIPxxKernel*>&
   */
  std::vector<HIPxxKernel*>& getKernels();

  /**
   * @brief Get the Kernel object
   *
   * @param host_f_ptr host-side function pointer
   * @return HIPxxKernel*
   */
  HIPxxKernel* getKernel(const void* host_f_ptr);
};

/**
 * @brief Contains information about the function on the host and device
 */
class HIPxxKernel {
 protected:
  /**
   * @brief hidden default constructor. Only derived type constructor should be
   * called.
   *
   */
  HIPxxKernel() = default;
  /// Name of the function
  std::string host_f_name;
  /// Pointer to the host function
  const void* host_f_ptr;
  /// Pointer to the device function
  const void* dev_f_ptr;

 public:
  ~HIPxxKernel();

  /**
   * @brief Get the Name object
   *
   * @return std::string
   */
  std::string getName();
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
 * HIPxxBackend::configureCall(). Because of this, we get the kernel last so the
 * kernel so the launch() takes a kernel argument as opposed to queue receiving
 * a HIPxxExecItem containing the kernel and arguments
 *
 */
class HIPxxExecItem {
 protected:
  size_t shared_mem;
  std::vector<uint8_t> arg_data;
  std::vector<std::tuple<size_t, size_t>> offset_sizes;

  dim3 grid_dim;
  dim3 block_dim;

  HIPxxQueue* stream;
  HIPxxKernel* hipxx_kernel;
  HIPxxQueue* hipxx_queue;

 public:
  /**
   * @brief Deleted default constructor
   * Doesn't make sense for HIPxxExecItem to exist without arguments
   *
   */
  HIPxxExecItem() = delete;
  /**
   * @brief Construct a new HIPxxExecItem object
   *
   * @param grid_dim_
   * @param block_dim_
   * @param shared_mem_
   * @param hipxx_queue_
   */
  HIPxxExecItem(dim3 grid_dim_, dim3 block_dim_, size_t shared_mem_,
                hipStream_t hipxx_queue_);
  ~HIPxxExecItem();

  /**
   * @brief Get the Kernel object
   *
   * @return HIPxxKernel* Kernel to be executed
   */
  HIPxxKernel* getKernel();
  /**
   * @brief Get the Queue object
   *
   * @return HIPxxQueue*
   */
  HIPxxQueue* getQueue();

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
   * @brief Setup a single argument.
   * gets called by hipSetupArgument calls to which are emitted by hip-clang.
   *
   * @param arg
   * @param size
   * @param offset
   */
  void setArg(const void* arg, size_t size, size_t offset);

  /**
   * @brief Submit a kernel to the associated queue for execution.
   * hipxx_queue must be set prior to this call.
   *
   * @param Kernel kernel which is to be launched
   * @return hipError_t possible values: hipSuccess, hipErrorLaunchFailure
   */
  virtual hipError_t launch(HIPxxKernel* Kernel);

  /**
   * @brief Launch a kernel associated with a host function pointer.
   * Looks up the HIPxxKernel associated with this pointer and calls launch()
   *
   * @param hostPtr pointer to the host function
   * @return hipError_t possible values: hipSuccess, hipErrorLaunchFailure
   */
  hipError_t launchByHostPtr(const void* hostPtr);
};

/**
 * @brief Compute device class
 */
class HIPxxDevice {
 protected:
  std::string device_name;
  std::mutex mtx;
  std::vector<HIPxxKernel*> hipxx_kernels;
  HIPxxContext* ctx;
  std::vector<HIPxxQueue*> hipxx_queues;
  int active_queue_id = 0;

  // TODO Implement filling this in. Seems redudant with props
  hipDeviceAttribute_t attrs;

 public:
  /// hipxx_modules in binary representation
  std::vector<std::string*> modules_str;
  /// hipxx_modules in parsed representation
  std::vector<HIPxxModule*> hipxx_modules;

  /// Map host pointer to module in binary representation
  std::unordered_map<const void*, std::string*> host_f_ptr_to_module_str_map;
  /// Map host pointer to module in parsed representation
  std::unordered_map<const void*, HIPxxModule*> host_f_ptr_to_hipxxmodule_map;
  /// Map host pointer to a function name
  std::unordered_map<const void*, std::string> host_f_ptr_to_host_f_name_map;
  /// Map host pointer to HIPxxKernel
  std::unordered_map<const void*, HIPxxKernel*> host_ptr_to_hipxxkernel_map;
  /// Map host variable address to device pointer and size for statically loaded
  /// global vars
  std::unordered_map<const void*, HIPxxDeviceVar*>
      host_var_ptr_to_hipxxdevicevar_stat;
  /// Map host variable address to device pointer and size for dynamically
  /// loaded global vars
  std::unordered_map<const void*, HIPxxDeviceVar*>
      host_var_ptr_to_hipxxdevicevar_dyn;

  int idx;
  hipDeviceProp_t hip_device_props;
  size_t total_used_mem, max_used_mem;
  /**
   * @brief Construct a new HIPxxDevice object
   *
   */
  HIPxxDevice();
  /**
   * @brief Destroy the HIPxxDevice object
   *
   */
  ~HIPxxDevice();

  /**
   * @brief Get the Kernels object
   *
   * @return std::vector<HIPxxKernel*>&
   */
  std::vector<HIPxxKernel*>& getKernels();

  /**
   * @brief Use a backend to populate device properties such as memory
   * available, frequencies, etc.
   */
  virtual void populateDeviceProperties() = 0;
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
   * @return HIPxxKernel* HIPxxKernel associated with this host pointer
   */
  HIPxxKernel* findKernelByHostPtr(const void* hostPtr);

  /**
   * @brief Get the context object
   *
   * @return HIPxxContext* pointer to the HIPxxContext object this HIPxxDevice
   * was created with
   */
  HIPxxContext* getContext();
  /**
   * @brief Construct an additional queue for this device
   *
   * @param flags
   * @param priority
   * @return HIPxxQueue* pointer to the newly created queue (can also be found
   * in hipxx_queues vector)
   */
  void addQueue(HIPxxQueue* hipxx_queue_);

  /**
   * @brief Get the Queues object
   *
   * @return std::vector<HIPxxQueue*>
   */
  std::vector<HIPxxQueue*> getQueues();  // TODO HIPxx
  /**
   * @brief HIP API allows for setting the active device, not the active queue
   * so active device's active queue is always it's 0th/default/primary queue
   *
   * @return HIPxxQueue*
   */
  HIPxxQueue* getActiveQueue();  // TODO HIPxx
  /**
   * @brief Remove a queue from this device's queue vector
   *
   * @param q
   * @return true
   * @return false
   */
  bool removeQueue(HIPxxQueue* q);  // TODO HIPxx

  /**
   * @brief Get the integer ID of this device as it appears in the Backend's
   * hipxx_devices list
   *
   * @return int
   */
  int getDeviceId();
  /**
   * @brief Get the device name
   *
   * @return std::string
   */
  virtual std::string getName() = 0;

  bool allocate(size_t bytes);
  bool free(size_t bytes);

  /**
   * @brief Reset the device
   *
   */
  virtual void reset() = 0;

  int getAttr(int* pi, hipDeviceAttribute_t attr);

  /**
   * @brief Get the total global memory available for this device.
   *
   * @return size_t
   */
  size_t getGlobalMemSize();  // TODO HIPxx

  /**
   * @brief
   *
   */
  /*virtual*/ void setCacheConfig(hipFuncCache_t);
  /* = 0;*/                         // TODO HIPxx
  hipFuncCache_t getCacheConfig();  // TODO HIPxx
  /*virtual*/ void setSharedMemConfig(hipSharedMemConfig config);
  /* = 0;*/                                 // TODO HIPxx
  hipSharedMemConfig getSharedMemConfig();  // TODO HIPxx
  void setFuncCacheConfig(const void* func,
                          hipFuncCache_t config);  // TODO HIPxx

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
  bool hasPCIBusId(int pciDomainID, int pciBusID,
                   int pciDeviceID);  // TODO HIPxx

  /**
   * @brief Get peer-accesability between this and another device
   *
   * @param peerDevice
   * @return int
   */
  int getPeerAccess(HIPxxDevice* peerDevice);  // TODO HIPxx

  /**
   * @brief Set access between this and another device
   *
   * @param peer
   * @param flags
   * @param canAccessPeer
   * @return hipError_t
   */
  hipError_t setPeerAccess(HIPxxDevice* peer, int flags,
                           bool canAccessPeer);  // TODO HIPxx

  /**
   * @brief Get the total used global memory
   *
   * @return size_t
   */
  size_t getUsedGlobalMem();  // TODO HIPxx

  HIPxxDeviceVar* getDynGlobalVar(const void* host_var_ptr);   // TODO HIPxx
  HIPxxDeviceVar* getStatGlobalVar(const void* host_var_ptr);  // TODO HIPxx
  HIPxxDeviceVar* getGlobalVar(const void* host_var_ptr);      // TODO HIPxx

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
};

/**
 * @brief Context class
 * Contexts contain execution queues and are created on top of a single or
 * multiple devices. Provides for creation of additional queues, events, and
 * interaction with devices.
 */
class HIPxxContext {
 protected:
  std::vector<HIPxxDevice*> hipxx_devices;
  std::vector<HIPxxQueue*> hipxx_queues;
  std::mutex mtx;

 public:
  HIPxxContext();
  ~HIPxxContext();

  /**
   * @brief Add a device to this context
   *
   * @param dev pointer to HIPxxDevice object
   * @return true if device was added successfully
   * @return false upon failure
   */
  bool addDevice(HIPxxDevice* dev);
  void addQueue(HIPxxQueue* q);
  HIPxxQueue* getDefaultQueue();
  hipStream_t findQueue(hipStream_t stream);
  std::vector<HIPxxDevice*>& getDevices();
  std::vector<HIPxxQueue*>& getQueues();

  virtual void* allocate(size_t size) = 0;
  virtual void* allocate(size_t size, HIPxxMemoryType mem_type);  // TODO HIPxx
  virtual void* allocate(size_t size, size_t alignment,
                         HIPxxMemoryType mem_type);  // TODO HIPxx
  bool free(void* ptr);                              // TODO HIPxx
  virtual hipError_t memCopy(void* dst, const void* src, size_t size,
                             hipStream_t stream);  // TODO HIPxx

  hipError_t launchHostFunc(const void* HostFunction);
  void finishAll();
  bool findPointerInfo(hipDeviceptr_t* pbase, size_t* psize,
                       hipDeviceptr_t dptr);           // TODO HIPxx
  unsigned int getFlags();                             // TODO HIPxx
  bool setFlags(unsigned int flags);                   // TODO HIPxx
  void reset();                                        // TODO HIPxx
  HIPxxContext* retain();                              // TODO HIPxx
  bool recordEvent(HIPxxQueue* q, HIPxxEvent* event);  // TODO HIPxx
  size_t getPointerSize(void* ptr);                    // TODO HIPxx
  HIPxxTexture* createImage(hipResourceDesc* resDesc, hipTextureDesc* texDesc);
};

/**
 * @brief Primary object to interact with the backend
 */
class HIPxxBackend {
 protected:
  /**
   * @brief hipxx_modules stored in binary representation.
   * During compilation each translation unit is parsed for functions that are
   * marked for execution on the device. These functions are then compiled to
   * device code and stored in binary representation.
   *  */
  std::vector<std::string*> modules_str;
  std::mutex mtx;

  HIPxxContext* active_ctx;
  HIPxxDevice* active_dev;
  HIPxxQueue* active_q;

 public:
  // Adds -std=c++17 requirement
  inline static thread_local hipError_t tls_last_error = hipSuccess;
  inline static thread_local HIPxxContext* tls_active_ctx;

  std::stack<HIPxxExecItem*> hipxx_execstack;
  std::vector<HIPxxContext*> hipxx_contexts;
  std::vector<HIPxxQueue*> hipxx_queues;
  std::vector<HIPxxDevice*> hipxx_devices;

  // key for caching compiled modules. To get a cached compiled module on a
  // particular device you must make sure that you have a module which matches
  // the host funciton pointer and also that this module was compiled for the
  // same device model.
  // typedef  std::pair<const void*, std::string> ptr_dev;
  // /**
  //  * @brief
  //  *
  //  */
  // std::unordered_map<ptr_dev, HIPxxModule*> host_f_ptr_to_hipxxmodule_map;

  /**
   * @brief Construct a new HIPxxBackend object
   *
   */
  HIPxxBackend();
  /**
   * @brief Destroy the HIPxxBackend objectk
   *
   */
  ~HIPxxBackend();

  /**
   * @brief
   *
   * @param platform_str
   * @param device_type_str
   * @param device_ids_str
   */
  virtual void initialize(std::string platform_str, std::string device_type_str,
                          std::string device_ids_str);

  /**
   * @brief
   *
   */
  virtual void initialize() = 0;

  /**
   * @brief
   *
   */
  virtual void uninitialize() = 0;

  /**
   * @brief Get the Queues object
   *
   * @return std::vector<HIPxxQueue*>&
   */
  std::vector<HIPxxQueue*>& getQueues();
  /**
   * @brief Get the Active Queue object
   *
   * @return HIPxxQueue*
   */
  HIPxxQueue* getActiveQueue();
  /**
   * @brief Get the Active Context object. Returns the context of the active
   * queue.
   *
   * @return HIPxxContext*
   */
  HIPxxContext* getActiveContext();
  /**
   * @brief Get the Active Device object. Returns the device of the active
   * queue.
   *
   * @return HIPxxDevice*
   */
  HIPxxDevice* getActiveDevice();
  /**
   * @brief Set the active device. Sets the active queue to this device's
   * first/default/primary queue.
   *
   * @param hipxx_dev
   */
  void setActiveDevice(HIPxxDevice* hipxx_dev);

  std::vector<HIPxxDevice*>& getDevices();
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
  void addContext(HIPxxContext* ctx_in);
  /**
   * @brief Add a context to this backend.
   *
   * @param q_in
   */
  void addQueue(HIPxxQueue* q_in);
  /**
   * @brief  Add a device to this backend.
   *
   * @param dev_in
   */
  void addDevice(HIPxxDevice* dev_in);
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
   * @return HIPxxDevice*
   */
  HIPxxDevice* findDeviceMatchingProps(const hipDeviceProp_t* props);

  /**
   * @brief Add a HIPxxModule to every initialized device
   *
   * @param hipxx_module pointer to HIPxxModule object
   * @return hipError_t
   */
  hipError_t addModule(HIPxxModule* hipxx_module);
  /**
   * @brief Remove this module from every device
   *
   * @param hipxx_module pointer to the module which is to be removed
   * @return hipError_t
   */
  hipError_t removeModule(HIPxxModule* hipxx_module);
};

/**
 * @brief Queue class for submitting kernels to for execution
 */
class HIPxxQueue {
 protected:
  std::mutex mtx;
  int priority;
  unsigned int flags;
  /// Device on which this queue will execute
  HIPxxDevice* hipxx_device;
  /// Context to which device belongs to
  HIPxxContext* hipxx_context;

 public:
  /**
   * @brief Construct a new HIPxxQueue object
   *
   * @param hipxx_dev
   */
  HIPxxQueue(HIPxxDevice* hipxx_dev);
  /**
   * @brief Construct a new HIPxxQueue object
   *
   * @param hipxx_dev
   * @param flags
   */
  HIPxxQueue(HIPxxDevice* hipxx_dev, unsigned int flags);
  /**
   * @brief Construct a new HIPxxQueue object
   *
   * @param hipxx_dev
   * @param flags
   * @param priority
   */
  HIPxxQueue(HIPxxDevice* hipxx_dev, unsigned int flags, int priority);
  /**
   * @brief Destroy the HIPxxQueue object
   *
   */
  ~HIPxxQueue();

  /**
   * @brief Blocking memory copy
   *
   * @param dst Destination
   * @param src Source
   * @param size Transfer size
   * @return hipError_t
   */

  virtual hipError_t memCopy(void* dst, const void* src, size_t size);
  /**
   * @brief Non-blocking memory copy
   *
   * @param dst Destination
   * @param src Source
   * @param size Transfer size
   * @return hipError_t
   */
  virtual hipError_t memCopyAsync(void* dst, const void* src, size_t size);

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
                            size_t pattern_size);

  /**
   * @brief Submit a HIPxxExecItem to this queue for execution. HIPxxExecItem
   * needs to be complete - contain the kernel and arguments
   *
   * @param exec_item
   * @return hipError_t
   */
  virtual hipError_t launch(HIPxxExecItem* exec_item);

  /**
   * @brief Get the Device obj
   *
   * @return HIPxxDevice*
   */

  HIPxxDevice* getDevice();
  /**
   * @brief Wait for this queue to finish.
   *
   */

  virtual void finish();
  /**
   * @brief Check if the queue is still actively executing
   *
   * @return true
   * @return false
   */

  bool query();  // TODO HIPxx
  /**
   * @brief Get the Priority Range object defining the bounds for
   * hipStreamCreateWithPriority
   *
   * @param lower_or_upper 0 to get lower bound, 1 to get upper bound
   * @return int bound
   */

  int getPriorityRange(int lower_or_upper);  // TODO HIPxx
  /**
   * @brief Insert an event into this queue
   *
   * @param e
   * @return true
   * @return false
   */

  bool enqueueBarrierForEvent(HIPxxEvent* e);  // TODO HIPxx
  /**
   * @brief Get the Flags object with which this queue was created.
   *
   * @return unsigned int
   */

  unsigned int getFlags();  // TODO HIPxx
  /**
   * @brief Get the Priority object with which this queue was created.
   *
   * @return int
   */

  int getPriority();  // TODO HIPxx
  /**
   * @brief Add a callback funciton to be called on the host after the specified
   * stream is done
   *
   * @param callback function pointer for a ballback function
   * @param userData
   * @return true
   * @return false
   */

  bool addCallback(hipStreamCallback_t callback, void* userData);  // TODO HIPxx
  /**
   * @brief Insert a memory prefetch
   *
   * @param ptr
   * @param count
   * @return true
   * @return false
   */

  bool memPrefetch(const void* ptr, size_t count);  // TODO HIPxx
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
                      void** args, size_t sharedMemBytes);  // TODO HIPxx

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
                                    HIPxxKernel* kernel);

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
                                   HIPxxKernel* kernel);
};

#endif
