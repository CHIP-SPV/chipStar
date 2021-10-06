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

 public:
  /**
   * @brief HIPxxEvent constructor. Must always be created with some context.
   *
   */
  HIPxxEvent(HIPxxContext* ctx_,
             HIPxxEventType flags_ = HIPxxEventType::Default);
  /**
   * @brief Deleted default constructor for HIPxxEvent
   *
   */
  HIPxxEvent() = delete;
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

 public:
  /**
   * @brief Deleted default constuctor
   *
   */
  HIPxxModule() = delete;
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
   */
  void compileOnce();
  /**
   * @brief Kernel JIT compilation can be lazy. This is configured via Cmake
   * LAZY_JIT option. If LAZY_JIT is set to true then this module won't be
   * compiled until the first call to one of its kernels. If LAZY_JIT is set to
   * false(default) then this method should be called in the constructor;
   *
   */
  virtual void compile();
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
  /// Name of the function
  std::string host_f_name;
  /// Pointer to the host function
  const void* host_f_ptr;
  /// Pointer to the device function
  const void* dev_f_ptr;

 public:
  HIPxxKernel();
  ~HIPxxKernel();
  std::string getName();
  const void* getHostPtr();
  const void* getDevPtr();
};

/**
 * @brief a HIPxxKernel and argument container to be submitted to HIPxxQueue
 */
class HIPxxExecItem {
 protected:
  size_t shared_mem;
  hipStream_t stream;
  std::vector<uint8_t> arg_data;
  std::vector<std::tuple<size_t, size_t>> offset_sizes;

 public:
  HIPxxKernel* hipxx_kernel;
  HIPxxQueue* hipxx_queue;
  dim3 grid_dim;
  dim3 block_dim;

  HIPxxExecItem(dim3 grid_dim_, dim3 block_dim_, size_t shared_mem_,
                hipStream_t hipxx_queue_);
  ~HIPxxExecItem();

  void setArg(const void* arg, size_t size, size_t offset);
  virtual hipError_t launch(HIPxxKernel* Kernel);
  virtual hipError_t launchByHostPtr(const void* hostPtr);
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
  HIPxxQueue* q;
  int active_queue_id = 0;

  // TODO Implement filling this in. Seems redudant with props
  hipDeviceAttribute_t attrs;

 public:
  /// hipxx_modules in binary representation
  std::vector<std::string*> modules_str;
  /// hipxx_modules in parsed representation
  std::vector<HIPxxModule*> hipxx_modules;

  /// Map host pointer to module in binary representation
  std::unordered_map<const void*, std::string*> host_ptr_to_module_str_map;
  /// Map host pointer to module in parsed representation
  std::unordered_map<const void*, HIPxxModule*> host_ptr_to_hipxxmodule_map;
  /// Map host pointer to a function name
  std::unordered_map<const void*, std::string> host_ptr_to_name_map;
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

  HIPxxDevice();
  ~HIPxxDevice();

  void addKernel(HIPxxKernel* kernel);
  std::vector<HIPxxKernel*>& getKernels();

  /**
   * @brief Use a backend to populate device properties such as memory
   * available, frequencies, etc.
   */
  virtual void populateDeviceProperties() = 0;
  void copyDeviceProperties(hipDeviceProp_t* prop);

  HIPxxKernel* findKernelByHostPtr(const void* hostPtr);

  /**
   * @brief Get the context object
   *
   * @return HIPxxContext* pointer to the HIPxxContext object this HIPxxDevice
   * was created with
   */
  HIPxxContext* getContext();
  HIPxxQueue* addQueue(unsigned int flags,
                       int priority);    // TODO HIPxx
  std::vector<HIPxxQueue*> getQueues();  // TODO HIPxx
  HIPxxQueue* getActiveQueue();          // TODO HIPxx
  bool removeQueue(HIPxxQueue* q);       // TODO HIPxx

  int getDeviceId();
  virtual std::string getName() = 0;

  bool getModuleAndFName(const void* host_f_ptr, std::string& host_f_name,
                         HIPxxModule* hipxx_module);
  bool allocate(size_t bytes);
  bool free(size_t bytes);

  virtual void reset() = 0;
  int getAttr(int* pi, hipDeviceAttribute_t attr);
  size_t getGlobalMemSize();  // TODO HIPxx
  /*virtual*/ void setCacheConfig(hipFuncCache_t);
  /* = 0;*/                         // TODO HIPxx
  hipFuncCache_t getCacheConfig();  // TODO HIPxx
  /*virtual*/ void setSharedMemConfig(hipSharedMemConfig config);
  /* = 0;*/                                 // TODO HIPxx
  hipSharedMemConfig getSharedMemConfig();  // TODO HIPxx
  void setFuncCacheConfig(const void* func,
                          hipFuncCache_t config);  // TODO HIPxx
  // Check if the current device has same PCI bus ID as the
  // one given by input
  bool hasPCIBusId(int pciDomainID, int pciBusID,
                   int pciDeviceID);           // TODO HIPxx
  int getPeerAccess(HIPxxDevice* peerDevice);  // TODO HIPxx

  // Enable/Disable the peer access
  // from given devince
  hipError_t setPeerAccess(HIPxxDevice* peer, int flags,
                           bool canAccessPeer);  // TODO HIPxx

  size_t getUsedGlobalMem();  // TODO HIPxx

  HIPxxDeviceVar* getDynGlobalVar(const void* host_var_ptr);   // TODO HIPxx
  HIPxxDeviceVar* getStatGlobalVar(const void* host_var_ptr);  // TODO HIPxx
  HIPxxDeviceVar* getGlobalVar(const void* host_var_ptr);      // TODO HIPxx
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

  virtual bool registerFunctionAsKernel(std::string* module_str,
                                        const void* HostFunctionPtr,
                                        const char* FunctionName) = 0;
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

  HIPxxBackend();
  ~HIPxxBackend();

  virtual void initialize(std::string platform_str, std::string device_type_str,
                          std::string device_ids_str);
  virtual void initialize() = 0;
  virtual void uninitialize() = 0;

  std::vector<HIPxxQueue*>& getQueues();
  HIPxxQueue* getActiveQueue();
  HIPxxContext* getActiveContext();
  HIPxxDevice* getActiveDevice();
  void setActiveDevice(HIPxxDevice* hipxx_dev);

  std::vector<HIPxxDevice*>& getDevices();
  size_t getNumDevices();
  std::vector<std::string*>& getModulesStr();
  void addContext(HIPxxContext* ctx_in);
  void addQueue(HIPxxQueue* q_in);
  void addDevice(HIPxxDevice* dev_in);
  void registerModuleStr(std::string* mod_str);
  void unregisterModuleStr(std::string* mod_str);
  hipError_t configureCall(dim3 grid, dim3 block, size_t shared, hipStream_t q);
  hipError_t setArg(const void* arg, size_t size, size_t offset);

  /**
   * @brief Register this function as a kernel for all devices initialized in
   * this backend
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

  HIPxxDevice* findDeviceMatchingProps(
      const hipDeviceProp_t* props);  // HIPxx TODO

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

 public:
  /// Device on which this queue will execute
  HIPxxDevice* hipxx_device;
  /// Context to which device belongs to
  HIPxxContext* hipxx_context;

  // TODO these should take device and context as arguments.
  HIPxxQueue();
  ~HIPxxQueue();

  virtual hipError_t memCopy(void* dst, const void* src, size_t size);
  virtual hipError_t memCopyAsync(void* dst, const void* src, size_t size);

  virtual void memFill(void* dst, size_t size, const void* pattern,
                       size_t pattern_size);
  virtual void memFillAsync(void* dst, size_t size, const void* pattern,
                            size_t pattern_size);

  /// Submit a kernel for execution
  virtual hipError_t launch(HIPxxExecItem* exec_item) = 0;

  HIPxxDevice* getDevice();
  virtual void finish() = 0;
  bool query();                                                    // TODO HIPxx
  int getPriorityRange(int lower_or_upper);                        // TODO HIPxx
  bool enqueueBarrierForEvent(HIPxxEvent* e);                      // TODO HIPxx
  unsigned int getFlags();                                         // TODO HIPxx
  int getPriority();                                               // TODO HIPxx
  bool addCallback(hipStreamCallback_t callback, void* userData);  // TODO HIPxx
  bool memPrefetch(const void* ptr, size_t count);                 // TODO HIPxx
  bool launchHostFunc(const void* hostFunction, dim3 numBlocks, dim3 dimBlocks,
                      void** args, size_t sharedMemBytes);  // TODO HIPxx
  hipError_t launchWithKernelParams(dim3 grid, dim3 block,
                                    unsigned int sharedMemBytes, void** args,
                                    HIPxxKernel* kernel);
  hipError_t launchWithExtraParams(dim3 grid, dim3 block,
                                   unsigned int sharedMemBytes, void** extra,
                                   HIPxxKernel* kernel);
};

#endif
