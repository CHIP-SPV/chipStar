/*
Copyright (c) 2023 chipStar developers.

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

// Hosts a register for managing embedded device binary sources.

#ifndef SRC_SPVREGISTER_HH
#define SRC_SPVREGISTER_HH

#include "Utils.hh"

#include <string_view>
#include <memory>
#include <set>
#include <unordered_map>
#include <map>
#include <optional>
#include <cassert>
#include <list>
#include <mutex>

class SPVModule;

/// Represents a host pointer used for pointing host side shadow
/// __global__, __device__ and __constant__ objects.
///
/// Shadow object: A dummy object that clang generates on the host
/// side.
struct HostPtr {
  const void *Value;

  template <typename T> explicit HostPtr(T *TheValue) : Value(TheValue) {}

  operator const void *() const { return Value; }
};

/// Represents a function or global variable in a module.
struct SPVGlobalObject {
  /// The module the object is associated with.
  SPVModule *Parent = nullptr;
  HostPtr Ptr;           ///< The host-pointer the entity is associated with.
  std::string Name; ///< The name of the entity in the device.
};

struct SPVFunction : public SPVGlobalObject {};

struct SPVVariable : public SPVGlobalObject {
  size_t Size; ///< The size of the variable.
};

class SPVModule {
  friend class SPVRegister;

  /// The original source given by a client in binary format
  std::string_view OriginalBinary_;

  /// Post-processed, finalized source. It's empty if the
  /// post-processing step has not been performed (yet).
  std::string FinalizedBinary_;

public:
  // Using lists for iterator stability.
  std::list<SPVFunction> Kernels;
  std::list<SPVVariable> Variables;
  /// True if the module has flag variable for signaling device side abort.
  bool HasAbortFlag = false;

  std::string_view getBinary() const {
    assert(FinalizedBinary_.size() && "Has not finalized yet!");
    return FinalizedBinary_;
  }
};

class SPVRegister {
private:
  std::mutex Mtx_; ///< Mutex for all of the members of this class

  std::set<std::unique_ptr<SPVModule>, PointerCmp<SPVModule>> Sources_;
  std::unordered_map<const void *, SPVGlobalObject *> HostPtrLookup_;

public:
  // the PowerVR OpenCL implementation for some reason demangles SPIR-V function
  // names, e.g. a SPIRV with a "_Z8testfunc" kernel turned into a cl_program
  // returns "testfunc" in clGetProgramInfo(CL_PROGRAM_KERNEL_NAMES, ...)
  // This is a major issue not only for finding kernels, but also potential
  // name conflicts with function overloads.
  bool PreventNameDemangling;

  /// A handle for an incomplete SPIR-V module used in the registration
  /// process. Contents of it are not meant to be accessed by clients.
  struct Handle {
    void *Module;
  };

  Handle registerSource(std::string_view SourceModule);

  void bindFunction(Handle Handle, HostPtr Ptr, const char *Name);
  void bindVariable(Handle Handle, HostPtr Ptr, const std::string &Name,
                    size_t Size);

  void unregisterSource(Handle Src);
  void unregisterSource(const SPVModule *Src);

  const SPVModule *getSource(Handle Src);
  const SPVModule *getSource(HostPtr Ptr);

  size_t getNumSources() const { return Sources_.size(); }

private:
  SPVModule *getFinalizedSource(SPVModule *Src);
};

SPVRegister &getSPVRegister();

#endif // SRC_SPVREGISTER_HH
