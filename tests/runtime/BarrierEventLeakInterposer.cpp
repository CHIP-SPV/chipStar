// LD_PRELOAD interposer for TestFix1543SwitchModeEventLeak.
//
// Tracks the reference balance of every cl_event handed out by
// clEnqueueBarrierWithWaitList: the enqueue itself counts as one reference,
// clRetainEvent adds one and clReleaseEvent removes one, and an entry is
// forgotten once its count reaches zero. At process exit it prints how many
// barrier events were observed and how many still have references, which is
// the number of barrier events that leaked.
//
// Every entry point chains to the real implementation, so preloading this
// library only adds bookkeeping. The chaining prototypes take plain integers
// and pointers instead of the OpenCL types: each argument is in the same ABI
// class as the real one, and using them keeps the interposer buildable whether
// or not the OpenCL headers are present.
//
// The report runs from a shared library destructor so that it comes after
// chipStar's own teardown, which is an atexit handler; a balance taken before
// that would count events the runtime is still about to release.

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>

namespace {

// Function-local statics so the table is usable from the first interposed call
// regardless of static initialisation order. They are heap allocated and never
// freed: a static object would be destroyed by its __cxa_atexit handler before
// the library destructor that reads it runs.
std::mutex &tableMutex() {
  static std::mutex *Mutex = new std::mutex;
  return *Mutex;
}

std::map<void *, int> &barrierRefs() {
  static std::map<void *, int> *Table = new std::map<void *, int>;
  return *Table;
}

unsigned &numObserved() {
  static unsigned Count = 0;
  return Count;
}

template <typename FnTy> FnTy realSymbol(const char *Name) {
  void *Symbol = dlsym(RTLD_NEXT, Name);
  if (!Symbol) {
    // Reached only if something calls an interposed entry point that the real
    // runtime does not provide. Say so instead of jumping through a null.
    std::fprintf(stderr, "interposer: no real %s to chain to\n", Name);
    std::fflush(stderr);
    std::abort();
  }
  return reinterpret_cast<FnTy>(Symbol);
}

// Child processes must not inherit the interposer: pocl compiles kernels
// through clang and lld subprocesses, and each of them would print an empty
// report of its own at exit.
__attribute__((constructor)) void dropFromChildren() { unsetenv("LD_PRELOAD"); }

__attribute__((destructor)) void reportAtExit() {
  std::lock_guard<std::mutex> Lock(tableMutex());
  unsigned Leaked = 0;
  for (const auto &Entry : barrierRefs())
    if (Entry.second > 0)
      ++Leaked;
  std::fprintf(stderr, "interposer: barrier events observed: %u, leaked: %u\n",
               numObserved(), Leaked);
  std::fflush(stderr);
}

} // namespace

extern "C" {

int clEnqueueBarrierWithWaitList(void *Queue, unsigned NumEvents,
                                 const void *WaitList, void **Event) {
  using FnTy = int (*)(void *, unsigned, const void *, void **);
  int Status = realSymbol<FnTy>("clEnqueueBarrierWithWaitList")(
      Queue, NumEvents, WaitList, Event);
  if (Status == 0 && Event && *Event) {
    std::lock_guard<std::mutex> Lock(tableMutex());
    barrierRefs()[*Event] = 1;
    ++numObserved();
  }
  return Status;
}

int clRetainEvent(void *Event) {
  using FnTy = int (*)(void *);
  int Status = realSymbol<FnTy>("clRetainEvent")(Event);
  if (Status == 0) {
    std::lock_guard<std::mutex> Lock(tableMutex());
    auto It = barrierRefs().find(Event);
    if (It != barrierRefs().end())
      ++It->second;
  }
  return Status;
}

int clReleaseEvent(void *Event) {
  {
    // Drop the entry before the real release so a handle the runtime recycles
    // for a later barrier starts from a fresh count.
    std::lock_guard<std::mutex> Lock(tableMutex());
    auto It = barrierRefs().find(Event);
    if (It != barrierRefs().end() && --It->second == 0)
      barrierRefs().erase(It);
  }
  using FnTy = int (*)(void *);
  return realSymbol<FnTy>("clReleaseEvent")(Event);
}

} // extern "C"
