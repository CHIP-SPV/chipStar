// Reproduces #2269: ~Backend erased from ChipContexts while iterating it,
// leaking one context and deleting another twice.
#include "CHIPBackend.hh"
#include "CHIPDriver.hh"
#include <cstdio>

static int Destroyed = 0;

struct FakeContext : chipstar::Context {
  ~FakeContext() override { ++Destroyed; }
  void *allocateImpl(size_t, size_t, hipMemoryType,
                     chipstar::HostAllocFlags) override {
    return nullptr;
  }
  void importHostMemory(void *, size_t) override {}
  void releaseHostMemory(void *) override {}
  bool isAllocatedPtrMappedToVM(void *) override { return false; }
  void freeImpl(void *) override {}
};

struct FakeBackend : chipstar::Backend {
  chipstar::ExecItem *createExecItem(dim3, dim3, size_t, hipStream_t) override {
    return nullptr;
  }
  std::string getDefaultJitFlags() override { return ""; }
  int ReqNumHandles() override { return 0; }
  void initializeImpl() override {}
  void initializeFromNative(const uintptr_t *, int) override {}
  void uninitialize() override {}
  chipstar::Queue *createCHIPQueue(chipstar::Device *) override {
    return nullptr;
  }
  std::shared_ptr<chipstar::Event> createEventShared(chipstar::Context *,
                                                     chipstar::EventFlags,
                                                     std::string) override {
    return nullptr;
  }
  chipstar::Event *createEvent(chipstar::Context *,
                               chipstar::EventFlags) override {
    return nullptr;
  }
  chipstar::CallbackData *createCallbackData(hipStreamCallback_t, void *,
                                             chipstar::Queue *) override {
    return nullptr;
  }
  chipstar::EventMonitor *createEventMonitor_() override { return nullptr; }
  hipEvent_t getHipEvent(void *) override { return nullptr; }
  void *getNativeEvent(hipEvent_t) override { return nullptr; }
};

int main() {
  ::Backend = new FakeBackend;
  ::Backend->addContext(new FakeContext);
  ::Backend->addContext(new FakeContext);
  delete ::Backend;
  ::Backend = nullptr;
  if (Destroyed != 2) {
    printf("FAIL: %d context destructor runs, expected 2\n", Destroyed);
    return 1;
  }
  printf("PASS\n");
  return 0;
}
