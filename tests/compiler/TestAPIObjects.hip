#include <hip/hip_runtime.h>
#include <vector>

// A reduced case from PyTorch.
struct ihipEvent_t; // Forward declaration
std::vector<ihipEvent_t *> testIHipEvent() {
  ihipEvent_t *Event;
  hipEventCreate(&Event);
  std::vector<ihipEvent_t *> Result = {Event};
  return Result;
}
