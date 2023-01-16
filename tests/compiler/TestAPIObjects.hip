// Test an user defined pointer derived from ihipEvent_t found in a
// hipified CUDA application.
#include <hip/hip_runtime.h>
#include <vector>

struct ihipEvent_t; // Forward declaration

std::vector<ihipEvent_t *> testIHipEvent() {
  ihipEvent_t *Event;
  hipEventCreate(&Event);
  std::vector<ihipEvent_t *> Result = {Event};
  return Result;
}
