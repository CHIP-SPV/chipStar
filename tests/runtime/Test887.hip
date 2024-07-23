#include <cstdio>
#include <hip/hip_runtime.h>

class Test {
public:
  float *output;
  Test();
  ~Test();
};

// Constructor
Test::Test() {
  output = NULL;
  hipMalloc(&output, sizeof(float) * 100);
}

// Destructor
Test::~Test() {
  printf("destructor called\n");
  hipFree(output);
}

Test test = Test();

int main() {
  printf("testing\n");
  return 0;
}