#include <iostream>

#include "CHIPBackend.hh"
#include "CHIPDriver.hh"
#include "include/hip/hip.hh"
void testf() { std::cout << "Test Function Executed\n"; }

int main() {
  std::cout << "Hello, World!\n";
  CHIPInitialize();

  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);

  CHIPExecItem ex;
  Backend->submit(&ex);
}