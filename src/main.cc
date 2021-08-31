#include <iostream>

#include "HIPxxBackend.hh"
#include "HIPxxDriver.hh"
#include "include/hip/hip.hh"
void testf() { std::cout << "Test Function Executed\n"; }

int main() {
  std::cout << "Hello, World!\n";
  HIPxxInitialize();

  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);

  HIPxxExecItem ex;
  Backend->submit(&ex);
}