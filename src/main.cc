#include <iostream>

#include "HIPxxBackend.hh"
#include "HIPxxDriver.hh"

void testf() { std::cout << "Test Function Executed\n"; }

int main() {
  std::cout << "Hello, World!\n";
  HIPxxInitialize();

  HIPxxExecItem ex;
  Backend->submit(&ex);
}