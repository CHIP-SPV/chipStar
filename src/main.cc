#include <iostream>

#include "HIPxxBackend.hh"
#include "HIPxxDriver.hh"

void testf() { std::cout << "Test Function Executed\n"; }

int main() {
  std::cout << "Hello, World!\n";
  initialize();
  initialize();

  HIPxxExecItem ex;
  Backend->submit(&ex);
}