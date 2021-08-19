#include "HIPxxDriver.hh"

std::once_flag initialized;

void _initialize() { std::cout << "HIPxxDriver Initialize\n"; };
void initialize() { std::call_once(initialized, &_initialize); }