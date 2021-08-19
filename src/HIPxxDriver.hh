#ifndef HIPXX_DRIVER_H
#define HIPXX_DRIVER_H
#include <iostream>
#include <mutex>

extern std::once_flag initialized;
extern void initialize();
extern void _initialize();

#endif