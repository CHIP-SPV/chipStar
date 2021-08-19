#ifndef HIPXX_DRIVER_H
#define HIPXX_DRIVER_H
#include <iostream>
#include <mutex>

#include "HIPxxBackend.hh"
#include "backend/backends.hh"

extern HIPxxBackend* Backend;
extern std::once_flag initialized;
extern void initialize();
extern void _initialize();

#endif