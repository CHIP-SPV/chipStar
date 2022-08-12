#ifndef CHIP_BACKENDS_H
#define CHIP_BACKENDS_H

#ifdef HAVE_LEVEL0
#include "Level0/CHIPBackendLevel0.hh"
#endif
#ifdef HAVE_OPENCL
#include "OpenCL/CHIPBackendOpenCL.hh"
#endif

#endif