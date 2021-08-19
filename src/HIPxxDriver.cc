/**
 * @file HIPxxDriver.cc
 * @author Paulius Velesko (pvelesko@gmail.com)
 * @brief Definitions of extern declared functions and objects in HIPxxDriver.hh
 * Initializing the HIPxx runtime with backend selection through HIPXX_BE
 * environment variable.
 * @version 0.1
 * @date 2021-08-19
 *
 * @copyright Copyright (c) 2021
 *
 */
#include "HIPxxDriver.hh"

std::once_flag initialized;
HIPxxBackend* Backend;

void _initialize() {
  std::cout << "HIPxxDriver Initialize\n";
  // Get the current Backend Env Var
  Backend = new HIPxxBackendOpenCL();
  Backend->initialize();
};

void initialize() { std::call_once(initialized, &_initialize); }