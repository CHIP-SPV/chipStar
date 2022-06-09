/**
 * @file CHIPDriver.hh
 * @author Paulius Velesko (pvelesko@gmail.com)
 * @brief Header defining global CHIP classes and functions such as
 * CHIPBackend type pointer Backend which gets initialized at the start of
 * execution.
 * @version 0.1
 * @date 2021-08-19
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef CHIP_DRIVER_H
#define CHIP_DRIVER_H
#include <iostream>
#include <mutex>

#include "CHIPBackend.hh"

/**
 * @brief
 * Global Backend pointer through which backend-specific operations are
 * performed
 */
extern CHIPBackend *Backend;

/**
 * @brief
 * Singleton backend initialization flag
 */
extern std::once_flag Initialized;
extern std::once_flag Uninitialized;

/**
 * @brief
 * Singleton backend initialization function outer wrapper
 */
extern void CHIPInitialize(std::string BackendStr = "");

/**
 * @brief
 * Singleton backend initialization function outer wrapper
 */
extern void CHIPUninitialize();

/**
 * @brief
 * Singleton backend initialization function called via std::call_once
 */
void CHIPInitializeCallOnce(std::string BackendStr = "");

/**
 * @brief
 * Singleton backend uninitialization function called via std::call_once
 */
void CHIPUninitializeCallOnce();


extern hipError_t CHIPReinitialize(const uintptr_t *NativeHandles, int NumHandles);

std::string read_env_var(std::string EnvVar, bool Lower);

#endif