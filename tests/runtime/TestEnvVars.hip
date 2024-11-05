#include <hip/hip_runtime.h>
#include "CHIPDriver.hh"
#include <iostream>
#include <cassert>
#include <cstdlib>
#include <limits>
#include <type_traits>

// Helper function to set environment variables
void setEnvVar(const char* name, const char* value) {
    if (setenv(name, value, 1) != 0) {
        std::cerr << "Failed to set environment variable: " << name << std::endl;
        exit(1);
    }
}

// Helper function to unset environment variables
void unsetEnvVar(const char* name) {
    if (unsetenv(name) != 0) {
        std::cerr << "Failed to unset environment variable: " << name << std::endl;
        exit(1);
    }
}

// Helper function to check if two values are equal and print an error if not
template<typename T>
void assertEqual(const T& expected, const T& actual, const char* varName) {
    if (expected != actual) {
        std::cerr << "Assertion failed for " << varName << ": expected ";
        if constexpr (std::is_same<T, std::string>::value) {
            std::cerr << '\"' << expected << '\"';
        } else {
            std::cerr << expected;
        }
        std::cerr << ", got ";
        if constexpr (std::is_same<T, std::string>::value) {
            std::cerr << '\"' << actual << '\"';
        } else {
            std::cerr << actual;
        }
        std::cerr << std::endl;
        exit(1);
    }
}

// Define individual test functions for clarity
void testEnvironmentVariableParsing() {
    // Test environment variable parsing
    {
        // Set environment variables
        CHIPInitialize();
        setEnvVar("CHIP_PLATFORM", "2");
        setEnvVar("CHIP_DEVICE_TYPE", "gpu");
        setEnvVar("CHIP_DEVICE", "1");
        setEnvVar("CHIP_BE", "level0");
        setEnvVar("CHIP_DUMP_SPIRV", "1");
        setEnvVar("CHIP_SKIP_UNINIT", "on");
        setEnvVar("CHIP_LAZY_JIT", "1");
        setEnvVar("CHIP_JIT_FLAGS", "-O2");
        setEnvVar("CHIP_L0_COLLECT_EVENTS_TIMEOUT", "1000");
        setEnvVar("CHIP_L0_EVENT_TIMEOUT", "2000");
        setEnvVar("CHIP_OCL_DISABLE_QUEUE_PROFILING", "1");
        setEnvVar("CHIP_OCL_USE_ALLOC_STRATEGY", "custom_strategy");

        // Create an EnvVars instance to parse the environment variables
        EnvVars envVars;

        // Check if the parsed values match the set environment variables
        assertEqual(2, envVars.getPlatformIdx(), "CHIP_PLATFORM");
        assertEqual(static_cast<int>(DeviceType::GPU), static_cast<int>(envVars.getDevice().getType()), "CHIP_DEVICE_TYPE");
        assertEqual(1, envVars.getDeviceIdx(), "CHIP_DEVICE");
        assertEqual(static_cast<int>(BackendType::Level0), static_cast<int>(envVars.getBackend().getType()), "CHIP_BE");
        assertEqual(true, envVars.getDumpSpirv(), "CHIP_DUMP_SPIRV");
        assertEqual(true, envVars.getSkipUninit(), "CHIP_SKIP_UNINIT");
        assertEqual(true, envVars.getLazyJit(), "CHIP_LAZY_JIT");
        assertEqual(std::string("-O2"), envVars.getJitFlags(), "CHIP_JIT_FLAGS");
        assertEqual(1000, envVars.getL0CollectEventsTimeout(), "CHIP_L0_COLLECT_EVENTS_TIMEOUT");
        assertEqual(static_cast<unsigned long>(2000), envVars.getL0EventTimeout(), "CHIP_L0_EVENT_TIMEOUT");
        assertEqual(true, envVars.getOCLDisableQueueProfiling(), "CHIP_OCL_DISABLE_QUEUE_PROFILING");
        assertEqual(std::string("custom_strategy"), envVars.getOclUseAllocStrategy().value_or(""), "CHIP_OCL_USE_ALLOC_STRATEGY");

        // Unset environment variables
        unsetEnvVar("CHIP_PLATFORM");
        unsetEnvVar("CHIP_DEVICE_TYPE");
        unsetEnvVar("CHIP_DEVICE");
        unsetEnvVar("CHIP_BE");
        unsetEnvVar("CHIP_DUMP_SPIRV");
        unsetEnvVar("CHIP_SKIP_UNINIT");
        unsetEnvVar("CHIP_LAZY_JIT");
        unsetEnvVar("CHIP_JIT_FLAGS");
        unsetEnvVar("CHIP_L0_COLLECT_EVENTS_TIMEOUT");
        unsetEnvVar("CHIP_L0_EVENT_TIMEOUT");
        unsetEnvVar("CHIP_OCL_DISABLE_QUEUE_PROFILING");
        unsetEnvVar("CHIP_OCL_USE_ALLOC_STRATEGY");
    }
}

void testDefaultValues() {
    // Test default values when environment variables are not set
    {
        EnvVars envVars;

        // Check if the default values are used when environment variables are not set
        assertEqual(0, envVars.getPlatformIdx(), "Default CHIP_PLATFORM");
        assertEqual(static_cast<int>(DeviceType::GPU), static_cast<int>(envVars.getDevice().getType()), "Default CHIP_DEVICE_TYPE");
        assertEqual(0, envVars.getDeviceIdx(), "Default CHIP_DEVICE");
        assertEqual(BackendType::OpenCL, envVars.getBackend().getType(), "Default CHIP_BE");
        assertEqual(false, envVars.getDumpSpirv(), "Default CHIP_DUMP_SPIRV");
        assertEqual(false, envVars.getSkipUninit(), "Default CHIP_SKIP_UNINIT");
        assertEqual(true, envVars.getLazyJit(), "Default CHIP_LAZY_JIT");
        assertEqual(std::string(CHIP_DEFAULT_JIT_FLAGS), envVars.getJitFlags(), "Default CHIP_JIT_FLAGS");
        assertEqual(0, envVars.getL0CollectEventsTimeout(), "Default CHIP_L0_COLLECT_EVENTS_TIMEOUT");
        assertEqual(UINT64_MAX, envVars.getL0EventTimeout(), "Default CHIP_L0_EVENT_TIMEOUT");
        assertEqual(false, envVars.getOCLDisableQueueProfiling(), "Default CHIP_OCL_DISABLE_QUEUE_PROFILING");
        assertEqual(false, envVars.getOclUseAllocStrategy().has_value(), "Default CHIP_OCL_USE_ALLOC_STRATEGY");
    }
}

void testPartialEnvironmentVariableSetting() {
    // Test partial environment variable setting
    {
        setEnvVar("CHIP_PLATFORM", "1");
        setEnvVar("CHIP_DEVICE_TYPE", "cpu");
        setEnvVar("CHIP_BE", "opencl");

        EnvVars envVars;

        assertEqual(1, envVars.getPlatformIdx(), "Partial CHIP_PLATFORM");
        assertEqual(static_cast<int>(DeviceType::CPU), static_cast<int>(envVars.getDevice().getType()), "Partial CHIP_DEVICE_TYPE");
        assertEqual(0, envVars.getDeviceIdx(), "Default CHIP_DEVICE when not set");
        assertEqual(static_cast<int>(BackendType::OpenCL), static_cast<int>(envVars.getBackend().getType()), "Partial CHIP_BE");

        unsetEnvVar("CHIP_PLATFORM");
        unsetEnvVar("CHIP_DEVICE_TYPE");
        unsetEnvVar("CHIP_BE");
    }
}

void testEnvironmentVariableOverriding() {
    // Test environment variable overriding
    {
        setEnvVar("CHIP_PLATFORM", "2");
        setEnvVar("CHIP_PLATFORM", "3");
        setEnvVar("CHIP_DEVICE_TYPE", "gpu");
        setEnvVar("CHIP_DEVICE_TYPE", "cpu");

        EnvVars envVars;

        assertEqual(3, envVars.getPlatformIdx(), "Overridden CHIP_PLATFORM");
        assertEqual(static_cast<int>(DeviceType::CPU), static_cast<int>(envVars.getDevice().getType()), "Overridden CHIP_DEVICE_TYPE");

        unsetEnvVar("CHIP_PLATFORM");
        unsetEnvVar("CHIP_DEVICE_TYPE");
    }
}

void testInvalidValues() {
    // Test invalid and edge case values
    {
        bool exceptionThrown;

        // Test invalid CHIP_PLATFORM
        setEnvVar("CHIP_PLATFORM", "invalid");
        exceptionThrown = false;
        try {
            EnvVars envVars;
        } catch (const CHIPError&) {
            exceptionThrown = true;
        }
        assert(exceptionThrown && "Expected CHIPError exception due to invalid CHIP_PLATFORM");
        unsetEnvVar("CHIP_PLATFORM");

        // Test invalid CHIP_DEVICE_TYPE
        setEnvVar("CHIP_DEVICE_TYPE", "invalid");
        exceptionThrown = false;
        try {
            EnvVars envVars;
        } catch (const CHIPError&) {
            exceptionThrown = true;
        }
        assert(exceptionThrown && "Expected CHIPError exception due to invalid CHIP_DEVICE_TYPE");
        unsetEnvVar("CHIP_DEVICE_TYPE");

        // Test invalid CHIP_DEVICE
        setEnvVar("CHIP_DEVICE", "-1");
        exceptionThrown = false;
        try {
            EnvVars envVars;
        } catch (const CHIPError&) {
            exceptionThrown = true;
        }
        assert(exceptionThrown && "Expected CHIPError exception due to invalid CHIP_DEVICE");
        unsetEnvVar("CHIP_DEVICE");

        // Test invalid CHIP_BE
        setEnvVar("CHIP_BE", "invalid");
        exceptionThrown = false;
        try {
            EnvVars envVars;
        } catch (const CHIPError&) {
            exceptionThrown = true;
        }
        assert(exceptionThrown && "Expected CHIPError exception due to invalid CHIP_BE");
        unsetEnvVar("CHIP_BE");

        // Test negative CHIP_L0_COLLECT_EVENTS_TIMEOUT
        setEnvVar("CHIP_L0_COLLECT_EVENTS_TIMEOUT", "-1");
        exceptionThrown = false;
        try {
            EnvVars envVars;
        } catch (const CHIPError&) {
            exceptionThrown = true;
        }
        assert(exceptionThrown && "Expected CHIPError exception due to invalid CHIP_L0_COLLECT_EVENTS_TIMEOUT");
        unsetEnvVar("CHIP_L0_COLLECT_EVENTS_TIMEOUT");
    }
}

int main() {
    try {
        // Refactor test cases into separate functions for better clarity
        testEnvironmentVariableParsing();
        testDefaultValues();
        testPartialEnvironmentVariableSetting();
        testEnvironmentVariableOverriding();
        testInvalidValues();

        std::cout << "PASSED!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAILED!" << std::endl;
        return 1;
    }
}
