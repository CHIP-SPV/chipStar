#include <catch2/catch_test_macros.hpp>

#include "CHIPBackend.hh"
#include "CHIPDriver.hh"

TEST_CASE("OpenCL - Backend init creates context, device, and a queue",
          "[CHIPBackend]") {
  CHIPInitialize("OPENCL");
  REQUIRE(Backend->hipxx_contexts.size() == 1);
  REQUIRE(Backend->hipxx_devices.size() == 1);
  REQUIRE(Backend->hipxx_queues.size() == 1);
}

TEST_CASE("Level0 - Backend init creates context, device, and a queue",
          "[CHIPBackend]") {
  CHIPInitialize("LEVEL0");
  REQUIRE(Backend->hipxx_contexts.size() == 1);
  REQUIRE(Backend->hipxx_devices.size() == 1);
  REQUIRE(Backend->hipxx_queues.size() == 1);
}

// TODO test if initialize() takes in string arg properly
