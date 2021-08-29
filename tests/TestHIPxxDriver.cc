#include <catch2/catch_test_macros.hpp>

#include "HIPxxBackend.hh"
#include "HIPxxDriver.hh"

TEST_CASE("Backend init creates context, device, and a queue",
          "[HIPxxBackend]") {
  HIPxxInitialize();
  REQUIRE(Backend->hipxx_contexts.size() == 1);
  REQUIRE(Backend->hipxx_devices.size() == 1);
  REQUIRE(Backend->hipxx_queues.size() == 1);
}

// TODO test if initialize() takes in string arg properly
