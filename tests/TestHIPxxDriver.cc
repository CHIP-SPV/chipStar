#include <catch2/catch_test_macros.hpp>

#include "HIPxxBackend.hh"
#include "HIPxxDriver.hh"

TEST_CASE("Backend Initialization", "[HIPxxBackend]") {
  initialize();
  REQUIRE(Backend->hipxx_contexts.size() == 1);
}
