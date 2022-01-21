#include "logging.hh"
#include <iostream>
#include <mutex>

std::once_flag SpdlogWasSetup;

void setupSpdlog() { std::call_once(SpdlogWasSetup, &_setupSpdlog); }

void _setupSpdlog() {
  spdlog::set_default_logger(spdlog::stderr_color_mt("CHIP"));
  spdlog::set_level(spdlog::level::debug);
  spdlog::set_pattern("%n %^%l%$ [TID %t] [%E.%F] : %v");

  spdlog::level::level_enum SpdLogLevel;

  const char *LogLevel = getenv("CHIP_LOGLEVEL");
  if (LogLevel) {
    // std::cout << "CHIP_LOGLEVEL=" << loglevel << "\n";
    std::string Level(LogLevel);
    if (Level == "debug")
      SpdLogLevel = spdlog::level::debug;
    if (Level == "info")
      SpdLogLevel = spdlog::level::info;
    if (Level == "warn")
      SpdLogLevel = spdlog::level::warn;
    if (Level == "err")
      SpdLogLevel = spdlog::level::err;
    if (Level == "crit")
      SpdLogLevel = spdlog::level::critical;
    if (Level == "off")
      SpdLogLevel = spdlog::level::off;
  } else {
    // std::cout << "CHIP_LOGLEVEL was not set. Default to trace\n";
    SpdLogLevel = spdlog::level::trace;
  }

  spdlog::set_level(SpdLogLevel);
}
