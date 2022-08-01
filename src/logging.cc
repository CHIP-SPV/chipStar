#include "logging.hh"

#include <iostream>
#include <mutex>

std::once_flag SpdlogWasSetup;

void setupSpdlog() { std::call_once(SpdlogWasSetup, &_setupSpdlog); }

void _setupSpdlog() {
  spdlog::set_default_logger(spdlog::stderr_color_mt("CHIP"));
  spdlog::set_pattern("%n %^%l%$ [TID %t] [%E.%F] : %v");

  spdlog::level::level_enum SpdLogLevel =
#if (SPDLOG_ACTIVE_LEVEL == SPDLOG_LEVEL_TRACE)
      spdlog::level::debug;
#else
      spdlog::level::warn;
#endif

  const char *LogLevel = getenv("CHIP_LOGLEVEL");
  if (LogLevel) {
    // std::cout << "CHIP_LOGLEVEL=" << loglevel << "\n";
    std::string Level(LogLevel);
    if (Level == "trace")
      SpdLogLevel = spdlog::level::trace;
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
  }

  spdlog::set_level(SpdLogLevel);
}
