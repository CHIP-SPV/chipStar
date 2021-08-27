#include "logging.hh"

static int SpdlogWasSetup = 0;

void setupSpdlog() {
  if (SpdlogWasSetup)
    return;
  spdlog::set_default_logger(spdlog::stderr_color_mt("HIPLZ"));
  spdlog::set_level(spdlog::level::debug);
  spdlog::set_pattern("%n %^%l%$ [TID %t] [%E.%F] : %v");

  spdlog::level::level_enum spd_loglevel = spdlog::level::err;

  const char *loglevel = getenv("HIPLZ_LOGLEVEL");
  if (loglevel) {
    std::string level(loglevel);
    if (level == "debug")
      spd_loglevel = spdlog::level::debug;
    if (level == "info")
      spd_loglevel = spdlog::level::info;
    if (level == "warn")
      spd_loglevel = spdlog::level::warn;
    if (level == "err")
      spd_loglevel = spdlog::level::err;
    if (level == "crit")
      spd_loglevel = spdlog::level::critical;
    if (level == "off")
      spd_loglevel = spdlog::level::off;
  }

  spdlog::set_level(spd_loglevel);

  SpdlogWasSetup = 1;
}
