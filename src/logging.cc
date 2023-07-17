/*
 * Copyright (c) 2021-22 chipStar developers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

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
