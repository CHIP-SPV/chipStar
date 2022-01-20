#ifndef LOGGING_H
#define LOGGING_H

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#if !defined(SPDLOG_ACTIVE_LEVEL)
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#endif

extern std::once_flag SpdlogWasSetup;
extern void setupSpdlog();
extern void _setupSpdlog();

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_TRACE
template <typename... TypeArgs>
void logTrace(const char* Fmt, const TypeArgs&... Args) {
  setupSpdlog();
  spdlog::trace(Fmt, std::forward<const TypeArgs>(Args)...);
}
#else
#define logTrace(...) void(0)
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG
template <typename... TypeArgs>
void logDebug(const char* Fmt, const TypeArgs&... Args) {
  setupSpdlog();
  spdlog::debug(Fmt, std::forward<const TypeArgs>(Args)...);
}
#else
#define logDebug(...) void(0)
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_INFO
template <typename... TypeArgs>
void logInfo(const char* Fmt, const TypeArgs&... Args) {
  setupSpdlog();
  spdlog::info(Fmt, std::forward<const TypeArgs>(Args)...);
}
#else
#define logInfo(...) void(0)
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_WARN
template <typename... TypeArgs>
void logWarn(const char* Fmt, const TypeArgs&... Args) {
  setupSpdlog();
  spdlog::warn(Fmt, std::forward<const TypeArgs>(Args)...);
}
#else
#define logWarn(...) void(0)
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_ERROR
template <typename... TypeArgs>
void logError(const char* Fmt, const TypeArgs&... Args) {
  setupSpdlog();
  spdlog::error(Fmt, std::forward<const TypeArgs>(Args)...);
}
#else
#define logError(...) void(0)
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_CRITICAL
template <typename... TypeArgs>
void logCritical(const char* Fmt, const TypeArgs&... Args) {
  setupSpdlog();
  spdlog::critical(Fmt, std::forward<const TypeArgs>(Args)...);
}
#else
#define logCritical(...) void(0)
#endif

#endif