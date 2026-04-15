#ifndef NEXUS_LOG_H
#define NEXUS_LOG_H

#include <nexus/log_manager.h>

#ifndef NEXUS_LOG_MODULE
#define NEXUS_LOG_MODULE "nexus"
#endif

#ifndef NEXUS_LOG_PADDING
#define NEXUS_LOG_PADDING 10
#endif

#ifndef NEXUS_LOG_MODULE_COLOR
#define NEXUS_LOG_MODULE_COLOR ((const char*)0)
#endif

// fmt + __VA_ARGS__ only — define NEXUS_LOG_MODULE per .cpp before the first #include of
// this header (nexus-api pulls log_manager.h only, not these macros).
// Optional: NEXUS_LOG_MODULE_COLOR = ANSI SGR prefix string (e.g. "\033[32m"), or 0 for defaults.
// Log call is expanded here so the format string is a literal at the call site (no SPDLOG_FMT_RUNTIME).
#define NXSLOG_MACRO(level, fmt, ...)                                                                \
  do {                                                                                               \
    auto& _nxslog_inst = ::nexus::LogManager::getInstance();                                         \
    if (_nxslog_inst.isOpen()) {                                                                     \
      _nxslog_inst.logger()->log((level), "{} | \x1b[40m\x1b[97m" fmt "\x1b[0m",                     \
          ::nexus::LogManager::format_module_column(NEXUS_LOG_MODULE, NEXUS_LOG_PADDING,             \
                                                     NEXUS_LOG_MODULE_COLOR),                        \
          ##__VA_ARGS__);                                                                            \
    }                                                                                                \
  } while (0)


#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_TRACE
#define NXSLOG_TRACE(fmt, ...)                                                     \
  NXSLOG_MACRO(spdlog::level::trace, fmt, ##__VA_ARGS__)
#else
#define NXSLOG_TRACE(fmt, ...) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG
#define NXSLOG_DEBUG(fmt, ...)                                                     \
  NXSLOG_MACRO(spdlog::level::debug, fmt, ##__VA_ARGS__)
#else
#define NXSLOG_DEBUG(fmt, ...) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_INFO
#define NXSLOG_INFO(fmt, ...)                                                      \
  NXSLOG_MACRO(spdlog::level::info, fmt, ##__VA_ARGS__)
#else
#define NXSLOG_INFO(fmt, ...) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_WARN
#define NXSLOG_WARN(fmt, ...)                                                      \
  NXSLOG_MACRO(spdlog::level::warn, fmt, ##__VA_ARGS__)
#else
#define NXSLOG_WARN(fmt, ...) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_ERROR
#define NXSLOG_ERROR(fmt, ...)                                                     \
  NXSLOG_MACRO(spdlog::level::err, fmt, ##__VA_ARGS__)
#else
#define NXSLOG_ERROR(fmt, ...) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_CRITICAL
#define NXSLOG_CRITICAL(fmt, ...)                                                  \
  NXSLOG_MACRO(spdlog::level::critical, fmt, ##__VA_ARGS__)
#else
#define NXSLOG_CRITICAL(fmt, ...) (void)0
#endif

#endif  // NEXUS_LOG_H
