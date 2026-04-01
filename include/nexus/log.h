#ifndef NEXUS_LOG_H
#define NEXUS_LOG_H

#include <nexus/log_manager.h>

#ifndef NEXUS_LOG_MODULE
#define NEXUS_LOG_MODULE "nexus"
#endif

#ifndef NEXUS_LOG_PADDING
#define NEXUS_LOG_PADDING "10"
#endif

// fmt + __VA_ARGS__ only — define NEXUS_LOG_MODULE per .cpp before the first #include of
// this header (nexus-api pulls log_manager.h only, not these macros).
#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_TRACE
#define NXSLOG_TRACE(fmt, ...)                                                     \
  ::nexus::LogManager::log<spdlog::level::trace>(NEXUS_LOG_MODULE, "{:" NEXUS_LOG_PADDING "s} | " fmt, ##__VA_ARGS__)
#else
#define NXSLOG_TRACE(fmt, ...) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG
#define NXSLOG_DEBUG(fmt, ...)                                                     \
  ::nexus::LogManager::log<spdlog::level::debug>(NEXUS_LOG_MODULE, "{:" NEXUS_LOG_PADDING "s} | " fmt, ##__VA_ARGS__)
#else
#define NXSLOG_DEBUG(fmt, ...) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_INFO
#define NXSLOG_INFO(fmt, ...)                                                      \
  ::nexus::LogManager::log<spdlog::level::info>(NEXUS_LOG_MODULE, "{:" NEXUS_LOG_PADDING "s} | " fmt, ##__VA_ARGS__)
#else
#define NXSLOG_INFO(fmt, ...) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_WARN
#define NXSLOG_WARN(fmt, ...)                                                      \
  ::nexus::LogManager::log<spdlog::level::warn>(NEXUS_LOG_MODULE, "{:" NEXUS_LOG_PADDING "s} | " fmt, ##__VA_ARGS__)
#else
#define NXSLOG_WARN(fmt, ...) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_ERROR
#define NXSLOG_ERROR(fmt, ...)                                                     \
  ::nexus::LogManager::log<spdlog::level::err>(NEXUS_LOG_MODULE, "{:" NEXUS_LOG_PADDING "s} | " fmt, ##__VA_ARGS__)
#else
#define NXSLOG_ERROR(fmt, ...) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_CRITICAL
#define NXSLOG_CRITICAL(fmt, ...)                                                  \
  ::nexus::LogManager::log<spdlog::level::critical>(NEXUS_LOG_MODULE, "{:" NEXUS_LOG_PADDING "s} | " fmt, ##__VA_ARGS__)
#else
#define NXSLOG_CRITICAL(fmt, ...) (void)0
#endif

#endif  // NEXUS_LOG_H
