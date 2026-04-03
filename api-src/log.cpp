#include <nexus/log_manager.h>

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

namespace {

bool noColorEnv() {
  static const char* s = std::getenv("NO_COLOR");
  return s != nullptr && s[0] != '\0';
}

// NEXUS_LOG_LEVEL: when NEXUS_LOG_ENABLE is set, minimum level to emit (stderr/stdout/file).
// Names are spdlog-style: trace, debug, info, warning/warn, error/err, critical, off (case-insensitive).
// Or an integer 0–6 matching spdlog::level_enum (trace..off). Unset defaults to trace.
// Unknown names are treated like off (spdlog::level::from_str).
spdlog::level::level_enum levelFromEnv() {
  const char* s = std::getenv("NEXUS_LOG_LEVEL");
  if (!s || s[0] == '\0') {
    return spdlog::level::trace;
  }
  bool all_digits = true;
  for (const char* p = s; *p != '\0'; ++p) {
    if (*p < '0' || *p > '9') {
      all_digits = false;
      break;
    }
  }
  if (all_digits) {
    const int v = std::atoi(s);
    if (v >= 0 && v <= static_cast<int>(spdlog::level::off)) {
      return static_cast<spdlog::level::level_enum>(v);
    }
  }
  return spdlog::level::from_str(std::string(s));
}

std::shared_ptr<spdlog::logger> makeNexusLogger(const char *name, std::shared_ptr<spdlog::sinks::sink> sink) {
  auto logger = std::make_shared<spdlog::logger>(name, std::move(sink));
  logger->set_pattern("\x1b[90m[%Y-%m-%d %H:%M:%S.%e]\x1b[0m %^%-4!l%$ | %v");
  logger->set_level(levelFromEnv());
  spdlog::register_logger(logger);
  spdlog::set_default_logger(logger);
  return logger;
}

std::shared_ptr<spdlog::logger> makeDisabledLogger(const char *name) {
  auto sink = std::make_shared<spdlog::sinks::null_sink_mt>();
  auto logger = std::make_shared<spdlog::logger>(name, std::move(sink));
  logger->set_level(spdlog::level::off);
  spdlog::register_logger(logger);
  spdlog::set_default_logger(logger);
  return logger;
}

}  // namespace

namespace nexus {

std::string LogManager::format_module_column(const std::string& module, int module_width,
                                            const char* color_ansi) {
  int w = module_width;
  if (w < 1) {
    w = 10;
  }

  std::string padded = module;
  if (static_cast<int>(padded.size()) < w) {
    padded.append(static_cast<size_t>(w) - padded.size(), ' ');
  } else if (static_cast<int>(padded.size()) > w) {
    padded.resize(static_cast<size_t>(w));
  }

  auto& inst = getInstance();
  if (!inst.impl_->color_module || noColorEnv()) {
    return padded;
  }

  const char* fore = nullptr;
  if (color_ansi != nullptr && color_ansi[0] != '\0') {
    fore = color_ansi;
  } else {
    const bool nxs_api = module.size() >= 7 && module.compare(0, 7, "nxs-api") == 0;
    fore = nxs_api ? "\033[36m" : "\033[90m";
  }
  return std::string(fore) + padded + "\033[0m";
}

LogManager::LogManager(const char *_name) : impl_(std::make_unique<Impl>()), name(_name) {
  const char* enableEnv = std::getenv("NEXUS_LOG_ENABLE");
  const bool enable = enableEnv ? (std::atoi(enableEnv) != 0) : false;
  setOpen(enable);
}

LogManager::~LogManager() { setOpen(false); }

void LogManager::resetLogger() {
  if (impl_->logger) {
    impl_->logger->flush();
    spdlog::drop(name);
    impl_->logger.reset();
  }
}

void LogManager::installDisabledLogger() {
  resetLogger();
  impl_->color_module = false;
  impl_->logger = makeDisabledLogger(name.c_str());
}

void LogManager::openFile(const std::string& filename) {
  resetLogger();
  impl_->color_module = false;
  try {
    auto sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename, false);
    impl_->logger = makeNexusLogger(name.c_str(), std::move(sink));
  } catch (const spdlog::spdlog_ex& ex) {
    std::cerr << "Failed to open log file: " << filename << " (" << ex.what() << ")\n";
    impl_->logger.reset();
    installDisabledLogger();
  }
}

void LogManager::openStdout() {
  resetLogger();
  impl_->color_module = true;
  try {
    auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    impl_->logger = makeNexusLogger(name.c_str(), std::move(sink));
  } catch (const spdlog::spdlog_ex& ex) {
    std::cerr << "Failed to create stdout logger (" << ex.what() << ")\n";
    impl_->logger.reset();
    installDisabledLogger();
  }
}

void LogManager::setOpen(bool open) {
  if (open) {
    const char* logPath = std::getenv("NEXUS_LOG_FILE");
    if (logPath && logPath[0] != '\0') {
      openFile(logPath);
    } else {
      openStdout();
    }
  } else {
    installDisabledLogger();
  }
}

bool LogManager::isOpen() const {
  return impl_->logger && impl_->logger->level() != spdlog::level::off;
}

void LogManager::setLogFile(const std::string& filename) { openFile(filename); }

}  // namespace nexus
