#ifndef NEXUS_LOG_MANAGER_H
#define NEXUS_LOG_MANAGER_H

#include <memory>
#include <string>

#ifndef SPDLOG_ACTIVE_LEVEL
#ifndef NDEBUG
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#else
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO
#endif
#endif

#ifndef NEXUS_LOG_NAME
#define NEXUS_LOG_NAME "nexus"
#endif

// Nexus handles location via NEXUS_LOG_MODULE prefix only; avoid file:line in output.
#define SPDLOG_NO_SOURCE_LOC
#include <spdlog/spdlog.h>
#include <spdlog/logger.h>

namespace nexus {

class LogManager {
 public:
  /// Pads `module` to `module_width` characters, then wraps with ANSI colors when logging to a
  /// color terminal (not when logging to a file). If `color_ansi` is non-null and non-empty, it is
  /// used as the SGR foreground prefix (e.g. "\033[32m"); otherwise nxs-api vs core defaults apply.
  static std::string format_module_column(const std::string& module, int module_width,
                                         const char* color_ansi);

  /// True when logging is enabled and the active sink is not at `level::off`.
  bool isOpen() const;

  // Disable copying and moving
  LogManager(const LogManager&) = delete;
  LogManager& operator=(const LogManager&) = delete;
  LogManager(LogManager&&) = delete;
  LogManager& operator=(LogManager&&) = delete;

  ~LogManager();

  static LogManager& getInstance() {
    static LogManager instance(NEXUS_LOG_NAME);
    return instance;
  }

  inline auto logger() const noexcept -> spdlog::logger* { return impl_->logger.get(); }
 private:

  void setOpen(bool open);
  void setLogFile(const std::string& filename);

  LogManager(const char *name);

  void resetLogger();
  void openFile(const std::string& filename);
  void openStdout();
  void installDisabledLogger();

  struct Impl {
    std::shared_ptr<spdlog::logger> logger;
    bool color_module{false};
  };
  std::unique_ptr<Impl> impl_;
  std::string name;
};

}  // namespace nexus

#endif  // NEXUS_LOG_MANAGER_H
