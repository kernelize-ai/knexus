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

// Nexus handles location via NEXUS_LOG_MODULE prefix only; avoid file:line in output.
#define SPDLOG_NO_SOURCE_LOC
#include <spdlog/spdlog.h>
#include <spdlog/logger.h>

namespace nexus {

class LogManager {
 public:

  template <spdlog::level::level_enum Tlevel>
  static inline void log(const std::string& module, const char *message) {
    auto &inst = getInstance();
    if (inst.isOpen()) {
      inst.logger()->log(Tlevel, message, module);
    }
  }
  template <spdlog::level::level_enum Tlevel, typename... Args>
  static inline void log(const std::string& module, const char *message, Args... args) {
    auto &inst = getInstance();
    if (inst.isOpen()) {
      inst.logger()->log(Tlevel, message, module, std::forward<Args>(args)...);
    }
  }

  // Disable copying and moving
  LogManager(const LogManager&) = delete;
  LogManager& operator=(const LogManager&) = delete;
  LogManager(LogManager&&) = delete;
  LogManager& operator=(LogManager&&) = delete;

  ~LogManager();

 private:
  static LogManager& getInstance() {
    static LogManager instance;
    return instance;
  }

  inline auto logger() const noexcept -> spdlog::logger* { return impl_->logger.get(); }

  void setOpen(bool open);
  bool isOpen() const;
  void setLogFile(const std::string& filename);

  LogManager();

  void resetLogger();
  void openFile(const std::string& filename);
  void openStdout();
  void installDisabledLogger();

  struct Impl {
    std::shared_ptr<spdlog::logger> logger;
  };
  std::unique_ptr<Impl> impl_;
};

}  // namespace nexus

#endif  // NEXUS_LOG_MANAGER_H
