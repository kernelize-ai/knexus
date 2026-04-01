
#include "cpu_schedule.h"

#include "cpu_runtime.h"

float CpuSchedule::getTime() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                               start_time)
      .count();
}

nxs_status CpuSchedule::run(nxs_int stream, nxs_uint run_settings) {
  nxs_uint settings = getSettings() | run_settings;

  if (settings & NXS_ExecutionSettings_Timing) {
    start_time = std::chrono::steady_clock::now();
  }

  for (auto cmd : getCommands()) {
    NXSLOG_INFO("runCommand - {}", static_cast<int>(cmd->getType()));
    auto status = cmd->runCommand(stream);
    if (!nxs_success(status)) return status;
  }

  if (settings & NXS_ExecutionSettings_Timing) {
    end_time = std::chrono::steady_clock::now();
  }
  return NXS_Success;
}

nxs_status CpuSchedule::release() {
  nxs_status status = Schedule::release();
  return status;
}
